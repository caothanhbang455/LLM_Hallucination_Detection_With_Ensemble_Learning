# src/fold_train.py
import os, time, json, joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
import torch.nn.functional as F

from .config import cfg
from .utils import validate_models, set_repro, seed_worker
from .data import HalluDataset, collate_fn_train, collate_fn_infer, build_tokenizer, LabelCodec, make_sampler, kfold_splits
from .modeling import HalluModel
from .engine import train_one_epoch, eval_one_epoch, collect_logits_and_labels
from .calibration import fit_temperature_from_logits

@torch.no_grad()
def _aggregate_windows_mean(orig_index: torch.Tensor, probs: np.ndarray) -> np.ndarray:
    idx = orig_index.cpu().numpy().astype(int)
    n = int(idx.max()) + 1 if len(idx) > 0 else 0
    C = probs.shape[1] if probs.size else 0
    out = np.zeros((n, C), dtype=np.float64)
    cnt = np.zeros((n, 1), dtype=np.float64)
    for i, p in zip(idx, probs):
        out[i] += p
        cnt[i] += 1.0
    cnt[cnt == 0] = 1.0
    out /= cnt
    return out

def _save_ckpt(model, model_cfg, out_path):
    torch.save({
        "state_dict": model.state_dict(),
        "pretrained": model_cfg["pretrained"],
        "label_names": cfg.label_names,
        "dropout": float(cfg.dropout),
        "proj_dim": int(getattr(cfg, "proj_dim", 128)),
    }, out_path)

def _collect_val_probs(model, tokenizer, val_df, device, T, model_cfg):
    codec = LabelCodec(cfg.label_names)
    max_len = int(model_cfg.get("text_max_len", cfg.text_max_len))
    stride = int(model_cfg.get("doc_stride", cfg.doc_stride))
    seg_vi = bool(model_cfg.get("segment_vi", False))

    ds = HalluDataset(val_df, tokenizer, max_len, codec, has_labels=True,
                      doc_stride=stride, sliding_windows=True, segment_vi=seg_vi)
    dl = DataLoader(ds, batch_size=cfg.val_batch_size, shuffle=False,
                    collate_fn=collate_fn_infer, num_workers=0,
                    pin_memory=(device.type == "cuda"))

    model.eval()
    logits_all, orig_idx_all = [], []
    for b in dl:
        ids = b["input_ids"].to(device, non_blocking=True)
        mask = b["attention_mask"].to(device, non_blocking=True)
        logits = model(ids, mask)
        logits_all.append(logits.detach().cpu())
        orig_idx_all.append(b["orig_index"])
    logits_all = torch.cat(logits_all, 0)
    orig_idx_all = torch.cat(orig_idx_all, 0)

    probs_win = torch.softmax(logits_all / max(float(T), 1e-6), dim=-1).cpu().numpy()
    probs_ex = _aggregate_windows_mean(orig_idx_all, probs_win)
    labels_ex = val_df["label"].values
    return probs_ex, labels_ex

def train_model_on_fold(model_cfg, fold_idx, train_df, val_df, out_dir, temps_weights_path):
    # Reproducible (but not strictly deterministic) fold seeding
    set_repro(cfg.seed + fold_idx)
    device = torch.device(cfg.device)
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed + fold_idx)

    codec = LabelCodec(cfg.label_names)
    max_len = int(model_cfg.get("text_max_len", cfg.text_max_len))
    stride = int(model_cfg.get("doc_stride", cfg.doc_stride))
    use_fast = bool(model_cfg.get("use_fast", False))
    seg_vi = bool(model_cfg.get("segment_vi", False))
    tokenizer = build_tokenizer(model_cfg["pretrained"], use_fast)

    train_ds = HalluDataset(train_df, tokenizer, max_len, codec, has_labels=True,
                            doc_stride=stride, sliding_windows=True, segment_vi=seg_vi)
    val_ds   = HalluDataset(val_df,   tokenizer, max_len, codec, has_labels=True,
                            doc_stride=stride, sliding_windows=True, segment_vi=seg_vi)

    sampler = make_sampler(train_df, codec, gen=gen)
    train_loader = DataLoader(train_ds, batch_size=int(cfg.batch_size), sampler=sampler,
                              collate_fn=collate_fn_train, num_workers=2,
                              worker_init_fn=seed_worker, generator=gen,
                              pin_memory=(device.type=="cuda"), persistent_workers=True) # sửa
    val_loader   = DataLoader(val_ds,   batch_size=int(cfg.val_batch_size), shuffle=False,
                              collate_fn=collate_fn_train, num_workers=2,
                              worker_init_fn=seed_worker, generator=gen,
                              pin_memory=(device.type=="cuda"), persistent_workers=True) # sửa

    # model = HalluModel(model_cfg["pretrained"], num_labels=len(cfg.label_names),
    #                    dropout=cfg.dropout, proj_dim=int(getattr(cfg, "proj_dim", 128))).to(device)

    pooling_strategy = model_cfg.get("pooling_strategy", "mean")
    print(f"Using pooling strategy: {pooling_strategy}") # Thêm log để xác nhận

    model = HalluModel(model_cfg["pretrained"], num_labels=len(cfg.label_names),
                       dropout=cfg.dropout, proj_dim=int(getattr(cfg, "proj_dim", 128)),
                       pooling_strategy=pooling_strategy).to(device)

    optimizer = AdamW([{"params": [p for p in model.parameters() if p.requires_grad],
                        "weight_decay": cfg.weight_decay}], lr=cfg.lr)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(cfg.warmup_ratio * total_steps),
                                                num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing,
                                    weight=torch.ones(len(cfg.label_names), device=device))
    scaler = GradScaler(enabled=(device.type == "cuda"))

    os.makedirs(out_dir, exist_ok=True)
    best_f1, patience_counter, history, min_delta = 0.0, 0, [], 0.005

    for epoch in range(cfg.epochs):
        tr_loss, tr_f1 = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, device, cfg.grad_clip)
        val_loss, val_f1 = eval_one_epoch(model, val_loader, criterion, device)
        history.append({"train_loss": tr_loss, "train_f1": tr_f1, "val_loss": val_loss, "val_f1": val_f1})
        
        print(f"{model_cfg['name']} | Fold {fold_idx} | Epoch {epoch+1} | train_f1={tr_f1:.4f} | val_f1={val_f1:.4f}")

        if val_f1 > best_f1 + min_delta:
            best_f1 = val_f1; patience_counter = 0
            _save_ckpt(model, model_cfg, os.path.join(out_dir, f"{model_cfg['name']}_fold{fold_idx}.pth"))
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                break
    
    with open(os.path.join(out_dir, f"{model_cfg['name']}_fold{fold_idx}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    best_ckpt_path = os.path.join(out_dir, f"{model_cfg['name']}_fold{fold_idx}.pth")
    if not os.path.exists(best_ckpt_path):
        _save_ckpt(model, model_cfg, best_ckpt_path)

    # Temperature fit on true val set
    blob = torch.load(best_ckpt_path, map_location="cpu")
    model.load_state_dict(blob["state_dict"], strict=False)
    model.to(device)
    val_logits, val_labels = collect_logits_and_labels(model, val_loader, device)
    T = fit_temperature_from_logits(val_logits, val_labels, device=device.type)

    tw = {"temps": {}, "weights": {}}
    if os.path.exists(temps_weights_path):
        try:
            with open(temps_weights_path, "r") as f:
                tw = json.load(f)
        except Exception:
            tw = {"temps": {}, "weights": {}}
    tw["temps"][best_ckpt_path] = float(T)
    tw["weights"][best_ckpt_path] = float(best_f1)
    with open(temps_weights_path, "w") as f:
        json.dump(tw, f, indent=2)

    # OOF
    probs_ex, _ = _collect_val_probs(model, tokenizer, val_df, device, T, model_cfg)
    oof = pd.DataFrame({
        "id": val_df["id"].values if "id" in val_df.columns else np.arange(len(val_df)),
        "label": val_df["label"].values,
    })
    assert probs_ex.shape[0] == len(oof), f"OOF length mismatch: probs={probs_ex.shape[0]} vs val={len(oof)}"
    for ci, cname in enumerate(cfg.label_names):
        oof[f"p_{cname}"] = probs_ex[:, ci]
    oof_path = os.path.join(out_dir, f"{model_cfg['name']}_fold{fold_idx}_oof.parquet")
    oof.to_parquet(oof_path, index=False)
    return best_f1

# Thêm import ở đầu file
from sklearn.model_selection import train_test_split

# Thay thế toàn bộ hàm run_kfold_training bằng đoạn sau:
def run_kfold_training():
    ok, invalid = validate_models(cfg.models)
    if not ok:
        msg = "\n".join([f"- name={n}, repo={r}, error={e}" for n, r, e in invalid])
        raise ValueError(f"Model validation failed:\n{msg}")

    # Seed once per run to stabilize splits and any global RNG use
    set_repro(cfg.seed)

    df = pd.read_csv(cfg.train_csv)
    for col in ["context", "response", "label"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column in train CSV: {col}")

    # Nếu config có val_csv, dùng file đó làm validation; nếu không, chia ngẫu nhiên theo val_frac
    val_frac = getattr(cfg, "val_frac", 0.2)
    if hasattr(cfg, "val_csv") and cfg.val_csv:
        val_df = pd.read_csv(cfg.val_csv)
        train_df = df.copy()
    else:
        # stratify nếu có label
        if "label" in df.columns:
            train_df, val_df = train_test_split(df, test_size=val_frac, random_state=cfg.seed, stratify=df["label"])
        else:
            train_df, val_df = train_test_split(df, test_size=val_frac, random_state=cfg.seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    root_out = os.path.join(cfg.output_dir, timestamp)
    os.makedirs(root_out, exist_ok=True)

    fold_scores = {}
    model_oofs = {}

    # Duyệt từng model, train một lần trên split (fold_idx = 0)
    for model_cfg in cfg.models:
        model_out = os.path.join(root_out, model_cfg["name"])
        os.makedirs(model_out, exist_ok=True)
        temps_weights_path = os.path.join(model_out, "temps_weights.json")

        fold_scores[model_cfg["name"]] = []
        oofs = []

        # Chạy 1 lần (fold_idx = 0). Gọi lại hàm train_model_on_fold để tận dụng logic sẵn có.
        best_f1 = train_model_on_fold(model_cfg, 0, train_df.copy(), val_df.copy(), model_out, temps_weights_path)
        fold_scores[model_cfg["name"]].append(best_f1)

        # Load OOF tương ứng (tên file vẫn giữ định dạng _fold0_oof.parquet)
        oof_path = os.path.join(model_out, f"{model_cfg['name']}_fold0_oof.parquet")
        if not os.path.exists(oof_path):
            raise FileNotFoundError(f"Expected OOF file not found: {oof_path}")
        oofs.append(pd.read_parquet(oof_path))
        model_oofs[model_cfg["name"]] = pd.concat(oofs, axis=0).reset_index(drop=True)

    # Lưu summary
    with open(os.path.join(root_out, "fold_summary.json"), "w") as f:
        json.dump(fold_scores, f, indent=2)

    # Assemble stacker features (giữ nguyên logic)
    base = None
    for m in cfg.models:
        df_m = model_oofs[m["name"]].copy()
        rename_cols = {f"p_{c}": f"{m['name']}__p_{c}" for c in cfg.label_names}
        df_m = df_m.rename(columns=rename_cols)
        cols = ["id", "label"] + list(rename_cols.values())
        df_m = df_m[cols]
        base = df_m if base is None else base.merge(df_m, on=["id", "label"], how="inner")

    feature_cols = []
    for m in cfg.models:
        for c in cfg.label_names:
            feature_cols.append(f"{m['name']}__p_{c}")

    X = base[feature_cols].values.astype(np.float64)
    label_to_id = {n: i for i, n in enumerate(cfg.label_names)}
    y = np.array([label_to_id[str(n)] for n in base["label"].values], dtype=np.int64)

    stacker = LogisticRegression(multi_class="multinomial", solver="lbfgs",
                                 max_iter=1000, C=1.0, penalty="l2",
                                 class_weight=None, random_state=cfg.seed).fit(X, y)

    joblib.dump({
        "stacker": stacker,
        "feature_cols": feature_cols,
        "model_order": [m["name"] for m in cfg.models],
        "label_names": cfg.label_names,
    }, os.path.join(root_out, "stacker.joblib"))

    print("Saved models and stacker to:", root_out)
    return root_out

