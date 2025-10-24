# src/fold_train.py
import os, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
import joblib

from .config import cfg
from .utils import validate_models
from .data import HalluDataset, collate_fn_train, build_tokenizer, LabelCodec, make_sampler, kfold_splits
from .modeling import HalluModel
from .engine import train_one_epoch, eval_one_epoch, collect_logits_and_labels
from .calibration import fit_temperature_from_logits
from .data import collate_fn_infer

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

    # Use inference collator to get orig_index/window_index for aggregation
    ds = HalluDataset(val_df, tokenizer, max_len, codec, has_labels=True,
                      doc_stride=stride, sliding_windows=True, segment_vi=seg_vi)
    dl = DataLoader(ds, batch_size=cfg.val_batch_size, shuffle=False,
                    collate_fn=collate_fn_infer, pin_memory=(device.type == "cuda"))

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
    probs_ex = _aggregate_windows_mean(orig_idx_all, probs_win)  # shape [N_val, C]
    labels_ex = val_df["label"].values
    return probs_ex, labels_ex

def train_model_on_fold(model_cfg, fold_idx, train_df, val_df, out_dir, temps_weights_path):
    device = torch.device(cfg.device)
    codec = LabelCodec(cfg.label_names)

    max_len = int(model_cfg.get("text_max_len", cfg.text_max_len))
    stride = int(model_cfg.get("doc_stride", cfg.doc_stride))
    use_fast = bool(model_cfg.get("use_fast", False))
    seg_vi = bool(model_cfg.get("segment_vi", False))

    tokenizer = build_tokenizer(model_cfg["pretrained"], use_fast)

    train_ds = HalluDataset(train_df, tokenizer, max_len, codec, has_labels=True, doc_stride=stride, sliding_windows=True, segment_vi=seg_vi)
    val_ds   = HalluDataset(val_df,   tokenizer, max_len, codec, has_labels=True, doc_stride=stride, sliding_windows=True, segment_vi=seg_vi)

    sampler = make_sampler(train_df, codec)
    train_loader = DataLoader(train_ds, batch_size=int(cfg.batch_size), sampler=sampler, collate_fn=collate_fn_train, pin_memory=(device.type=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=int(cfg.val_batch_size), shuffle=False, collate_fn=collate_fn_train, pin_memory=(device.type=="cuda"))

    model = HalluModel(model_cfg["pretrained"], num_labels=len(cfg.label_names), dropout=cfg.dropout, proj_dim=int(getattr(cfg, "proj_dim", 128))).to(device)

    optimizer = AdamW([{"params": [p for p in model.parameters() if p.requires_grad], "weight_decay": cfg.weight_decay}], lr=cfg.lr)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(cfg.warmup_ratio * total_steps), num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing, weight=torch.ones(len(cfg.label_names), device=device))
    scaler = GradScaler(enabled=(device.type == "cuda"))

    os.makedirs(out_dir, exist_ok=True)
    best_f1, patience_counter, history, min_delta = 0.0, 0, [], 0.005
    for epoch in range(cfg.epochs):
        tr_loss, tr_f1 = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, device, cfg.grad_clip)
        val_loss, val_f1 = eval_one_epoch(model, val_loader, criterion, device)
        history.append({"epoch": epoch+1, "train_loss": tr_loss, "train_f1": tr_f1, "val_loss": val_loss, "val_f1": val_f1})
        print(f"{model_cfg['name']} | Fold {fold_idx} | Epoch {epoch+1} | train_f1={tr_f1:.4f} | val_f1={val_f1:.4f}")
        if val_f1 > best_f1 + min_delta:
            best_f1 = val_f1; patience_counter = 0
            _save_ckpt(model, model_cfg, os.path.join(out_dir, f"{model_cfg['name']}_fold{fold_idx}.pth"))
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"{model_cfg['name']} | Fold {fold_idx} | Early stopping at epoch {epoch+1}")
                break
    with open(os.path.join(out_dir, f"{model_cfg['name']}_fold{fold_idx}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    best_ckpt_path = os.path.join(out_dir, f"{model_cfg['name']}_fold{fold_idx}.pth")
    if not os.path.exists(best_ckpt_path):
        _save_ckpt(model, model_cfg, best_ckpt_path)

    # Load best and fit temperature
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

    # Calibrated OOF probabilities for this fold
    probs_ex, labels_ex = _collect_val_probs(model, tokenizer, val_df, device, T, model_cfg)
    oof = pd.DataFrame({
        "id": val_df["id"].values if "id" in val_df.columns else np.arange(len(val_df)),
        "label": val_df["label"].values,
    })
    for ci, cname in enumerate(cfg.label_names):
        oof[f"p_{cname}"] = probs_ex[:, ci]
    oof_path = os.path.join(out_dir, f"{model_cfg['name']}_fold{fold_idx}_oof.parquet")
    
    assert probs_ex.shape[0] == len(oof), f"OOF length mismatch: probs={probs_ex.shape[0]} vs val={len(oof)}"

    oof.to_parquet(oof_path, index=False)
    return best_f1

def run_kfold_training():
    ok, invalid = validate_models(cfg.models)
    if not ok:
        msg = "\n".join([f"- name={n}, repo={r}, error={e}" for n, r, e in invalid])
        raise ValueError(f"Model validation failed, fix these before training:\n{msg}")

    df = pd.read_csv(cfg.train_csv)
    for col in ["context", "response", "label"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column in train CSV: {col}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    root_out = os.path.join(cfg.output_dir, timestamp)
    os.makedirs(root_out, exist_ok=True)

    fold_scores = {}
    model_oofs = {}

    for model_cfg in cfg.models:
        model_out = os.path.join(root_out, model_cfg["name"])
        os.makedirs(model_out, exist_ok=True)
        temps_weights_path = os.path.join(model_out, "temps_weights.json")

        fold_scores[model_cfg["name"]] = []
        oofs = []
        for fold, tr_df, va_df in kfold_splits(df, cfg.n_splits, cfg.seed):
            best_f1 = train_model_on_fold(model_cfg, fold, tr_df, va_df, model_out, temps_weights_path)
            fold_scores[model_cfg["name"]].append(best_f1)
            oofs.append(pd.read_parquet(os.path.join(model_out, f"{model_cfg['name']}_fold{fold}_oof.parquet")))
        model_oofs[model_cfg["name"]] = pd.concat(oofs, axis=0).reset_index(drop=True)

    with open(os.path.join(root_out, "fold_summary.json"), "w") as f:
        json.dump(fold_scores, f, indent=2)

    # Assemble stacker training data
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

    stacker = LogisticRegression(
        multi_class="multinomial",
        max_iter=int(cfg.stacker_max_iter),
        C=float(cfg.stacker_C),
        penalty=str(cfg.stacker_penalty),
        class_weight=(None if cfg.stacker_class_weight is None else cfg.stacker_class_weight),
        solver="lbfgs",
        n_jobs=None,
        random_state=cfg.seed,
    ).fit(X, y)

    artifact = {
        "stacker": stacker,
        "feature_cols": feature_cols,
        "model_order": [m["name"] for m in cfg.models],
        "label_names": cfg.label_names,
    }
    joblib.dump(artifact, os.path.join(root_out, "stacker.joblib"))
    print("Saved k-fold models and stacker to:", root_out)
    return root_out
