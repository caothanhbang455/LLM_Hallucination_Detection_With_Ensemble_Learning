# src/ensemble.py
import os, glob, json, joblib, numpy as np, torch, torch.nn.functional as F, pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.config import cfg
from src.data import HalluDataset, LabelCodec, collate_fn_infer

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

def _load_ckpts_per_model(root_dir: str) -> dict[str, list[str]]:
    per_model = {}
    for mdir in sorted(os.listdir(root_dir)):
        full = os.path.join(root_dir, mdir)
        if os.path.isdir(full):
            ckpts = sorted(glob.glob(os.path.join(full, "*.pth")))
            if ckpts:
                per_model[mdir] = ckpts
    return per_model

def _lookup(mapping: dict, key: str):
    if key in mapping:
        return mapping[key]
    return mapping.get(os.path.basename(key), None)

def _cfg_by_name():
    return {m["name"]: m for m in cfg.models}

@torch.no_grad()
def stacked_predict(csv_path: str, root_dir: str):
    art_path = os.path.join(root_dir, "stacker.joblib")
    if not os.path.exists(art_path):
        raise ValueError(f"Missing stacker artifact at {art_path}")
    artifact = joblib.load(art_path)
    feature_cols = artifact["feature_cols"]
    model_order = artifact["model_order"]
    label_names = artifact["label_names"]

    per_model_ckpts = _load_ckpts_per_model(root_dir)
    tw_path = os.path.join(root_dir, "temps_weights.json")
    temps = {}
    if os.path.exists(tw_path):
        try:
            with open(tw_path, "r") as f:
                tw = json.load(f)
            temps = {k: float(v) for k, v in tw.get("temps", {}).items()}
        except Exception:
            temps = {}

    model_cfg_map = _cfg_by_name()

    df = pd.read_csv(csv_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = LabelCodec(label_names)

    per_model_avg_probs: dict[str, np.ndarray] = {}

    for mname in model_order:
        ckpts = per_model_ckpts.get(mname, [])
        if not ckpts:
            raise ValueError(f"No checkpoints found for base model '{mname}' under {root_dir}")

        blob_first = torch.load(ckpts[0], map_location="cpu")
        pretrained = blob_first.get("pretrained") or blob_first.get("backbone") or blob_first.get("model_name")
        tok = AutoTokenizer.from_pretrained(pretrained, use_fast=bool(model_cfg_map.get(mname, {}).get("use_fast", False)))

        max_len = int(model_cfg_map.get(mname, {}).get("text_max_len", cfg.text_max_len))
        stride   = int(model_cfg_map.get(mname, {}).get("doc_stride", cfg.doc_stride))
        seg_vi   = bool(model_cfg_map.get(mname, {}).get("segment_vi", False))

        ds = HalluDataset(df=df, tokenizer=tok, max_len=max_len, label_codec=codec, has_labels=False,
                          doc_stride=stride, sliding_windows=True, id_column="id", segment_vi=seg_vi)
        dl = DataLoader(ds, batch_size=getattr(cfg, "infer_batch_size", 64), shuffle=False,
                        collate_fn=collate_fn_infer, pin_memory=(device.type == "cuda"))

        from src.modeling import HalluModel
        probs_models = []
        for ck in ckpts:
            blob = torch.load(ck, map_location="cpu")
            state_dict = blob["state_dict"]
            pretrained_ck = blob.get("pretrained") or blob.get("backbone") or blob.get("model_name")
            dropout = blob.get("dropout", getattr(cfg, "dropout", 0.1))
            proj_dim = int(blob.get("proj_dim", getattr(cfg, "proj_dim", 128)))
            model = HalluModel(pretrained_ck, len(label_names), float(dropout), proj_dim).to(device)
            model.load_state_dict(state_dict, strict=False)
            model.eval()

            logits_all, orig_idx_all = [], []
            for b in dl:
                ids = b["input_ids"].to(device, non_blocking=True)
                mask = b["attention_mask"].to(device, non_blocking=True)
                logits = model(ids, mask)
                logits_all.append(logits.cpu())
                orig_idx_all.append(b["orig_index"])
            logits_all = torch.cat(logits_all, 0)
            orig_idx_all = torch.cat(orig_idx_all, 0)

            T = float(_lookup(temps, ck) or 1.0)
            probs_win = F.softmax(logits_all / max(T, 1e-6), dim=-1).cpu().numpy()
            probs_ex = _aggregate_windows_mean(orig_idx_all, probs_win)
            probs_models.append(probs_ex)

        per_model_avg_probs[mname] = np.mean(np.stack(probs_models, axis=0), axis=0)

    feats = [per_model_avg_probs[mname] for mname in model_order]
    X_meta = np.concatenate(feats, axis=1)
    if X_meta.shape[1] != len(feature_cols):
        raise ValueError(f"Feature width mismatch: got {X_meta.shape[1]}, expected {len(feature_cols)}")

    stacker = artifact["stacker"]
    preds = stacker.predict(X_meta)
    return preds, label_names
