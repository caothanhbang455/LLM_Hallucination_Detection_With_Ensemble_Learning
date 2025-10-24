# src/data.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------
# Vietnamese segmentation support (for PhoBERT only)
# Priority: project-local jar -> VNCORENLP_JAR -> py_vncorenlp (VNCORENLP_DIR)
# ---------------------------------------------------------------------
_VI_SEG = {"ready": False, "seg": None}

def _candidate_local_jar_paths():
    here = os.path.dirname(__file__)
    root = os.path.abspath(os.path.join(here, os.pardir))
    return [
        os.path.join(root, "vncorenlp", "VnCoreNLP-1.2.jar"),
        os.path.join(here, "..", "vncorenlp", "VnCoreNLP-1.2.jar"),
        os.path.join(here, "vncorenlp", "VnCoreNLP-1.2.jar"),
    ]

def _resolve_jar_path():
    for p in _candidate_local_jar_paths():
        p_abs = os.path.abspath(p)
        if os.path.exists(p_abs):
            return p_abs
    jar = os.environ.get("VNCORENLP_JAR", None)
    if jar and os.path.exists(jar):
        return os.path.abspath(jar)
    return None

def load_vi_segmenter(save_dir: str | None = None):
    if _VI_SEG["ready"]:
        return _VI_SEG["seg"]

    jar = _resolve_jar_path()
    if jar is not None:
        from vncorenlp import VnCoreNLP
        seg = VnCoreNLP(jar, annotators="wseg", max_heap_size="-Xmx1g")
        _VI_SEG["seg"] = seg
        _VI_SEG["ready"] = True
        return seg

    try:
        import py_vncorenlp
        if save_dir is None:
            save_dir = os.environ.get("VNCORENLP_DIR", "./vncorenlp")
        os.makedirs(save_dir, exist_ok=True)
        py_vncorenlp.download_model(save_dir=save_dir)
        seg = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=save_dir)
        _VI_SEG["seg"] = seg
        _VI_SEG["ready"] = True
        return seg
    except Exception as e:
        raise RuntimeError(
            "VnCoreNLP not available. Place VnCoreNLP-1.2.jar under ./vncorenlp or ../vncorenlp, "
            "or set VNCORENLP_JAR=/abs/path/VnCoreNLP-1.2.jar; alternatively install Java+pyjnius "
            f"so py_vncorenlp can load the jar (original error: {e})."
        )

def segment_vi_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    seg = load_vi_segmenter()
    if hasattr(seg, "word_segment"):
        sents = seg.word_segment(text)
        return " ".join(sents)
    if hasattr(seg, "tokenize"):
        toks_by_sent = seg.tokenize(text)
        sents = [" ".join(toks) for toks in toks_by_sent]
        return " ".join(sents)
    return text

class LabelCodec:
    def __init__(self, names):
        self.name2id = {n: i for i, n in enumerate(names)}
        self.id2name = {i: n for n, i in self.name2id.items()}

class HalluDataset(Dataset):
    """
    Sliding-window dataset for NLI-style classification with optional Vietnamese
    word segmentation for PhoBERT-only usage.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_len: int,
        label_codec: LabelCodec | None,
        has_labels: bool = True,
        doc_stride: int = 128,
        sliding_windows: bool = True,
        id_column: str | None = "id",
        segment_vi: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.codec = label_codec
        self.has_labels = has_labels
        self.doc_stride = int(doc_stride)
        self.sliding_windows = bool(sliding_windows)
        self.id_column = id_column
        self.segment_vi = bool(segment_vi)

        if "context" not in self.df.columns:
            raise ValueError("Missing required column: 'context'")
        if "response" not in self.df.columns:
            raise ValueError("Missing required column: 'response'")
        if self.has_labels and self.codec is None:
            raise ValueError("label_codec is required when has_labels=True")

        self.features = []
        for row_idx, row in self.df.iterrows():
            base_ctx = str(row.get("retrieved_context", "") or "").strip()
            context = base_ctx if base_ctx else str(row.get("context", "") or "").strip()
            response = str(row.get("response", "") or "").strip()

            if self.segment_vi:
                context = segment_vi_text(context)
                response = segment_vi_text(response)

            enc = self.tokenizer(
                text=context,
                text_pair=response,
                max_length=self.max_len,
                padding="max_length",
                truncation="only_first",
                return_overflowing_tokens=self.sliding_windows,
                stride=self.doc_stride if self.sliding_windows else 0,
                return_tensors="pt",
                return_token_type_ids=False,
                return_attention_mask=True,
            )

            if "attention_mask" not in enc:
                pad_id = self.tokenizer.pad_token_id
                if pad_id is None:
                    pad_id = 0
                att = (enc["input_ids"] != pad_id).long()
                enc["attention_mask"] = att

            if enc.input_ids.dim() == 1:
                for k in enc:
                    enc[k] = enc[k].unsqueeze(0)

            n_windows = enc.input_ids.size(0)
            for win_idx in range(n_windows):
                item = {
                    "input_ids": enc.input_ids[win_idx],
                    "attention_mask": enc.attention_mask[win_idx],
                    "orig_index": torch.tensor(row_idx, dtype=torch.long),
                    "window_index": torch.tensor(win_idx, dtype=torch.long),
                }
                if self.id_column and (self.id_column in self.df.columns):
                    item["orig_id"] = row.get(self.id_column)
                if self.has_labels:
                    raw = row.get("label", None)
                    if pd.isna(raw) or raw is None or str(raw) == "":
                        y = 0
                    else:
                        y = self.codec.name2id.get(str(raw), 0)
                    item["labels"] = torch.tensor(y, dtype=torch.long)
                self.features.append(item)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

def collate_fn_train(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def collate_fn_infer(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    orig_index = torch.stack([b["orig_index"] for b in batch])
    window_index = torch.stack([b["window_index"] for b in batch])
    out = {"input_ids": input_ids, "attention_mask": attention_mask, "orig_index": orig_index, "window_index": window_index}
    if "orig_id" in batch[0]:
        out["orig_id"] = [b["orig_id"] for b in batch]
    return out

def build_tokenizer(model_name: str, use_fast: bool):
    return AutoTokenizer.from_pretrained(model_name, use_fast=bool(use_fast))

def make_sampler(df: pd.DataFrame, label_codec: LabelCodec, gen: torch.Generator | None = None):
    if "label" not in df.columns:
        raise ValueError("make_sampler requires a 'label' column")
    y = df["label"].map(label_codec.name2id).values
    counts = np.bincount(y, minlength=len(label_codec.name2id))
    weights = 1.0 / np.maximum(counts, 1)[y]
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(y),
        replacement=True,
        generator=gen,  # seeded outside for stable sampling
    )

def kfold_splits(df: pd.DataFrame, n_splits: int, seed: int):
    if "label" not in df.columns:
        raise ValueError("kfold_splits requires a 'label' column")
    y = df["label"].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(df, y)):
        yield fold, df.iloc[tr_idx].copy(), df.iloc[val_idx].copy()
