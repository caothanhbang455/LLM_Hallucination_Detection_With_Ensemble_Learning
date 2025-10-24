# src/config.py
from dataclasses import dataclass, field
from typing import List, Dict
import torch

@dataclass
class Config:
    # data
    train_csv: str = "vihallu-train - vihallu-train.csv"
    test_csv: str = "vihallu-private-test.csv"
    text_max_len: int = 512
    doc_stride: int = 96
    label_names: List[str] = field(default_factory=lambda: ["no", "intrinsic", "extrinsic"])

    # training
    seed: int = 1234
    epochs: int = 30
    batch_size: int = 32
    val_batch_size: int = 16
    infer_batch_size: int = 64
    lr: float = 5e-5
    weight_decay: float = 0.1
    warmup_ratio: float = 0.05
    dropout: float = 0.38
    label_smoothing: float = 0.08
    grad_clip: float = 1.0
    patience: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "outputs"

    # contrastive (kept for checkpoint compatibility)
    proj_dim: int = 128
    queue_capacity: int = 4096

    # freezing
    train_embeddings: bool = True
    last_n_layers_trainable: int = 2

    # k-fold / base models
    n_splits: int = 5
    # models: List[Dict] = field(default_factory=lambda: [
    #     # PhoBERT + segmentation + shorter max length
    #     # {"name": "phobert_base_v2",
    #     #  "pretrained": "vinai/phobert-base-v2",
    #     #  "use_fast": False,
    #     #  "segment_vi": True,
    #     #  "text_max_len": 256,
    #     #  "doc_stride": 64},
    #     {"name": "minilmv2", "pretrained": "MoritzLaurer/multilingual-MiniLMv2-L12-mnli-xnli", "use_fast": False},
    #     {"name": "mdeberta_v3_base", "pretrained": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "use_fast": False},
    #     # {"name": "xlmr_large_xnli", "pretrained": "joeddav/xlm-roberta-large-xnli", "use_fast": False},
    # ])

    models: List[Dict] = field(default_factory=lambda: [
        {"name": "minilmv2", 
         "pretrained": "MoritzLaurer/multilingual-MiniLMv2-L12-mnli-xnli", 
         "use_fast": False,
         "pooling_strategy": "mean"}, # <--- THÊM DÒNG NÀY

        {"name": "mdeberta_v3_base", 
         "pretrained": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", 
         "use_fast": False,
         "pooling_strategy": "mean"}, # <--- VÀ DÒNG NÀY

         {"name": "multilingual-e5", 
         "pretrained": "intfloat/multilingual-e5-base", 
         "use_fast": False,
         "pooling_strategy": "mean"}, # <--- VÀ DÒNG NÀY
    ])


    # stacker
    stacker_max_iter: int = 2000
    stacker_C: float = 1.0
    stacker_penalty: str = "l2"
    stacker_class_weight: str | None = None

cfg = Config()
