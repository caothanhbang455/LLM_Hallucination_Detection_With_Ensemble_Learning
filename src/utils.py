# src/utils.py
from typing import List, Dict, Tuple
import os, random
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, accuracy_score, log_loss

def validate_models(models: List[Dict]) -> Tuple[bool, list]:
    invalid = []
    for m in models:
        name = m.get("name", "")
        repo = m.get("pretrained", None)
        use_fast = m.get("use_fast", True)
        try:
            if repo is None:
                raise ValueError("missing 'pretrained' key")
            AutoConfig.from_pretrained(repo)
            AutoTokenizer.from_pretrained(repo, use_fast=use_fast)
            _ = AutoModel.from_pretrained(repo, dtype=torch.float32)
        except Exception as e:
            invalid.append((name, repo, str(e)))
    return (len(invalid) == 0), invalid

def macro_f1_from_probs(probs: np.ndarray, labels: np.ndarray) -> float:
    preds = probs.argmax(axis=1)
    return float(f1_score(labels, preds, average="macro"))

def basic_metrics_from_probs(probs: np.ndarray, labels: np.ndarray) -> dict:
    preds = probs.argmax(axis=1)
    K = probs.shape[1]
    one_hot = np.eye(K)[labels]
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    return {
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "accuracy": float(accuracy_score(labels, preds)),
        "nll": float(log_loss(labels, probs, labels=list(range(K)))),
        "brier": brier,
    }

def seed_worker(worker_id: int):
    # Ensure each worker has a stable derivative seed
    seed = torch.initial_seed() % 2**31
    random.seed(seed)
    np.random.seed(seed)

def set_repro(seed: int):
    """
    Reproducibility without strict determinism:
    - seeds Python, NumPy, PyTorch (CPU/CUDA)
    - prevents cuDNN kernel auto-tuning variability
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Keep cuDNN fast but avoid kernel roulette
    torch.backends.cudnn.benchmark = False
