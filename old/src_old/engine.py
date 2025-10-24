# src/engine.py
from typing import Tuple
import contextlib
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score
from tqdm import tqdm

def _maybe_clip(model: nn.Module, max_norm):
    if max_norm is None or max_norm <= 0:
        return
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    grad_clip: float | None = None,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    preds_all, labels_all = [], []
    use_amp = (device.type == "cuda")
    amp_ctx = autocast(device_type="cuda", dtype=torch.float16) if use_amp else contextlib.nullcontext()

    for batch in tqdm(dataloader, desc="Train"):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        _maybe_clip(model, grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.item())
        preds_all.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
        labels_all.extend(labels.detach().cpu().numpy())

    macro_f1 = f1_score(labels_all, preds_all, average="macro")
    n = max(len(dataloader), 1)
    return total_loss / n, macro_f1

@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    preds_all, labels_all = [], []
    for batch in tqdm(dataloader, desc="Eval"):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        total_loss += float(loss.item())
        preds_all.extend(torch.argmax(logits, dim=1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    n = max(len(dataloader), 1)
    macro_f1 = f1_score(labels_all, preds_all, average="macro")
    return total_loss / n, macro_f1

@torch.no_grad()
def collect_logits_and_labels(model: nn.Module, dataloader, device: torch.device):
    model.eval()
    logits_all, labels_all = [], []
    for batch in tqdm(dataloader, desc="Collect"):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        logits = model(input_ids, attention_mask)
        logits_all.append(logits.detach().cpu())
        labels_all.append(labels.detach().cpu())
    return torch.cat(logits_all, 0), torch.cat(labels_all, 0)
