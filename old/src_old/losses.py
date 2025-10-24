import torch
import torch.nn as nn

def info_nce_supervised(feats, labels, temperature, memory=None, eps=1e-8):
    feats = nn.functional.normalize(feats, p=2, dim=-1)
    device = feats.device
    B = feats.size(0)

    mem_feats, mem_labels = (None, None)
    if memory is not None:
        mem_feats, mem_labels = memory.get()

    if mem_feats is not None and mem_labels is not None and mem_feats.numel() > 0:
        all_feats = torch.cat([feats, nn.functional.normalize(mem_feats, p=2, dim=-1)], dim=0)
        all_labels = torch.cat([labels, mem_labels], dim=0)
    else:
        all_feats = feats
        all_labels = labels

    sim = torch.matmul(feats, all_feats.T) / torch.clamp(temperature, min=1e-4, max=1e2)
    targets = labels.unsqueeze(1) == all_labels.unsqueeze(0)

    self_mask = torch.zeros_like(sim, dtype=torch.bool, device=device)
    self_mask[:, :B] = torch.eye(B, dtype=torch.bool, device=device)
    pos_mask = targets & (~self_mask)

    row_max, _ = torch.max(sim, dim=1, keepdim=True)
    sim_stable = sim - row_max
    exp_sim = torch.exp(sim_stable)

    denom = exp_sim.sum(dim=1)
    numerator = (exp_sim * pos_mask.float()).sum(dim=1)

    valid = pos_mask.float().sum(dim=1) > 0
    if not valid.any():
        return torch.tensor(0.0, device=device)

    log_prob = torch.log(numerator + eps) - torch.log(denom + eps)
    return -log_prob[valid].mean()
