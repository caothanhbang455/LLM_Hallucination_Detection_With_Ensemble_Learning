# src/calibration.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemperatureScaler(nn.Module):
    """Single-parameter temperature scaling as in Guo et al. (2017)."""
    def __init__(self, init_T: float = 1.0):
        super().__init__()
        self.log_T = nn.Parameter(torch.tensor(float(init_T)).log())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T = self.log_T.exp().clamp(min=1e-3, max=1e3)
        return logits / T

    @property
    def T(self) -> float:
        return float(self.log_T.exp().detach().cpu().item())

@torch.no_grad()
def nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels.long())

def fit_temperature_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    device: str = "cpu",
    max_iter: int = 500,
    lr: float = 0.01,
) -> float:
    logits = logits.to(device)
    labels = labels.to(device).long()
    scaler = TemperatureScaler().to(device)

    optim = torch.optim.LBFGS(
        scaler.parameters(),
        lr=lr,
        max_iter=max_iter,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optim.zero_grad(set_to_none=True)
        scaled = scaler(logits)
        loss = F.cross_entropy(scaled, labels)
        loss.backward()
        return loss

    optim.step(closure)
    return scaler.T

@torch.no_grad()
def apply_temperature(logits: torch.Tensor, T: float) -> torch.Tensor:
    return logits / max(T, 1e-6)
