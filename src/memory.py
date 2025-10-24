import torch

class CrossBatchMemory:
    def __init__(self, feat_dim: int, capacity: int, device):
        self.capacity = int(capacity)
        self.device = device
        self.ptr = 0
        self.is_full = False
        self.feats = torch.zeros(self.capacity, feat_dim, dtype=torch.float32, device=self.device)
        self.labels = torch.full((self.capacity,), -1, dtype=torch.long, device=self.device)

    def add(self, feats, labels):
        bsz = feats.size(0)
        if bsz >= self.capacity:
            self.feats = feats[-self.capacity:].detach()
            self.labels = labels[-self.capacity:].detach()
            self.ptr = 0
            self.is_full = True
            return

        end = self.ptr + bsz
        if end <= self.capacity:
            self.feats[self.ptr:end] = feats.detach()
            self.labels[self.ptr:end] = labels.detach()
        else:
            first = self.capacity - self.ptr
            self.feats[self.ptr:] = feats[:first].detach()
            self.labels[self.ptr:] = labels[:first].detach()
            rem = bsz - first
            self.feats[:rem] = feats[first:].detach()
            self.labels[:rem] = labels[first:].detach()
        self.ptr = (self.ptr + bsz) % self.capacity
        if self.ptr == 0:
            self.is_full = True

    def get(self):
        if self.is_full:
            return self.feats, self.labels
        if self.ptr == 0:
            return None, None
        return self.feats[: self.ptr], self.labels[: self.ptr]
