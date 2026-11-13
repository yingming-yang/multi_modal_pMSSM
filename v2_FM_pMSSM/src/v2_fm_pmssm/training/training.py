import torch
import math

class EpochSampler:
    """Shuffled minibatches over the full split; reshuffles after each epoch."""
    def __init__(self, data, batch_size, generator):
        self.data, self.bs, self.g = data, batch_size, generator
        self.perm, self.pos, self.epoch = None, len(data), -1

    def next(self):
        if self.pos + self.bs > len(self.data):  # drop the last incomplete batch of each epoch
            self.perm = torch.randperm(len(self.data), device=self.data.device, generator=self.g)
            self.pos, self.epoch = 0, self.epoch + 1
        idx = self.perm[self.pos:self.pos + self.bs]
        self.pos += self.bs
        return self.data[idx]


def fm_loss(model, path, masks, target, mask_idx, times, noise):
    known = masks[mask_idx]
    path_sample = path.sample(x_0=noise, x_1=target, t=times)
    positions = torch.where(known, target, path_sample.x_t)
    xt = torch.column_stack([positions, times])
    model_field = model(xt, known=known)
    unknown = ~known
    return ((model_field - path_sample.dx_t) ** 2 * unknown).sum() / unknown.sum()


@torch.no_grad()
def evaluate(model, path, masks, val_set, eval_bs):
    """Weighted mean loss over the whole validation split (fixed noise/times/masks)."""
    model.eval()
    data, mask_idx, times, noise = val_set
    total, n_unknown = 0.0, 0
    for s in range(0, len(data), eval_bs):
        sl = slice(s, s + eval_bs)
        unknown = (~masks[mask_idx[sl]]).sum().item()
        loss = fm_loss(model, path, masks, data[sl], mask_idx[sl], times[sl], noise[sl])
        total += loss.item() * unknown
        n_unknown += unknown
    model.train()
    return total / n_unknown


def lr_at(step, max_iters, warmup, lr, min_lr):
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / (max_iters - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))