from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset


METHODS = ("sgd", "qat", "binary", "dbs")


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loaders(args):
    if args.synthetic:
        # Offline plumbing test only; never presented as an MNIST result.
        g = torch.Generator().manual_seed(args.seed)
        ntr, nte = args.train_limit or 512, args.test_limit or 128
        train = TensorDataset(torch.randn(ntr, 1, 28, 28, generator=g), torch.randint(0, 10, (ntr,), generator=g))
        test = TensorDataset(torch.randn(nte, 1, 28, 28, generator=g), torch.randint(0, 10, (nte,), generator=g))
    else:
        try:
            from torchvision import datasets, transforms
        except ImportError as e:
            raise RuntimeError("torchvision is required for MNIST; install requirements.txt") from e
        tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train = datasets.MNIST(args.data_dir, train=True, download=True, transform=tfm)
        test = datasets.MNIST(args.data_dir, train=False, download=True, transform=tfm)
    if args.train_limit and not args.synthetic:
        train = Subset(train, range(min(args.train_limit, len(train))))
    if args.test_limit and not args.synthetic:
        test = Subset(test, range(min(args.test_limit, len(test))))
    g = torch.Generator().manual_seed(args.seed)
    common = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=args.device.startswith("cuda"))
    return (DataLoader(train, shuffle=True, generator=g, **common),
            DataLoader(test, shuffle=False, **common))


class MLP(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = x.flatten(1)
        return self.fc2(F.relu(self.fc1(x)))


def fake_quant_ste(w, bins=256):
    scale = w.detach().abs().max().clamp_min(1e-8) / ((bins - 1) / 2)
    q = (w / scale).round().clamp(-(bins // 2), bins // 2 - 1) * scale
    return w + (q - w).detach()


class QATMLP(MLP):
    def __init__(self, hidden=256, bins=256):
        super().__init__(hidden)
        self.bins = bins

    def forward(self, x):
        x = x.flatten(1)
        x = F.linear(x, fake_quant_ste(self.fc1.weight, self.bins), fake_quant_ste(self.fc1.bias, self.bins))
        x = F.relu(x)
        return F.linear(x, fake_quant_ste(self.fc2.weight, self.bins), fake_quant_ste(self.fc2.bias, self.bins))


class BinaryMLP:
    """Persistent signs plus Bop momentum; gradients live only for the current step."""
    def __init__(self, hidden, device, tau=1e-4, gamma=0.99):
        self.shapes = [(hidden, 784), (hidden,), (10, hidden), (10,)]
        self.signs = [torch.where(torch.randn(s, device=device) >= 0, 1, -1).to(torch.int8) for s in self.shapes]
        self.momentum = [torch.zeros(s, device=device) for s in self.shapes]
        self.tau, self.gamma = tau, gamma
        self.flips = 0

    def tensors(self, grad=False):
        return [s.float().requires_grad_(grad) for s in self.signs]

    @staticmethod
    def forward_with(x, p):
        w1, b1, w2, b2 = p
        return F.linear(F.relu(F.linear(x.flatten(1), w1 / math.sqrt(784), b1)), w2 / math.sqrt(w1.shape[0]), b2)

    def train_step(self, x, y):
        p = self.tensors(True)
        loss = F.cross_entropy(self.forward_with(x, p), y)
        grads = torch.autograd.grad(loss, p)
        for i, g in enumerate(grads):
            self.momentum[i].mul_(self.gamma).add_(g, alpha=1 - self.gamma)
            flip = (self.momentum[i].abs() >= self.tau) & (self.momentum[i].sign() == self.signs[i].float())
            self.flips += int(flip.sum())
            self.signs[i][flip] *= -1
            self.momentum[i][flip] = 0
        return float(loss.detach())

    def predict(self, x):
        return self.forward_with(x, self.tensors(False))


class DBSMLP:
    """Only uint8 bin indices and integer evidence persist between steps."""
    def __init__(self, hidden, device, bins=256, block=1024, fraction=.25,
                 threshold=3, seed=1, variant="layer_balanced", init_levels=8,
                 evidence_rule="reset"):
        if not 2 <= bins <= 256:
            raise ValueError("DBS bins must be in [2, 256] for uint8 storage")
        if variant not in ("legacy", "layer_scaled", "layer_scheduled", "layer_balanced"):
            raise ValueError("Unknown DBS variant")
        if variant in ("layer_scaled", "layer_balanced") and bins < 3:
            raise ValueError("Layer-scaled DBS needs at least 3 bins to represent zero and both signs")
        if block < 1 or threshold < 1 or init_levels <= 0:
            raise ValueError("DBS block, threshold, and init levels must be positive")
        if evidence_rule not in ("reset", "residual"):
            raise ValueError("DBS evidence rule must be reset or residual")
        self.shapes = [(hidden, 784), (hidden,), (10, hidden), (10,)]
        self.offsets, n = [], 0
        for s in self.shapes:
            self.offsets.append((n, n + math.prod(s))); n += math.prod(s)
        gen = torch.Generator(device=device).manual_seed(seed)
        self.variant = variant
        self.zero_point = bins // 2
        self.q = torch.full((n,), self.zero_point, dtype=torch.uint8, device=device)
        if variant in ("legacy", "layer_scheduled"):
            # Preserve the original initialization and decoding for comparisons.
            self.q.fill_((bins - 1) // 2)
            jitter = torch.randint(-2, 3, (n,), generator=gen, device=device)
            self.q = (self.q.to(torch.int16) + jitter).clamp(0, bins - 1).to(torch.uint8)
            self.scales = None
        else:
            # Fixed scalar scales are metadata, not trainable or shadow weights.
            levels = min(init_levels, max(1.0, (bins - 1) / 8))
            first_std = math.sqrt(2.0 / 784)
            last_std = math.sqrt(1.0 / hidden)
            self.scales = (first_std / levels, first_std / levels,
                           last_std / levels, last_std / levels)
            for i, std in ((0, first_std), (2, last_std)):
                a, b = self.offsets[i]
                codes = (torch.randn(self.shapes[i], generator=gen, device=device)
                         * (std / self.scales[i])).round().to(torch.int16)
                self.q[a:b] = (codes.flatten() + self.zero_point).clamp(0, bins - 1).to(torch.uint8)
        self.evidence = torch.zeros(n, dtype=torch.int16, device=device)
        self.bins, self.block, self.fraction, self.threshold = bins, block, fraction, threshold
        self.evidence_rule = evidence_rule
        self.cursor, self.moves, self.gen = 0, 0, gen
        self.layer_cursors = [0] * len(self.shapes)

    def values(self, q=None):
        q = self.q if q is None else q
        if self.variant in ("legacy", "layer_scheduled"):
            flat = q.float().mul(2.0 / (self.bins - 1)).sub(1.0)
            return [flat[a:b].reshape(s) for (a, b), s in zip(self.offsets, self.shapes)]
        return [(q[a:b].float() - self.zero_point).mul(scale).reshape(shape)
                for (a, b), shape, scale in zip(self.offsets, self.shapes, self.scales)]

    def block_indices(self, device):
        n = self.q.numel()
        size = min(self.block, n)
        if self.variant in ("legacy", "layer_scaled"):
            idx = (torch.arange(size, device=device) + self.cursor) % n
            self.cursor = (self.cursor + size) % n
            return idx
        # Square-root allocation visits small but important layers more often,
        # while cycling through every parameter within each layer.
        lengths = [b - a for a, b in self.offsets]
        weights = [math.sqrt(length) for length in lengths]
        quotas = [size * weight / sum(weights) for weight in weights]
        counts = [min(length, int(quota)) for length, quota in zip(lengths, quotas)]
        for _ in range(size - sum(counts)):
            eligible = [i for i, length in enumerate(lengths) if counts[i] < length]
            best = max(eligible, key=lambda i: quotas[i] - counts[i])
            counts[best] += 1
        parts = []
        for i, ((a, b), count) in enumerate(zip(self.offsets, counts)):
            if count:
                length = b - a
                parts.append((torch.arange(count, device=device) + self.layer_cursors[i]) % length + a)
                self.layer_cursors[i] = (self.layer_cursors[i] + count) % length
        return torch.cat(parts)

    @staticmethod
    def forward_with(x, p):
        w1, b1, w2, b2 = p
        return F.linear(F.relu(F.linear(x.flatten(1), w1, b1)), w2, b2)

    def predict(self, x):
        return self.forward_with(x, self.values())

    @torch.no_grad()
    def apply_votes(self, idx, votes):
        self.evidence[idx] = (self.evidence[idx].to(torch.int32) + votes.to(torch.int32)).clamp(-32767, 32767).to(torch.int16)
        ready = self.evidence[idx].abs() >= self.threshold
        old = self.q[idx].to(torch.int16)
        new = (old + self.evidence[idx].sign() * ready).clamp(0, self.bins - 1)
        moved = ready & (new != old)
        self.moves += int(moved.sum())
        self.q[idx] = new.to(torch.uint8)
        if self.evidence_rule == "residual":
            # Preserve votes beyond the threshold, like error-feedback quantization.
            # A clipped boundary move cannot use its votes, so discard those.
            self.evidence[idx[moved]] -= self.threshold * self.evidence[idx[moved]].sign()
            self.evidence[idx[ready & ~moved]] = 0
        else:
            self.evidence[idx[moved]] = 0

    @torch.no_grad()
    def train_step(self, x, y, probes):
        idx = self.block_indices(x.device)
        size = idx.numel()
        active = max(1, round(size * self.fraction))
        ev = torch.zeros(size, dtype=torch.int16, device=x.device)
        total_loss = 0.0
        for _ in range(probes):
            chosen = torch.randperm(size, generator=self.gen, device=x.device)[:active]
            r = torch.zeros(size, dtype=torch.int16, device=x.device)
            r[chosen] = torch.where(torch.rand(active, generator=self.gen, device=x.device) < .5, -1, 1).to(torch.int16)
            base = self.q[idx].to(torch.int16)
            qp, qm = self.q.clone(), self.q.clone()
            qp[idx] = (base + r).clamp(0, self.bins - 1).to(torch.uint8)
            qm[idx] = (base - r).clamp(0, self.bins - 1).to(torch.uint8)
            lp = F.cross_entropy(self.forward_with(x, self.values(qp)), y)
            lm = F.cross_entropy(self.forward_with(x, self.values(qm)), y)
            direction = 1 if lp < lm else (-1 if lm < lp else 0)
            ev.add_(r, alpha=direction)
            total_loss += float((lp + lm) / 2)
        self.apply_votes(idx, ev)
        return total_loss / probes


class EnergyMeter:
    def __init__(self, device):
        self.available, self.joules, self.last_t, self.last_w = False, 0.0, None, None
        if not device.startswith("cuda"): return
        try:
            import pynvml
            pynvml.nvmlInit(); self.nvml = pynvml
            index = int(device.split(":")[1]) if ":" in device else 0
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            self.available = True; self.sample()
        except Exception: self.available = False

    def sample(self):
        if not self.available: return
        try:
            now, watts = time.perf_counter(), self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000
            if self.last_t is not None: self.joules += (now - self.last_t) * (watts + self.last_w) / 2
            self.last_t, self.last_w = now, watts
        except Exception: self.available = False


@dataclass
class Row:
    method: str; epoch: int; test_accuracy: float; train_loss: float
    forward_evaluations: int; bin_moves_or_bit_flips: int; wall_time_s: float
    peak_cuda_allocated_mb: float | None; peak_cuda_reserved_mb: float | None; energy_joules: float | None


@torch.no_grad()
def accuracy(model, loader, device, method):
    correct = total = 0
    if isinstance(model, nn.Module): model.eval()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x) if isinstance(model, nn.Module) else model.predict(x)
        correct += int((out.argmax(1) == y).sum()); total += y.numel()
    return correct / total


def run_method(method, args, train_loader, test_loader):
    seed_all(args.seed); device = torch.device(args.device)
    # Rewind shuffling so every method sees the same minibatch order.
    if train_loader.generator is not None:
        train_loader.generator.manual_seed(args.seed)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    if method == "sgd": model = MLP(args.hidden).to(device)
    elif method == "qat": model = QATMLP(args.hidden, args.bins).to(device)
    elif method == "binary": model = BinaryMLP(args.hidden, device, args.bop_tau, args.bop_gamma)
    else: model = DBSMLP(args.hidden, device, args.bins, args.block_size,
                         args.perturb_fraction, args.threshold, args.seed,
                         args.dbs_variant, args.dbs_init_levels, args.dbs_evidence_rule)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.9) if isinstance(model, nn.Module) else None
    meter, rows, forwards, started = EnergyMeter(args.device), [], 0, time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        losses = []
        if isinstance(model, nn.Module): model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if method in ("sgd", "qat"):
                opt.zero_grad(set_to_none=True); loss = F.cross_entropy(model(x), y); loss.backward(); opt.step()
                losses.append(float(loss.detach())); forwards += 1
            elif method == "binary": losses.append(model.train_step(x, y)); forwards += 1
            else: losses.append(model.train_step(x, y, args.probes)); forwards += 2 * args.probes
            meter.sample()
        if device.type == "cuda": torch.cuda.synchronize(device)
        moves = model.moves if method == "dbs" else (model.flips if method == "binary" else 0)
        peak_alloc = torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else None
        peak_reserved = torch.cuda.max_memory_reserved(device)/2**20 if device.type == "cuda" else None
        acc = accuracy(model, test_loader, device, method)
        rows.append(Row(method, epoch, acc, sum(losses)/len(losses), forwards, moves,
                        time.perf_counter()-started, peak_alloc, peak_reserved,
                        meter.joules if meter.available else None))
        print(f"{method:6s} epoch={epoch} acc={rows[-1].test_accuracy:.4f} loss={rows[-1].train_loss:.4f}")
    return rows


def save(rows, args):
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    records = [asdict(r) for r in rows]
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=records[0]); w.writeheader(); w.writerows(records)
    (out / "results.json").write_text(json.dumps({"config": vars(args), "results": records}, indent=2), encoding="utf-8")
    if args.skip_plots:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for metric, ylabel in (("test_accuracy", "Test accuracy"), ("wall_time_s", "Wall time (s)")):
            plt.figure()
            for method in sorted(set(r.method for r in rows)):
                rs = [r for r in rows if r.method == method]
                plt.plot([r.epoch for r in rs], [getattr(r, metric) for r in rs], marker="o", label=method)
            plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.legend(); plt.grid(alpha=.25); plt.tight_layout()
            plt.savefig(out / f"{metric}.png", dpi=160); plt.close()
    except ImportError: print("matplotlib unavailable: plots skipped")


def parse_args():
    p = argparse.ArgumentParser(description="MNIST benchmark: FP32 SGD, INT8 QAT, Binary/Bop, and DBS")
    p.add_argument("--method", choices=(*METHODS, "all"), default="all")
    p.add_argument("--epochs", type=int, default=10); p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=256); p.add_argument("--lr", type=float, default=.05)
    p.add_argument("--probes", type=int, default=8, help="DBS paired probes R per minibatch")
    p.add_argument("--block-size", type=int, default=4096); p.add_argument("--perturb-fraction", type=float, default=.25)
    p.add_argument("--threshold", type=int, default=3); p.add_argument("--bins", type=int, default=256)
    p.add_argument("--dbs-variant", choices=("legacy", "layer_scaled", "layer_scheduled", "layer_balanced"),
                   default="layer_balanced")
    p.add_argument("--dbs-init-levels", type=float, default=8.0,
                   help="initial weight standard deviation measured in quantization steps")
    p.add_argument("--dbs-evidence-rule", choices=("reset", "residual"), default="reset")
    p.add_argument("--bop-tau", type=float, default=1e-4); p.add_argument("--bop-gamma", type=float, default=.99)
    p.add_argument("--seed", type=int, default=1); p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data-dir", default="data"); p.add_argument("--output-dir", default="runs/latest")
    p.add_argument("--skip-plots", action="store_true", help="save CSV/JSON only")
    p.add_argument("--train-limit", type=int); p.add_argument("--test-limit", type=int)
    p.add_argument("--synthetic", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args = parse_args()
    if not 0 < args.perturb_fraction <= 1: raise ValueError("--perturb-fraction must be in (0, 1]")
    if args.probes < 1: raise ValueError("--probes must be positive")
    train, test = loaders(args); methods = METHODS if args.method == "all" else (args.method,)
    rows = []
    for method in methods: rows.extend(run_method(method, args, train, test))
    save(rows, args)


if __name__ == "__main__": main()
