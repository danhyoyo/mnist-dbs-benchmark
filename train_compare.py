"""Train DBS variants with a validation split and compare held-out MNIST accuracy.

The test set is read only once per trained run, after choosing the best epoch on
validation data. Checkpoints contain persistent DBS integer state for inspection.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from benchmark import DBSMLP, MLP, seed_all


METHODS = ("sgd", "dbs_reset", "dbs_residual")


def datasets(args, seed):
    from torchvision import datasets as tv_datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_all = tv_datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test = tv_datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    n = min(args.train_limit or len(train_all), len(train_all))
    if not 0 < args.val_size < n:
        raise ValueError("val-size must be positive and smaller than the train-limit")
    # Fixed random split for every method at a given seed.
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    val = Subset(train_all, indices[:args.val_size])
    train = Subset(train_all, indices[args.val_size:])
    if args.test_limit:
        test = Subset(test, range(min(args.test_limit, len(test))))
    common = dict(batch_size=args.batch_size, num_workers=args.workers,
                  pin_memory=args.device.startswith("cuda"))
    return train, DataLoader(val, shuffle=False, **common), DataLoader(test, shuffle=False, **common)


def test_only_loader(args):
    from torchvision import datasets as tv_datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
    ])
    test = tv_datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    if args.test_limit:
        test = Subset(test, range(min(args.test_limit, len(test))))
    return DataLoader(test, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.workers, pin_memory=args.device.startswith("cuda"))


def make_model(method, args, device, seed):
    if method == "sgd":
        return MLP(args.hidden).to(device)
    return DBSMLP(args.hidden, device, bins=args.bins, block=args.block_size,
                  fraction=args.perturb_fraction, threshold=args.threshold,
                  seed=seed, variant="layer_balanced", init_levels=args.dbs_init_levels,
                  evidence_rule=method.removeprefix("dbs_"))


@torch.no_grad()
def evaluate(model, loader, device):
    if isinstance(model, MLP):
        model.eval()
    correct = count = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x) if isinstance(model, MLP) else model.predict(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        count += y.numel()
    return {"accuracy": correct / count, "loss": loss_sum / count}


def checkpoint_payload(model, method, args, seed, epoch, validation):
    meta = {"method": method, "hidden": args.hidden, "bins": args.bins,
            "block_size": args.block_size, "perturb_fraction": args.perturb_fraction,
            "threshold": args.threshold, "dbs_init_levels": args.dbs_init_levels,
            "seed": seed, "epoch": epoch, "validation": validation}
    if method == "sgd":
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    else:
        state = {"q": model.q.cpu(), "evidence": model.evidence.cpu(),
                 "cursor": model.cursor, "layer_cursors": model.layer_cursors,
                 "moves": model.moves}
    return {"meta": meta, "state": state}


def load_checkpoint(path, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    m = payload["meta"]
    config = argparse.Namespace(hidden=m["hidden"], bins=m["bins"], block_size=m["block_size"],
                                perturb_fraction=m["perturb_fraction"], threshold=m["threshold"],
                                dbs_init_levels=m["dbs_init_levels"])
    model = make_model(m["method"], config, device, m["seed"])
    state = payload["state"]
    if m["method"] == "sgd":
        model.load_state_dict(state)
    else:
        model.q.copy_(state["q"])
        model.evidence.copy_(state["evidence"])
        model.cursor = state["cursor"]
        model.layer_cursors = state["layer_cursors"]
        model.moves = state["moves"]
    return model, m


def train_one(method, args, seed, train_set, val_loader, test_loader, output):
    seed_all(seed)
    device = torch.device(args.device)
    model = make_model(method, args, device, seed)
    optimizer = (torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.9)
                 if method == "sgd" else None)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed),
                              num_workers=args.workers, pin_memory=args.device.startswith("cuda"))
    checkpoint = output / f"{method}_seed{seed}.pt"
    best_accuracy = -1.0
    history = []
    forwards = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        if method == "sgd":
            model.train()
        loss_total = count = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if method == "sgd":
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                optimizer.step()
                forwards += 1
                value = loss.item()
            else:
                value = model.train_step(x, y, args.probes)
                forwards += 2 * args.probes
            loss_total += value * y.numel()
            count += y.numel()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        val = evaluate(model, val_loader, device)
        elapsed = time.perf_counter() - started
        row = {"epoch": epoch, "train_loss": loss_total / count,
               "val_accuracy": val["accuracy"], "val_loss": val["loss"],
               "training_forwards": forwards, "elapsed_s": elapsed}
        history.append(row)
        if val["accuracy"] > best_accuracy:
            best_accuracy = val["accuracy"]
            torch.save(checkpoint_payload(model, method, args, seed, epoch, val), checkpoint)
        print(f"{method} seed={seed} epoch={epoch} val={val['accuracy']:.4f} loss={val['loss']:.4f}", flush=True)
    selected, meta = load_checkpoint(checkpoint, device)
    test = evaluate(selected, test_loader, device)
    best_epoch = meta["epoch"]
    return {"method": method, "seed": seed, "best_epoch": best_epoch,
            "val_accuracy": meta["validation"]["accuracy"], "test_accuracy": test["accuracy"],
            "test_loss": test["loss"], "training_forwards": history[-1]["training_forwards"],
            "wall_time_s": time.perf_counter() - started,
            "bin_moves": selected.moves if method != "sgd" else None,
            "checkpoint": str(checkpoint.resolve()), "history": history}


def write_results(results, args, output):
    (output / "results.json").write_text(json.dumps({"config": vars(args), "runs": results}, indent=2), encoding="utf-8")
    fields = [k for k in results[0] if k != "history"]
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in results)
    summary = {}
    for method in dict.fromkeys(row["method"] for row in results):
        accuracies = [row["test_accuracy"] for row in results if row["method"] == method]
        summary[method] = {"mean_test_accuracy": statistics.mean(accuracies),
                           "std_test_accuracy": statistics.stdev(accuracies) if len(accuracies) > 1 else None,
                           "seeds": len(accuracies)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=.05)
    parser.add_argument("--probes", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--perturb-fraction", type=float, default=.25)
    parser.add_argument("--threshold", type=int, default=3)
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--dbs-init-levels", type=float, default=8.0)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="runs/compare")
    parser.add_argument("--eval-checkpoint", type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.probes < 1 or args.batch_size < 1:
        parser.error("epochs, probes, and batch-size must be positive")
    if not 0 < args.perturb_fraction <= 1:
        parser.error("perturb-fraction must be in (0, 1]")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.eval_checkpoint:
        model, _ = load_checkpoint(args.eval_checkpoint, torch.device(args.device))
        test_loader = test_only_loader(args)
        print(json.dumps({"checkpoint": str(args.eval_checkpoint), "test": evaluate(model, test_loader, torch.device(args.device))}, indent=2))
        return
    results = []
    for seed in args.seeds:
        train_set, val_loader, test_loader = datasets(args, seed)
        for method in args.methods:
            results.append(train_one(method, args, seed, train_set, val_loader, test_loader, output))
            write_results(results, args, output)


if __name__ == "__main__":
    main()
