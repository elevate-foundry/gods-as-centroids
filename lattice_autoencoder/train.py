"""
Training Loop for Lattice Autoencoder
=======================================

Phase 1: Train single-modal autoencoder on consensus corpus
  - Reconstruction loss (cosine + L2)
  - Classification loss (tradition prediction)
  - Temperature annealing (τ: 1 → 10 over training)
  - Evaluation: reconstruction cosine, classification accuracy, bit utilization

Usage:
  python -m lattice_autoencoder.train --domain theology --epochs 200
  python -m lattice_autoencoder.train --domain theology --epochs 200 --eval-only --checkpoint best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .model import LatticeAutoencoder
from .data import ConsensusCorpusDataset, get_dataloader, AXES
from .lattice import DOMAINS, DOT_NAMES


# ─── Metrics ──────────────────────────────────────────────────────────

def compute_metrics(model, dataloader, device):
    """Compute evaluation metrics on a dataset."""
    model.eval()
    total_cos = 0.0
    total_l2 = 0.0
    correct = 0
    total = 0
    all_z = []

    with torch.no_grad():
        for x, labels in dataloader:
            x, labels = x.to(device), labels.to(device)
            out = model(x, labels)

            # Cosine similarity
            cos_sim = F.cosine_similarity(x, out["x_recon"], dim=-1)
            total_cos += cos_sim.sum().item()

            # L2 distance
            l2 = (x - out["x_recon"]).pow(2).sum(dim=-1).sqrt()
            total_l2 += l2.sum().item()

            # Classification accuracy
            preds = out["logits"].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.shape[0]

            all_z.append(out["z"])

    # Bit utilization: what fraction of bits are ever active?
    all_z = torch.cat(all_z, dim=0)
    bit_means = all_z.mean(dim=0)
    bit_utilization = ((bit_means > 0.01) & (bit_means < 0.99)).float().mean().item()

    # Per-dot-type utilization
    n_axes = model.n_axes
    dot_util = {}
    for d in range(8):
        dot_bits = all_z[:, d::8]  # every 8th bit starting at offset d
        dot_mean = dot_bits.mean().item()
        dot_util[DOT_NAMES[d]] = dot_mean

    return {
        "cosine_sim": total_cos / total,
        "l2_distance": total_l2 / total,
        "accuracy": correct / total,
        "bit_utilization": bit_utilization,
        "dot_utilization": dot_util,
        "n_samples": total,
    }


# ─── Training ─────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader = get_dataloader(
        domain=args.domain,
        batch_size=args.batch_size,
        noise_std=args.noise_std,
        samples_per_passage=args.samples_per_passage,
        split="train",
    )
    val_loader = get_dataloader(
        domain=args.domain,
        batch_size=args.batch_size,
        split="val",
    )

    # Get n_classes from dataset
    train_dataset = train_loader.dataset
    if hasattr(train_dataset, 'n_classes'):
        n_classes = train_dataset.n_classes
    else:
        # TensorDataset fallback
        n_classes = 37

    # Model
    model = LatticeAutoencoder(
        domain=args.domain,
        hidden_dim=args.hidden_dim,
        n_classes=n_classes,
        tau_init=args.tau_init,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    print(f"Lattice: {model.n_axes} axes × 8 dots = {model.n_bits} bits")
    print(f"Domain: {args.domain}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_cos = 0.0
    history = []

    print(f"\n{'='*70}")
    print(f"TRAINING — {args.epochs} epochs, τ: {args.tau_init} → {args.tau_max}")
    print(f"{'='*70}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()

        # Anneal temperature
        model.lattice.anneal_tau(epoch, args.epochs, args.tau_init, args.tau_max)
        tau = model.lattice.tau

        epoch_loss = 0.0
        epoch_samples = 0

        for x, labels in train_loader:
            x, labels = x.to(device), labels.to(device)

            out = model(x, labels)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item() * x.shape[0]
            epoch_samples += x.shape[0]

        scheduler.step()

        avg_loss = epoch_loss / epoch_samples

        # Evaluate periodically
        if epoch % args.eval_every == 0 or epoch == 1 or epoch == args.epochs:
            val_metrics = compute_metrics(model, val_loader, device)
            train_metrics = compute_metrics(model, train_loader, device)

            record = {
                "epoch": epoch,
                "tau": round(tau, 2),
                "lr": round(scheduler.get_last_lr()[0], 6),
                "train_loss": round(avg_loss, 4),
                "train_cos": round(train_metrics["cosine_sim"], 4),
                "train_acc": round(train_metrics["accuracy"], 4),
                "val_cos": round(val_metrics["cosine_sim"], 4),
                "val_acc": round(val_metrics["accuracy"], 4),
                "bit_util": round(val_metrics["bit_utilization"], 4),
            }
            history.append(record)

            # Print
            print(f"  Epoch {epoch:>4d}/{args.epochs}  τ={tau:.1f}  "
                  f"loss={avg_loss:.4f}  "
                  f"cos={val_metrics['cosine_sim']:.4f}  "
                  f"acc={val_metrics['accuracy']:.1%}  "
                  f"bits={val_metrics['bit_utilization']:.1%}")

            # Save best
            if val_metrics["cosine_sim"] > best_val_cos:
                best_val_cos = val_metrics["cosine_sim"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "args": vars(args),
                }, out_dir / "best.pt")

    # Save final model
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "val_metrics": val_metrics,
        "args": vars(args),
    }, out_dir / "final.pt")

    # Save training history
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ─── Final evaluation ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FINAL EVALUATION")
    print(f"{'='*70}")

    # Load best model
    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    final_metrics = compute_metrics(model, val_loader, device)

    print(f"\n  Reconstruction cosine:  {final_metrics['cosine_sim']:.4f}")
    print(f"  Classification accuracy: {final_metrics['accuracy']:.1%}")
    print(f"  Bit utilization:         {final_metrics['bit_utilization']:.1%}")
    print(f"  L2 distance:             {final_metrics['l2_distance']:.4f}")

    print(f"\n  Per-dot utilization (fraction of 1s):")
    for dot_name, util in final_metrics["dot_utilization"].items():
        bar = "█" * int(util * 40)
        print(f"    {dot_name:<20s} {util:.3f}  {bar}")

    # Generate braille signatures for each tradition
    print(f"\n  Tradition braille signatures (from lattice):")
    model.eval()
    dataset = ConsensusCorpusDataset(augment=False, samples_per_passage=1)
    tradition_vecs = {}
    for i in range(len(dataset.base_vectors)):
        trad = dataset.traditions[i]
        if trad not in tradition_vecs:
            tradition_vecs[trad] = []
        tradition_vecs[trad].append(dataset.base_vectors[i])

    with torch.no_grad():
        for trad in sorted(tradition_vecs.keys()):
            vecs = torch.stack(tradition_vecs[trad]).to(device)
            z = model.encode(vecs)
            # Hamming centroid across passages
            centroid = model.lattice.hamming_centroid(z)
            braille = model.lattice.to_unicode(centroid.unsqueeze(0))[0]
            print(f"    {trad:<25s} {braille}")

    print(f"\n  Results saved to {out_dir}/")
    print()


# ─── Evaluation Only ──────────────────────────────────────────────────

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    val_loader = get_dataloader(domain=args.domain, batch_size=args.batch_size, split="val")

    # Determine n_classes
    dataset = ConsensusCorpusDataset(augment=False, samples_per_passage=1)
    n_classes = dataset.n_classes

    model = LatticeAutoencoder(
        domain=args.domain,
        hidden_dim=args.hidden_dim,
        n_classes=n_classes,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    metrics = compute_metrics(model, val_loader, device)
    print(json.dumps(metrics, indent=2, default=str))


# ─── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Lattice Autoencoder")
    parser.add_argument("--domain", default="theology", choices=["theology", "political", "personality"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--samples-per-passage", type=int, default=50)
    parser.add_argument("--tau-init", type=float, default=1.0)
    parser.add_argument("--tau-max", type=float, default=10.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--output-dir", default="lattice_autoencoder/runs/theology")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    if args.eval_only:
        if args.checkpoint is None:
            args.checkpoint = f"{args.output_dir}/best.pt"
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
