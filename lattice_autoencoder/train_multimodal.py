"""
Phase 2 + Phase 3 Training for Multi-Modal Lattice Autoencoder
================================================================

Phase 2: Train text + scorer encoders to align on shared lattice
Phase 3: Train differentiable operator layer (fusion, fission, perturbation)

Usage:
  python -m lattice_autoencoder.train_multimodal --phase 2 --epochs 300
  python -m lattice_autoencoder.train_multimodal --phase 3 --epochs 200
  python -m lattice_autoencoder.train_multimodal --phase both --epochs 300
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from .model import (
    MultiModalLatticeAutoencoder,
    LatticeOperatorModel,
    LatticeAutoencoder,
)
from .data import MultiModalDataset, ConsensusCorpusDataset, AXES
from .lattice import DOMAINS, DOT_NAMES


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: Multi-Modal Training
# ═══════════════════════════════════════════════════════════════════════

def train_phase2(args):
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_dataset = MultiModalDataset(
        noise_std=args.noise_std,
        augment=True,
        samples_per_passage=args.samples_per_passage,
    )
    val_dataset = MultiModalDataset(
        noise_std=0.0,
        augment=False,
        samples_per_passage=1,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    text_dim = train_dataset.text_dim
    n_classes = train_dataset.n_classes

    print(f"Text embed dim: {text_dim}")
    print(f"Traditions: {n_classes}")
    print(f"Scorers: {train_dataset.n_scorers} ({', '.join(train_dataset.scorer_names)})")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Model
    model = MultiModalLatticeAutoencoder(
        domain="theology",
        text_embed_dim=text_dim,
        hidden_dim=args.hidden_dim,
        n_classes=n_classes,
        tau_init=args.tau_init,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    print(f"Lattice: {model.n_axes} axes × 8 dots = {model.n_bits} bits")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Output
    out_dir = Path(args.output_dir) / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_align = float("inf")
    history = []

    print(f"\n{'='*70}")
    print(f"PHASE 2 — Multi-Modal Training ({args.epochs} epochs)")
    print(f"{'='*70}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.lattice.anneal_tau(epoch, args.epochs, args.tau_init, args.tau_max)
        tau = model.lattice.tau

        epoch_loss = 0.0
        epoch_samples = 0

        for text_embed, scorer_vec, consensus_vec, labels in train_loader:
            text_embed = text_embed.to(device)
            scorer_vec = scorer_vec.to(device)
            consensus_vec = consensus_vec.to(device)
            labels = labels.to(device)

            out = model(
                text_embed=text_embed,
                scorer_vec=scorer_vec,
                consensus_target=consensus_vec,
                labels=labels,
            )
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * text_embed.shape[0]
            epoch_samples += text_embed.shape[0]

        scheduler.step()
        avg_loss = epoch_loss / epoch_samples

        # Evaluate
        if epoch % args.eval_every == 0 or epoch == 1 or epoch == args.epochs:
            metrics = eval_phase2(model, val_loader, device)

            record = {
                "epoch": epoch,
                "tau": round(tau, 2),
                "loss": round(avg_loss, 4),
                **{k: round(v, 4) for k, v in metrics.items()},
            }
            history.append(record)

            print(f"  Epoch {epoch:>4d}/{args.epochs}  τ={tau:.1f}  "
                  f"loss={avg_loss:.4f}  "
                  f"align={metrics['bit_agreement']:.1%}  "
                  f"cos_text={metrics['cos_text']:.4f}  "
                  f"cos_scorer={metrics['cos_scorer']:.4f}  "
                  f"acc_text={metrics['acc_text']:.1%}  "
                  f"acc_scorer={metrics['acc_scorer']:.1%}")

            if metrics.get("alignment_loss", float("inf")) < best_val_align:
                best_val_align = metrics["alignment_loss"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                    "args": vars(args),
                }, out_dir / "best.pt")

    # Save final
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "args": vars(args),
    }, out_dir / "final.pt")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Final report
    print(f"\n{'='*70}")
    print("PHASE 2 FINAL EVALUATION")
    print(f"{'='*70}")

    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    final = eval_phase2(model, val_loader, device)

    print(f"\n  Cross-modal bit agreement: {final['bit_agreement']:.1%}")
    print(f"  Text → consensus cosine:   {final['cos_text']:.4f}")
    print(f"  Scorer → consensus cosine:  {final['cos_scorer']:.4f}")
    print(f"  Consensus → consensus cos:  {final.get('cos_consensus', 0):.4f}")
    print(f"  Text classification acc:    {final['acc_text']:.1%}")
    print(f"  Scorer classification acc:  {final['acc_scorer']:.1%}")
    print(f"  Consensus classification:   {final.get('acc_consensus', 0):.1%}")

    # Show tradition braille signatures from both modalities
    print(f"\n  Tradition signatures (text vs scorer modality):")
    model.eval()
    with torch.no_grad():
        tradition_text_z = {}
        tradition_scorer_z = {}
        for text_embed, scorer_vec, consensus_vec, labels in val_loader:
            text_embed = text_embed.to(device)
            scorer_vec = scorer_vec.to(device)
            z_text = model.encode_text(text_embed)
            z_scorer = model.encode_scorer(scorer_vec)
            for i in range(labels.shape[0]):
                trad = val_dataset.idx_to_tradition[labels[i].item()]
                if trad not in tradition_text_z:
                    tradition_text_z[trad] = []
                    tradition_scorer_z[trad] = []
                tradition_text_z[trad].append(z_text[i])
                tradition_scorer_z[trad].append(z_scorer[i])

        for trad in sorted(tradition_text_z.keys())[:12]:
            zt = torch.stack(tradition_text_z[trad])
            zs = torch.stack(tradition_scorer_z[trad])
            ct = model.lattice.hamming_centroid(zt)
            cs = model.lattice.hamming_centroid(zs)
            bt = model.lattice.to_unicode(ct.unsqueeze(0))[0]
            bs = model.lattice.to_unicode(cs.unsqueeze(0))[0]
            agree = (ct == cs).float().mean().item()
            print(f"    {trad:<22s} text={bt}  scorer={bs}  agree={agree:.0%}")

    print(f"\n  Results saved to {out_dir}/")
    return model


def eval_phase2(model, dataloader, device):
    model.eval()
    total_cos_text = 0.0
    total_cos_scorer = 0.0
    total_cos_consensus = 0.0
    correct_text = 0
    correct_scorer = 0
    correct_consensus = 0
    total = 0
    total_align = 0.0
    total_bit_agree = 0.0

    with torch.no_grad():
        for text_embed, scorer_vec, consensus_vec, labels in dataloader:
            text_embed = text_embed.to(device)
            scorer_vec = scorer_vec.to(device)
            consensus_vec = consensus_vec.to(device)
            labels = labels.to(device)

            out = model(
                text_embed=text_embed,
                scorer_vec=scorer_vec,
                consensus_target=consensus_vec,
                labels=labels,
            )

            bs = labels.shape[0]

            # Reconstruction cosine
            total_cos_text += F.cosine_similarity(
                consensus_vec, out["recon_text"], dim=-1
            ).sum().item()
            total_cos_scorer += F.cosine_similarity(
                consensus_vec, out["recon_scorer"], dim=-1
            ).sum().item()
            if "recon_consensus" in out:
                total_cos_consensus += F.cosine_similarity(
                    consensus_vec, out["recon_consensus"], dim=-1
                ).sum().item()

            # Classification
            correct_text += (out["logits_text"].argmax(-1) == labels).sum().item()
            correct_scorer += (out["logits_scorer"].argmax(-1) == labels).sum().item()
            if "logits_consensus" in out:
                correct_consensus += (out["logits_consensus"].argmax(-1) == labels).sum().item()

            # Alignment
            align = F.mse_loss(out["z_text"], out["z_scorer"]).item()
            total_align += align * bs

            # Bit agreement
            bit_agree = (out["z_text"] == out["z_scorer"]).float().mean().item()
            total_bit_agree += bit_agree * bs

            total += bs

    return {
        "cos_text": total_cos_text / total,
        "cos_scorer": total_cos_scorer / total,
        "cos_consensus": total_cos_consensus / total,
        "acc_text": correct_text / total,
        "acc_scorer": correct_scorer / total,
        "acc_consensus": correct_consensus / total,
        "alignment_loss": total_align / total,
        "bit_agreement": total_bit_agree / total,
    }


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: Operator Training
# ═══════════════════════════════════════════════════════════════════════

# Tradition family mappings for generating operator examples
TRADITION_FAMILIES = {
    "abrahamic": ["Judaism", "Christianity", "Islam", "Bahai", "Druze", "Samaritanism"],
    "dharmic": ["Hinduism", "Buddhism", "Jainism", "Sikhism"],
    "east_asian": ["Daoism", "Confucianism", "Shinto"],
    "african": ["Yoruba", "Akan", "Kemetic"],
    "african_diaspora": ["Candomble", "Vodou", "Rastafari"],
    "indigenous": ["Lakota", "Navajo", "Aboriginal Australian", "Maori", "Hawaiian"],
    "mesoamerican": ["Nahua", "Maya", "Inca"],
}

# Known historical schisms
SCHISM_PAIRS = [
    ("Judaism", "Christianity"),
    ("Christianity", "Islam"),
    ("Hinduism", "Buddhism"),
    ("Hinduism", "Jainism"),
    ("Buddhism", "Jainism"),
    ("Vodou", "Candomble"),
]


def generate_operator_dataset(encoder_model, dataset, device):
    """
    Generate operator training examples from the corpus.

    Returns: z_a, z_b, op_labels, target_vecs, target_labels
    """
    encoder_model.eval()

    # First, encode all passages to lattice codes
    all_z = []
    all_vecs = []
    all_labels = []
    all_traditions = []

    with torch.no_grad():
        for i in range(dataset.n_passages):
            vec = dataset.consensus_vecs[i:i+1].to(device)
            # Use the Phase 1 encoder if available, otherwise scorer encoder
            if hasattr(encoder_model, 'encode'):
                z = encoder_model.encode(vec)
            elif hasattr(encoder_model, 'encode_scorer'):
                z = encoder_model.encode_scorer(vec)
            else:
                raise ValueError("Encoder model must have encode() or encode_scorer()")
            all_z.append(z.squeeze(0))
            all_vecs.append(dataset.consensus_vecs[i])
            all_labels.append(dataset.labels[i])
            all_traditions.append(dataset.traditions_list[i])

    all_z = torch.stack(all_z)
    all_vecs = torch.stack(all_vecs)

    # Build tradition → indices mapping
    trad_to_idx = {}
    for i, t in enumerate(all_traditions):
        if t not in trad_to_idx:
            trad_to_idx[t] = []
        trad_to_idx[t].append(i)

    # Build family → traditions mapping
    trad_to_family = {}
    for family, trads in TRADITION_FAMILIES.items():
        for t in trads:
            trad_to_family[t] = family

    z_a_list, z_b_list = [], []
    op_labels_list = []
    target_vecs_list = []
    target_labels_list = []

    rng = random.Random(42)

    # ─── Fusion examples (op=0): same-family tradition pairs ──────
    for family, trads in TRADITION_FAMILIES.items():
        available = [t for t in trads if t in trad_to_idx]
        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                t_a, t_b = available[i], available[j]
                idx_a = rng.choice(trad_to_idx[t_a])
                idx_b = rng.choice(trad_to_idx[t_b])
                z_a_list.append(all_z[idx_a])
                z_b_list.append(all_z[idx_b])
                op_labels_list.append(0)  # fusion
                # Target: midpoint of the two traditions
                target = (all_vecs[idx_a] + all_vecs[idx_b]) / 2.0
                target_vecs_list.append(target)
                # Label: first tradition (arbitrary for fusion)
                target_labels_list.append(all_labels[idx_a])

    # ─── Fission examples (op=1): known schism pairs ─────────────
    for t_a, t_b in SCHISM_PAIRS:
        if t_a in trad_to_idx and t_b in trad_to_idx:
            for _ in range(3):  # multiple examples per schism
                idx_a = rng.choice(trad_to_idx[t_a])
                idx_b = rng.choice(trad_to_idx[t_b])
                z_a_list.append(all_z[idx_a])
                z_b_list.append(all_z[idx_b])
                op_labels_list.append(1)  # fission
                target_vecs_list.append(all_vecs[idx_a])
                target_labels_list.append(all_labels[idx_a])

    # ─── Perturbation examples (op=2): tradition + distant outlier ─
    all_traditions_unique = list(trad_to_idx.keys())
    for t in all_traditions_unique:
        if t not in trad_to_family:
            continue
        family = trad_to_family[t]
        # Find a distant tradition (different family)
        distant = [t2 for t2 in all_traditions_unique
                   if t2 in trad_to_family and trad_to_family[t2] != family]
        if distant:
            for _ in range(2):
                t_distant = rng.choice(distant)
                idx_a = rng.choice(trad_to_idx[t])
                idx_b = rng.choice(trad_to_idx[t_distant])
                z_a_list.append(all_z[idx_a])
                z_b_list.append(all_z[idx_b])
                op_labels_list.append(2)  # perturbation
                target_vecs_list.append(all_vecs[idx_a])
                target_labels_list.append(all_labels[idx_a])

    # ─── None examples (op=3): same tradition, similar passages ───
    for t in all_traditions_unique:
        indices = trad_to_idx[t]
        if len(indices) >= 2:
            for _ in range(2):
                idx_a, idx_b = rng.sample(indices, 2)
                z_a_list.append(all_z[idx_a])
                z_b_list.append(all_z[idx_b])
                op_labels_list.append(3)  # none
                target_vecs_list.append(all_vecs[idx_a])
                target_labels_list.append(all_labels[idx_a])

    z_a = torch.stack(z_a_list)
    z_b = torch.stack(z_b_list)
    op_labels = torch.tensor(op_labels_list, dtype=torch.long)
    target_vecs = torch.stack(target_vecs_list)
    target_labels = torch.tensor(target_labels_list, dtype=torch.long)

    # Print distribution
    for op, name in enumerate(["fusion", "fission", "perturbation", "none"]):
        count = (op_labels == op).sum().item()
        print(f"    {name}: {count} examples")

    return z_a, z_b, op_labels, target_vecs, target_labels


def train_phase3(args, phase2_model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load encoder for generating lattice codes
    if phase2_model is not None:
        encoder = phase2_model
    else:
        # Try loading Phase 2 model, fall back to Phase 1
        phase2_path = Path(args.output_dir) / "phase2" / "best.pt"
        phase1_path = Path(args.output_dir).parent / "runs" / "theology" / "best.pt"

        if phase2_path.exists():
            print("Loading Phase 2 encoder...")
            dataset = MultiModalDataset(augment=False, samples_per_passage=1)
            encoder = MultiModalLatticeAutoencoder(
                text_embed_dim=dataset.text_dim,
                n_classes=dataset.n_classes,
            ).to(device)
            ckpt = torch.load(phase2_path, map_location=device, weights_only=False)
            encoder.load_state_dict(ckpt["model_state_dict"])
        elif phase1_path.exists():
            print("Loading Phase 1 encoder...")
            dataset_p1 = ConsensusCorpusDataset(augment=False, samples_per_passage=1)
            encoder = LatticeAutoencoder(
                domain="theology",
                n_classes=dataset_p1.n_classes,
            ).to(device)
            ckpt = torch.load(phase1_path, map_location=device, weights_only=False)
            encoder.load_state_dict(ckpt["model_state_dict"])
        else:
            raise FileNotFoundError("No Phase 1 or Phase 2 checkpoint found. Train Phase 1 or 2 first.")

    encoder.eval()

    # Load corpus dataset for generating operator examples
    corpus = MultiModalDataset(augment=False, samples_per_passage=1)

    print("\n  Generating operator training examples...")
    z_a, z_b, op_labels, target_vecs, target_labels = generate_operator_dataset(
        encoder, corpus, device
    )
    n_examples = len(op_labels)
    print(f"  Total: {n_examples} operator examples")

    # Train/val split
    perm = torch.randperm(n_examples)
    n_val = max(1, int(n_examples * 0.15))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    # Augment training data by repeating with noise
    z_a_train = z_a[train_idx]
    z_b_train = z_b[train_idx]
    op_train = op_labels[train_idx]
    tv_train = target_vecs[train_idx]
    tl_train = target_labels[train_idx]

    # Repeat for more training data
    n_repeats = max(1, args.samples_per_passage)
    z_a_train = z_a_train.repeat(n_repeats, 1)
    z_b_train = z_b_train.repeat(n_repeats, 1)
    op_train = op_train.repeat(n_repeats)
    tv_train = tv_train.repeat(n_repeats, 1)
    tl_train = tl_train.repeat(n_repeats)

    train_ds = TensorDataset(z_a_train, z_b_train, op_train, tv_train, tl_train)
    val_ds = TensorDataset(z_a[val_idx], z_b[val_idx], op_labels[val_idx],
                           target_vecs[val_idx], target_labels[val_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Model
    n_bits = 96
    n_axes = 12
    model = LatticeOperatorModel(
        n_bits=n_bits,
        n_axes=n_axes,
        n_classes=corpus.n_classes,
        hidden_dim=args.hidden_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Operator model: {n_params:,} parameters")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    out_dir = Path(args.output_dir) / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    history = []

    print(f"\n{'='*70}")
    print(f"PHASE 3 — Operator Training ({args.epochs} epochs)")
    print(f"{'='*70}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for za, zb, ops, tvecs, tlabels in train_loader:
            za, zb = za.to(device), zb.to(device)
            ops = ops.to(device)
            tvecs, tlabels = tvecs.to(device), tlabels.to(device)

            out = model(za, zb, op_labels=ops, target_vecs=tvecs, target_labels=tlabels)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * za.shape[0]
            epoch_correct += (out["op_logits"].argmax(-1) == ops).sum().item()
            epoch_total += za.shape[0]

        scheduler.step()
        avg_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total

        if epoch % args.eval_every == 0 or epoch == 1 or epoch == args.epochs:
            val_metrics = eval_phase3(model, val_loader, device)

            record = {
                "epoch": epoch,
                "loss": round(avg_loss, 4),
                "train_acc": round(train_acc, 4),
                **{k: round(v, 4) for k, v in val_metrics.items()},
            }
            history.append(record)

            print(f"  Epoch {epoch:>4d}/{args.epochs}  "
                  f"loss={avg_loss:.4f}  "
                  f"train_acc={train_acc:.1%}  "
                  f"val_acc={val_metrics['op_accuracy']:.1%}  "
                  f"fusion_cos={val_metrics.get('fusion_cos', 0):.4f}")

            if val_metrics["op_accuracy"] > best_val_acc:
                best_val_acc = val_metrics["op_accuracy"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": val_metrics,
                    "args": vars(args),
                }, out_dir / "best.pt")

    # Save
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "args": vars(args),
    }, out_dir / "final.pt")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Final report
    print(f"\n{'='*70}")
    print("PHASE 3 FINAL EVALUATION")
    print(f"{'='*70}")

    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    final = eval_phase3(model, val_loader, device)

    print(f"\n  Operator prediction accuracy: {final['op_accuracy']:.1%}")
    print(f"  Per-operator accuracy:")
    for op, name in enumerate(["fusion", "fission", "perturbation", "none"]):
        acc = final.get(f"acc_{name}", 0)
        print(f"    {name:<15s} {acc:.1%}")
    print(f"\n  Fusion reconstruction cosine: {final.get('fusion_cos', 0):.4f}")
    print(f"  Fission child distance:       {final.get('fission_dist', 0):.4f}")
    print(f"  Perturbation pull ratio:      {final.get('perturb_ratio', 0):.4f}")

    # Demo: apply operators to example tradition pairs
    print(f"\n  Operator demos:")
    model.eval()
    with torch.no_grad():
        demos = [
            ("Judaism + Christianity → fusion", 0),
            ("Hinduism + Buddhism → fission", 1),
            ("Daoism + Yoruba → perturbation", 2),
        ]
        for i, (desc, expected_op) in enumerate(demos):
            if i < len(val_ds):
                za_demo = val_ds[i][0].unsqueeze(0).to(device)
                zb_demo = val_ds[i][1].unsqueeze(0).to(device)
                output, pred_op = model.predict_and_apply(za_demo, zb_demo)
                op_names = ["fusion", "fission", "perturbation", "none"]
                print(f"    {desc}")
                print(f"      Predicted: {op_names[pred_op[0].item()]}")
                hamming = (za_demo != zb_demo).float().sum().item()
                out_hamming_a = (output != za_demo).float().sum().item()
                out_hamming_b = (output != zb_demo).float().sum().item()
                print(f"      Input distance: {hamming:.0f} bits")
                print(f"      Output distance from A: {out_hamming_a:.0f}, from B: {out_hamming_b:.0f}")

    print(f"\n  Results saved to {out_dir}/")
    return model


def eval_phase3(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    per_op_correct = [0, 0, 0, 0]
    per_op_total = [0, 0, 0, 0]
    fusion_cos_sum = 0.0
    fusion_count = 0
    fission_dist_sum = 0.0
    fission_count = 0
    perturb_ratio_sum = 0.0
    perturb_count = 0

    with torch.no_grad():
        for za, zb, ops, tvecs, tlabels in dataloader:
            za, zb = za.to(device), zb.to(device)
            ops = ops.to(device)
            tvecs, tlabels = tvecs.to(device), tlabels.to(device)

            out = model(za, zb, op_labels=ops, target_vecs=tvecs, target_labels=tlabels)

            preds = out["op_logits"].argmax(-1)
            correct += (preds == ops).sum().item()
            total += ops.shape[0]

            for op in range(4):
                mask = (ops == op)
                per_op_correct[op] += (preds[mask] == ops[mask]).sum().item()
                per_op_total[op] += mask.sum().item()

            # Fusion quality
            fusion_mask = (ops == 0)
            if fusion_mask.sum() > 0:
                cos = F.cosine_similarity(
                    tvecs[fusion_mask], out["recon_fused"][fusion_mask], dim=-1
                )
                fusion_cos_sum += cos.sum().item()
                fusion_count += fusion_mask.sum().item()

            # Fission quality
            fission_mask = (ops == 1)
            if fission_mask.sum() > 0:
                child_dist = (out["z_fiss_a"][fission_mask] - out["z_fiss_b"][fission_mask]).abs().mean(dim=-1)
                fission_dist_sum += child_dist.sum().item()
                fission_count += fission_mask.sum().item()

            # Perturbation quality
            perturb_mask = (ops == 2)
            if perturb_mask.sum() > 0:
                orig_dist = (za[perturb_mask] - zb[perturb_mask]).abs().mean(dim=-1)
                new_dist = (out["z_perturbed"][perturb_mask] - zb[perturb_mask]).abs().mean(dim=-1)
                ratio = (new_dist / (orig_dist + 1e-8))
                perturb_ratio_sum += ratio.sum().item()
                perturb_count += perturb_mask.sum().item()

    op_names = ["fusion", "fission", "perturbation", "none"]
    result = {
        "op_accuracy": correct / max(1, total),
    }
    for op, name in enumerate(op_names):
        result[f"acc_{name}"] = per_op_correct[op] / max(1, per_op_total[op])

    result["fusion_cos"] = fusion_cos_sum / max(1, fusion_count)
    result["fission_dist"] = fission_dist_sum / max(1, fission_count)
    result["perturb_ratio"] = perturb_ratio_sum / max(1, perturb_count)

    return result


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 2+3 Training")
    parser.add_argument("--phase", default="both", choices=["2", "3", "both"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--samples-per-passage", type=int, default=50)
    parser.add_argument("--tau-init", type=float, default=1.0)
    parser.add_argument("--tau-max", type=float, default=10.0)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--output-dir", default="lattice_autoencoder/runs")
    args = parser.parse_args()

    phase2_model = None

    if args.phase in ("2", "both"):
        phase2_model = train_phase2(args)

    if args.phase in ("3", "both"):
        # Phase 3 uses fewer epochs by default
        phase3_args = argparse.Namespace(**vars(args))
        if args.phase == "both":
            phase3_args.epochs = min(200, args.epochs)
        train_phase3(phase3_args, phase2_model=phase2_model)


if __name__ == "__main__":
    main()
