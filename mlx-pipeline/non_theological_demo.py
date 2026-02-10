#!/usr/bin/env python3
"""
Non-Theological Braille Lattice Demo — Generalizability Proof

Applies the SAME 72-bit braille bottleneck architecture to two completely
different domains to prove the lattice is not theology-specific:

  Domain 1: POLITICAL IDEOLOGIES (10 dimensions, Moral Foundations + policy axes)
  Domain 2: PERSONALITY TYPES (Big Five, 5 dimensions)

If the braille lattice preserves structure in these domains as well as it does
for theology, the architecture generalizes — and the AGI bridge claim has teeth.

Architecture is identical:
  Encoder(ℝ^D) → BrailleQuantizer({0,1}^N) → Decoder(ℝ^D) + Classifier
"""

import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# ─── Domain 1: Political Ideologies ──────────────────────────────────
# 10 axes derived from Moral Foundations Theory + standard policy dimensions

POLITICAL_AXES = [
    "care_harm",           # MFT: sensitivity to suffering
    "fairness_cheating",   # MFT: proportionality, equality
    "loyalty_betrayal",    # MFT: in-group solidarity
    "authority_subversion", # MFT: respect for hierarchy
    "sanctity_degradation", # MFT: purity, disgust sensitivity
    "liberty_oppression",  # MFT (Haidt 2012 addition)
    "economic_left_right", # redistribution vs free market
    "social_progressive_conservative", # social liberalism vs traditionalism
    "internationalist_nationalist",    # globalism vs nationalism
    "institutional_trust",  # trust in institutions vs anti-establishment
]

IDEOLOGY_PRIORS = {
    "Social Democrat":     [0.9, 0.9, 0.4, 0.3, 0.2, 0.7, 0.2, 0.8, 0.8, 0.7],
    "Libertarian":         [0.4, 0.5, 0.3, 0.2, 0.2, 0.95, 0.9, 0.6, 0.5, 0.2],
    "Conservative":        [0.4, 0.4, 0.8, 0.9, 0.9, 0.5, 0.8, 0.2, 0.3, 0.6],
    "Progressive":         [0.95, 0.9, 0.3, 0.2, 0.1, 0.8, 0.1, 0.95, 0.9, 0.5],
    "Nationalist":         [0.3, 0.3, 0.95, 0.8, 0.8, 0.4, 0.5, 0.2, 0.1, 0.4],
    "Centrist":            [0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6],
    "Authoritarian Left":  [0.7, 0.8, 0.7, 0.7, 0.3, 0.2, 0.1, 0.6, 0.4, 0.8],
    "Green":               [0.9, 0.8, 0.4, 0.2, 0.5, 0.7, 0.3, 0.9, 0.8, 0.4],
    "Populist Right":      [0.3, 0.3, 0.9, 0.7, 0.8, 0.6, 0.6, 0.1, 0.1, 0.1],
    "Classical Liberal":   [0.5, 0.7, 0.3, 0.4, 0.3, 0.9, 0.8, 0.7, 0.7, 0.5],
    "Theocratic":          [0.4, 0.3, 0.8, 0.95, 0.95, 0.2, 0.4, 0.05, 0.3, 0.7],
    "Anarchist":           [0.6, 0.7, 0.5, 0.05, 0.1, 0.95, 0.2, 0.8, 0.6, 0.05],
}

# ─── Domain 2: Personality Types (Big Five) ──────────────────────────

PERSONALITY_AXES = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]

PERSONALITY_PRIORS = {
    "The Explorer":     [0.95, 0.4, 0.7, 0.6, 0.3],
    "The Organizer":    [0.3, 0.95, 0.5, 0.6, 0.3],
    "The Leader":       [0.6, 0.7, 0.95, 0.4, 0.2],
    "The Caregiver":    [0.5, 0.6, 0.6, 0.95, 0.4],
    "The Worrier":      [0.4, 0.5, 0.3, 0.5, 0.95],
    "The Maverick":     [0.9, 0.2, 0.8, 0.3, 0.4],
    "The Diplomat":     [0.6, 0.7, 0.7, 0.9, 0.3],
    "The Analyst":      [0.8, 0.8, 0.3, 0.4, 0.5],
    "The Free Spirit":  [0.95, 0.2, 0.6, 0.7, 0.5],
    "The Stoic":        [0.4, 0.8, 0.3, 0.5, 0.1],
}


# ─── Braille Lattice (generalized) ──────────────────────────────────

POLARITY_PAIRS_POLITICAL = {
    "care_harm": "authority_subversion",
    "authority_subversion": "care_harm",
    "fairness_cheating": "loyalty_betrayal",
    "loyalty_betrayal": "fairness_cheating",
    "sanctity_degradation": "liberty_oppression",
    "liberty_oppression": "sanctity_degradation",
    "economic_left_right": "social_progressive_conservative",
    "social_progressive_conservative": "economic_left_right",
    "internationalist_nationalist": "institutional_trust",
    "institutional_trust": "internationalist_nationalist",
}

POLARITY_PAIRS_PERSONALITY = {
    "openness": "conscientiousness",
    "conscientiousness": "openness",
    "extraversion": "neuroticism",
    "neuroticism": "extraversion",
    "agreeableness": "agreeableness",  # self-paired
}


@dataclass
class BrailleCell:
    dots: list  # 6 bools

    def to_unicode(self) -> str:
        code = 0x2800
        offsets = [1, 2, 4, 8, 16, 32]
        for i, d in enumerate(self.dots):
            if d:
                code += offsets[i]
        return chr(code)


def encode_axis(value: float, opposite_value: float) -> BrailleCell:
    pos_active = value > 0.3
    neg_active = opposite_value > value + 0.1
    tension = pos_active and opposite_value > 0.3 and abs(value - opposite_value) < 0.15
    intensity = min(3, int(value * 4))
    dot4 = (intensity & 2) != 0
    dot5 = (intensity & 1) != 0
    rigid = value > 0.7
    return BrailleCell(dots=[pos_active, neg_active, tension, dot4, dot5, rigid])


def decode_axis(cell: BrailleCell) -> float:
    pos_active, neg_active, tension, dot4, dot5, rigid = cell.dots
    intensity = (2 if dot4 else 0) + (1 if dot5 else 0)
    value = (intensity + 0.5) / 4
    if not pos_active and neg_active:
        value *= 0.3
    if tension:
        value *= 0.85
    if rigid:
        value = max(value, 0.75)
    return value


def encode_vector(vec: list, axes: list, polarity_pairs: dict) -> list:
    cells = []
    for i, axis in enumerate(axes):
        opp = polarity_pairs.get(axis, axis)
        opp_idx = axes.index(opp) if opp in axes else i
        cell = encode_axis(vec[i], vec[opp_idx])
        cells.append(cell)
    return cells


def decode_vector(cells: list) -> list:
    values = [decode_axis(c) for c in cells]
    n = math.sqrt(sum(v * v for v in values)) or 1
    return [v / n for v in values]


def braille_signature(cells: list) -> str:
    return "".join(c.to_unicode() for c in cells)


def to_bitstring(cells: list) -> list:
    bits = []
    for cell in cells:
        bits.extend(cell.dots)
    return bits


def cosine_sim(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1
    nb = math.sqrt(sum(x * x for x in b)) or 1
    return dot / (na * nb)


def hamming_distance(a: list, b: list) -> int:
    ba = to_bitstring(a)
    bb = to_bitstring(b)
    return sum(1 for x, y in zip(ba, bb) if x != y)


def normalize(vec: list) -> list:
    n = math.sqrt(sum(v * v for v in vec)) or 1
    return [v / n for v in vec]


# ─── Bottleneck Model (PyTorch) ─────────────────────────────────────

def run_bottleneck_experiment(
    domain_name: str,
    axes: list,
    priors: dict,
    polarity_pairs: dict,
    bits_per_axis: int = 6,
    noise_std: float = 0.08,
    samples_per_class: int = 500,
    epochs: int = 200,
):
    """Run the full bottleneck experiment for a given domain."""
    import torch
    import torch.nn as nn
    import torch.optim as optim

    device = torch.device("cpu")
    dim = len(axes)
    total_bits = dim * bits_per_axis
    num_classes = len(priors)
    names = list(priors.keys())

    print(f"\n{'='*60}")
    print(f"DOMAIN: {domain_name}")
    print(f"Dimensions: {dim}, Bits: {total_bits}, Classes: {num_classes}")
    print(f"{'='*60}")

    # ─── Generate data ───────────────────────────────────────────
    all_vecs = []
    all_labels = []
    for label_idx, (name, prior) in enumerate(priors.items()):
        prior_norm = normalize(prior)
        for _ in range(samples_per_class):
            noisy = [max(0, min(1, p + np.random.normal(0, noise_std))) for p in prior_norm]
            noisy = normalize(noisy)
            all_vecs.append(noisy)
            all_labels.append(label_idx)

    all_vecs = np.array(all_vecs, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)

    indices = np.random.permutation(len(all_vecs))
    split = int(0.8 * len(indices))
    X_train = torch.tensor(all_vecs[indices[:split]])
    y_train = torch.tensor(all_labels[indices[:split]])
    X_test = torch.tensor(all_vecs[indices[split:]])
    y_test = torch.tensor(all_labels[indices[split:]])

    # ─── Model ───────────────────────────────────────────────────
    class BottleneckModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(dim, 64), nn.ReLU(), nn.BatchNorm1d(64),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, total_bits),
            )
            self.decoder = nn.Sequential(
                nn.Linear(total_bits, 32), nn.ReLU(),
                nn.Linear(32, 64), nn.ReLU(),
                nn.Linear(64, dim),
            )
            self.classifier = nn.Sequential(
                nn.Linear(total_bits, 32), nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, num_classes),
            )
            self.temperature = 1.0

        def quantize(self, logits):
            soft = torch.sigmoid(logits * self.temperature)
            hard = (soft > 0.5).float()
            return hard - soft.detach() + soft, soft

        def forward(self, x):
            logits = self.encoder(x)
            bits, soft = self.quantize(logits)
            return {
                "bits": bits, "soft": soft,
                "reconstructed": self.decoder(bits),
                "logits": self.classifier(bits),
            }

    model = BottleneckModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    recon_loss_fn = nn.MSELoss()
    class_loss_fn = nn.CrossEntropyLoss()

    # ─── Train ───────────────────────────────────────────────────
    for epoch in range(epochs):
        model.train()
        model.temperature = 1.0 + 9.0 * (epoch / max(epochs - 1, 1))

        perm = torch.randperm(len(X_train))
        batch_size = 256
        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i+batch_size]
            out = model(X_train[idx])
            loss_r = recon_loss_fn(out["reconstructed"], X_train[idx])
            loss_c = class_loss_fn(out["logits"], y_train[idx])
            soft = out["soft"]
            loss_b = -torch.mean(soft * torch.log(soft + 1e-8) +
                                 (1 - soft) * torch.log(1 - soft + 1e-8))
            loss = loss_r + 0.5 * loss_c + 0.1 * loss_b
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            model.temperature = 10.0
            with torch.no_grad():
                out_t = model(X_test)
                acc = (out_t["logits"].argmax(1) == y_test).float().mean().item()
                recon = recon_loss_fn(out_t["reconstructed"], X_test).item()
            print(f"  Epoch {epoch+1}: acc={acc:.4f}, recon={recon:.6f}")

    # ─── Evaluate ────────────────────────────────────────────────
    model.eval()
    model.temperature = 10.0
    results = {"domain": domain_name, "dimensions": dim, "total_bits": total_bits}

    # Classification accuracy
    with torch.no_grad():
        out_t = model(X_test)
        acc = (out_t["logits"].argmax(1) == y_test).float().mean().item()
    results["classification_accuracy"] = acc

    # Centroid preservation
    centroid_results = {}
    with torch.no_grad():
        for name, prior in priors.items():
            pn = normalize(prior)
            x = torch.tensor([pn], dtype=torch.float32)
            out = model(x)
            recon = out["reconstructed"][0].numpy()
            cos = cosine_sim(pn, list(recon))
            l2 = float(np.linalg.norm(np.array(pn) - recon))
            centroid_results[name] = {"cosine_sim": cos, "l2_displacement": l2}

    mean_cos = np.mean([v["cosine_sim"] for v in centroid_results.values()])
    mean_l2 = np.mean([v["l2_displacement"] for v in centroid_results.values()])
    results["centroid_preservation"] = centroid_results
    results["mean_cosine_similarity"] = float(mean_cos)
    results["mean_l2_displacement"] = float(mean_l2)

    # Pairwise similarity preservation (Spearman)
    from scipy.stats import spearmanr
    orig_sims, recon_sims = [], []
    with torch.no_grad():
        recon_vecs = {}
        for name, prior in priors.items():
            pn = normalize(prior)
            x = torch.tensor([pn], dtype=torch.float32)
            out = model(x)
            recon_vecs[name] = out["reconstructed"][0].numpy()

        for i, n1 in enumerate(names):
            for j, n2 in enumerate(names):
                if j <= i:
                    continue
                orig_sims.append(cosine_sim(normalize(priors[n1]), normalize(priors[n2])))
                recon_sims.append(float(np.dot(recon_vecs[n1], recon_vecs[n2]) /
                    (np.linalg.norm(recon_vecs[n1]) * np.linalg.norm(recon_vecs[n2]) + 1e-8)))

    rho, p_val = spearmanr(orig_sims, recon_sims)
    results["spearman_rho"] = float(rho)
    results["spearman_p"] = float(p_val)

    # Braille lattice comparison (hand-crafted vs learned)
    braille_results = {}
    for name, prior in priors.items():
        pn = normalize(prior)
        cells = encode_vector(prior, axes, polarity_pairs)
        decoded = decode_vector(cells)
        braille_cos = cosine_sim(pn, decoded)
        braille_results[name] = {
            "braille_signature": braille_signature(cells),
            "braille_cosine_sim": braille_cos,
        }
    results["braille_lattice"] = braille_results
    results["mean_braille_cosine"] = float(np.mean([v["braille_cosine_sim"] for v in braille_results.values()]))

    # Print summary
    print(f"\n  RESULTS for {domain_name}:")
    print(f"    Classification accuracy: {acc:.4f}")
    print(f"    Mean centroid cosine sim: {mean_cos:.4f}")
    print(f"    Spearman ρ (similarity preservation): {rho:.4f} (p={p_val:.2e})")
    print(f"    Mean braille lattice cosine sim: {results['mean_braille_cosine']:.4f}")
    print(f"\n    Per-class results:")
    for name in names:
        cr = centroid_results[name]
        br = braille_results[name]
        print(f"      {name:25s}: cos={cr['cosine_sim']:.4f}, "
              f"braille={br['braille_signature']}, braille_cos={br['braille_cosine_sim']:.4f}")

    return results


# ─── Comparison Table Generator ──────────────────────────────────────

def generate_comparison_report(theology_results: dict, political_results: dict,
                                personality_results: dict, output_path: str):
    """Generate markdown report comparing all three domains."""
    lines = [
        "# Braille Lattice Generalizability — Non-Theological Domains",
        "",
        "## Claim",
        "",
        "> The braille lattice is not theology-specific. The same discrete bottleneck",
        "> architecture preserves semantic structure across fundamentally different domains,",
        "> suggesting it functions as a **general-purpose semantic compression operator**.",
        "",
        "---",
        "",
        "## Cross-Domain Comparison",
        "",
        "| Metric | Theology (12D) | Political Ideology (10D) | Personality (5D) |",
        "|--------|---------------|------------------------|-----------------|",
    ]

    domains = [
        ("Theology (12D)", theology_results),
        ("Political Ideology (10D)", political_results),
        ("Personality (5D)", personality_results),
    ]

    metrics = [
        ("Classification accuracy", "classification_accuracy", ".4f"),
        ("Mean centroid cosine sim", "mean_cosine_similarity", ".4f"),
        ("Spearman ρ (similarity)", "spearman_rho", ".4f"),
        ("Mean braille cosine sim", "mean_braille_cosine", ".4f"),
    ]

    for metric_name, key, fmt in metrics:
        row = f"| {metric_name} |"
        for _, res in domains:
            val = res.get(key, "N/A")
            if isinstance(val, float):
                row += f" {val:{fmt}} |"
            else:
                row += f" {val} |"
        lines.append(row)

    lines.extend([
        "",
        "---",
        "",
        "## Political Ideology Results",
        "",
        "| Ideology | Braille Sig | Centroid Cos | Braille Cos |",
        "|----------|-------------|-------------|-------------|",
    ])

    for name in political_results.get("centroid_preservation", {}):
        cr = political_results["centroid_preservation"][name]
        br = political_results["braille_lattice"].get(name, {})
        lines.append(f"| {name} | `{br.get('braille_signature', '')}` | "
                     f"{cr['cosine_sim']:.4f} | {br.get('braille_cosine_sim', 0):.4f} |")

    lines.extend([
        "",
        "## Personality Type Results",
        "",
        "| Type | Braille Sig | Centroid Cos | Braille Cos |",
        "|------|-------------|-------------|-------------|",
    ])

    for name in personality_results.get("centroid_preservation", {}):
        cr = personality_results["centroid_preservation"][name]
        br = personality_results["braille_lattice"].get(name, {})
        lines.append(f"| {name} | `{br.get('braille_signature', '')}` | "
                     f"{cr['cosine_sim']:.4f} | {br.get('braille_cosine_sim', 0):.4f} |")

    lines.extend([
        "",
        "---",
        "",
        "## Interpretation",
        "",
        "If all three domains show high centroid preservation (>0.99 cosine),",
        "high classification accuracy (>80%), and high similarity rank correlation",
        "(ρ > 0.95), then the braille lattice is a **domain-agnostic** semantic",
        "compression operator. The theological application is one instance of a",
        "general principle: structured meaning survives discrete compression",
        "regardless of the semantic domain.",
        "",
        "This is the foundation of the AGI bridge claim: if meaning is compressible",
        "to discrete lattice points across arbitrary domains, then a shared lattice",
        "could serve as a universal semantic substrate for heterogeneous AI systems",
        "that need to maintain coherent meaning across modalities and scales.",
        "",
        "---",
        "*Generated by the Non-Theological Braille Lattice Demo — Gods as Centroids*",
    ])

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {output_path}")
    return report


# ─── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    # Run political ideology experiment
    political_results = run_bottleneck_experiment(
        domain_name="Political Ideology",
        axes=POLITICAL_AXES,
        priors=IDEOLOGY_PRIORS,
        polarity_pairs=POLARITY_PAIRS_POLITICAL,
        bits_per_axis=6,
    )

    # Run personality experiment
    personality_results = run_bottleneck_experiment(
        domain_name="Personality Types (Big Five)",
        axes=PERSONALITY_AXES,
        priors=PERSONALITY_PRIORS,
        polarity_pairs=POLARITY_PAIRS_PERSONALITY,
        bits_per_axis=6,
    )

    # Theology baseline (from paper's existing results)
    theology_results = {
        "domain": "Theology",
        "dimensions": 12,
        "total_bits": 72,
        "classification_accuracy": 0.8406,
        "mean_cosine_similarity": 0.9954,
        "spearman_rho": 0.9713,
        "mean_braille_cosine": 0.9858,
    }

    # Generate comparison report
    output_path = os.path.join(os.path.dirname(__file__), "non_theological_results.md")
    generate_comparison_report(theology_results, political_results,
                                personality_results, output_path)

    # Save raw data
    raw_path = os.path.join(os.path.dirname(__file__), "non_theological_results.json")
    all_results = {
        "theology": theology_results,
        "political": political_results,
        "personality": personality_results,
    }
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Raw data saved to {raw_path}")
