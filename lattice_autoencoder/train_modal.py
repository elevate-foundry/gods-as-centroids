#!/usr/bin/env python3
"""
GPU-Accelerated Phase 2 + Phase 3 Training on Modal
=====================================================

Parallelization strategy:
  - Each Modal function trains one hyperparameter config end-to-end (Phase 2 → Phase 3)
  - Multiple configs run in parallel on separate GPUs
  - Both modality encoders train JOINTLY (shared lattice = consensus mechanism)
  - Results collected and best config selected

Usage:
  modal run lattice_autoencoder/train_modal.py
"""

import modal
import json
import math
import os

app = modal.App("lattice-autoencoder")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy")
)

# ═══════════════════════════════════════════════════════════════════════
# INLINED DATA — Modal containers are self-contained
# ═══════════════════════════════════════════════════════════════════════

AXES = [
    "authority", "transcendence", "care", "justice", "wisdom", "power",
    "fertility", "war", "death", "creation", "nature", "order"
]

POLARITY_PAIRS = {
    "authority": "care", "care": "authority",
    "transcendence": "nature", "nature": "transcendence",
    "justice": "fertility", "fertility": "justice",
    "wisdom": "war", "war": "wisdom",
    "power": "death", "death": "power",
    "creation": "order", "order": "creation",
}

TRADITION_FAMILIES = {
    "abrahamic": ["Judaism", "Christianity", "Islam", "Bahai", "Druze", "Samaritanism"],
    "dharmic": ["Hinduism", "Buddhism", "Jainism", "Sikhism"],
    "east_asian": ["Daoism", "Confucianism", "Shinto"],
    "african": ["Yoruba", "Akan", "Kemetic"],
    "african_diaspora": ["Candomble", "Vodou", "Rastafari"],
    "indigenous": ["Lakota", "Navajo", "Aboriginal Australian", "Maori", "Hawaiian"],
    "mesoamerican": ["Nahua", "Maya", "Inca"],
}

SCHISM_PAIRS = [
    ("Judaism", "Christianity"), ("Christianity", "Islam"),
    ("Hinduism", "Buddhism"), ("Hinduism", "Jainism"),
    ("Buddhism", "Jainism"), ("Vodou", "Candomble"),
]


# ═══════════════════════════════════════════════════════════════════════
# INLINED MODEL — All architecture code self-contained for Modal
# ═══════════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    gpu="T4",
    timeout=3600,
    volumes={"/data": modal.Volume.from_name("lattice-data", create_if_missing=True)},
)
def train_config(config_idx: int, config: dict, corpus_data: dict):
    """Train one hyperparameter config: Phase 2 → Phase 3 end-to-end."""
    import random
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Config {config_idx}: {config}")
    print(f"Device: {device}")

    # ─── Unpack data ──────────────────────────────────────────────
    text_embeds = torch.tensor(corpus_data["text_embeds"], dtype=torch.float32)
    consensus_vecs = torch.tensor(corpus_data["consensus_vecs"], dtype=torch.float32)
    labels = torch.tensor(corpus_data["labels"], dtype=torch.long)
    scorer_vecs_all = [torch.tensor(sv, dtype=torch.float32) for sv in corpus_data["scorer_vecs"]]
    traditions = corpus_data["traditions"]
    text_dim = text_embeds.shape[1]
    n_passages = text_embeds.shape[0]
    n_axes = 12
    n_bits = n_axes * 8
    n_classes = corpus_data["n_classes"]
    n_scorers = len(scorer_vecs_all)

    # ─── Straight-Through Estimator ───────────────────────────────
    class STE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, tau):
            probs = torch.sigmoid(logits * tau)
            hard = (probs > 0.5).float()
            ctx.save_for_backward(logits, torch.tensor(tau))
            return hard
        @staticmethod
        def backward(ctx, grad_output):
            logits, tau = ctx.saved_tensors
            probs = torch.sigmoid(logits * tau.item())
            return grad_output * probs * (1 - probs) * tau.item(), None

    def ste_binarize(logits, tau=1.0):
        return STE.apply(logits, tau)

    # ─── Model Components ─────────────────────────────────────────
    hidden_dim = config["hidden_dim"]

    class TextEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, n_axes), nn.Sigmoid(),
            )
        def forward(self, x): return self.net(x)

    class ScorerEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_axes, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.GELU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.GELU(),
                nn.Linear(hidden_dim // 2, n_axes), nn.Sigmoid(),
            )
        def forward(self, x): return self.net(x)

    class LatticeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.tau = 1.0
            self.encoder = nn.Sequential(
                nn.Linear(n_axes, n_axes * 4), nn.GELU(),
                nn.Linear(n_axes * 4, n_bits),
            )
            for m in self.encoder:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
        def forward(self, x):
            logits = self.encoder(x)
            return ste_binarize(logits, self.tau)
        def anneal(self, epoch, max_epochs, tau_min=1.0, tau_max=10.0):
            self.tau = tau_min + (tau_max - tau_min) * min(1.0, epoch / max(1, max_epochs))

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_bits, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, n_axes),
            )
        def forward(self, z): return self.net(z)

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_bits, hidden_dim // 2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, n_classes),
            )
        def forward(self, z): return self.net(z)

    # ─── Phase 2 Model ────────────────────────────────────────────
    text_enc = TextEncoder().to(device)
    scorer_enc = ScorerEncoder().to(device)
    lattice = LatticeLayer().to(device)
    decoder = Decoder().to(device)
    classifier = Classifier().to(device)

    all_params = (list(text_enc.parameters()) + list(scorer_enc.parameters()) +
                  list(lattice.parameters()) + list(decoder.parameters()) +
                  list(classifier.parameters()))
    n_params = sum(p.numel() for p in all_params)
    print(f"  Phase 2 model: {n_params:,} params")

    # ─── Build augmented dataset ──────────────────────────────────
    noise_std = config.get("noise_std", 0.02)
    spp = config.get("samples_per_passage", 50)

    def build_batches(n_repeats, shuffle=True):
        all_text, all_scorer, all_cons, all_lab = [], [], [], []
        rng = random.Random(42)
        for rep in range(n_repeats):
            for i in range(n_passages):
                te = text_embeds[i].clone()
                sv = scorer_vecs_all[rng.randint(0, n_scorers - 1)][i].clone()
                cv = consensus_vecs[i].clone()
                lb = labels[i]
                if rep > 0:
                    te = te + torch.randn_like(te) * noise_std
                    te = te / (te.norm() + 1e-8)
                    sv = sv + torch.randn_like(sv) * noise_std
                    sv = sv / (sv.norm() + 1e-8)
                all_text.append(te)
                all_scorer.append(sv)
                all_cons.append(cv)
                all_lab.append(lb)
        ds = TensorDataset(
            torch.stack(all_text), torch.stack(all_scorer),
            torch.stack(all_cons), torch.stack(all_lab),
        )
        return DataLoader(ds, batch_size=config.get("batch_size", 64), shuffle=shuffle)

    train_loader = build_batches(spp, shuffle=True)
    val_loader = build_batches(1, shuffle=False)

    # ─── Phase 2 Training ─────────────────────────────────────────
    lr = config.get("lr", 1e-3)
    align_weight = config.get("align_weight", 2.0)
    p2_epochs = config.get("p2_epochs", 300)

    optimizer = AdamW(all_params, lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=p2_epochs, eta_min=lr * 0.01)

    print(f"\n  Phase 2: {p2_epochs} epochs, lr={lr}, align_w={align_weight}")
    history = []

    for epoch in range(1, p2_epochs + 1):
        text_enc.train(); scorer_enc.train(); lattice.train()
        decoder.train(); classifier.train()
        lattice.anneal(epoch, p2_epochs, 1.0, config.get("tau_max", 10.0))

        epoch_loss = 0.0
        epoch_n = 0
        for te, sv, cv, lb in train_loader:
            te, sv, cv, lb = te.to(device), sv.to(device), cv.to(device), lb.to(device)

            h_text = text_enc(te)
            h_scorer = scorer_enc(sv)
            z_text = lattice(h_text)
            z_scorer = lattice(h_scorer)
            recon_text = decoder(z_text)
            recon_scorer = decoder(z_scorer)
            logits_text = classifier(z_text)
            logits_scorer = classifier(z_scorer)

            # Consensus
            z_cons = ((z_text + z_scorer) / 2.0 > 0.5).float()
            recon_cons = decoder(z_cons)

            # Losses
            cos_t = (1 - F.cosine_similarity(cv, recon_text, dim=-1)).mean()
            cos_s = (1 - F.cosine_similarity(cv, recon_scorer, dim=-1)).mean()
            l2_t = F.mse_loss(recon_text, cv)
            l2_s = F.mse_loss(recon_scorer, cv)
            cls_t = F.cross_entropy(logits_text, lb)
            cls_s = F.cross_entropy(logits_scorer, lb)
            align = F.mse_loss(z_text, z_scorer)
            cos_c = (1 - F.cosine_similarity(cv, recon_cons, dim=-1)).mean()

            loss = (cos_t + cos_s + 0.5 * (l2_t + l2_s) +
                    0.3 * (cls_t + cls_s) + align_weight * align + 0.5 * cos_c)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            epoch_loss += loss.item() * te.shape[0]
            epoch_n += te.shape[0]

        scheduler.step()

        if epoch % 30 == 0 or epoch == 1 or epoch == p2_epochs:
            # Eval
            text_enc.eval(); scorer_enc.eval(); lattice.eval()
            decoder.eval(); classifier.eval()
            with torch.no_grad():
                val_cos_t, val_cos_s, val_agree = 0.0, 0.0, 0.0
                val_acc_t, val_acc_s, val_n = 0, 0, 0
                for te, sv, cv, lb in val_loader:
                    te, sv, cv, lb = te.to(device), sv.to(device), cv.to(device), lb.to(device)
                    zt = lattice(text_enc(te))
                    zs = lattice(scorer_enc(sv))
                    rt = decoder(zt)
                    rs = decoder(zs)
                    val_cos_t += F.cosine_similarity(cv, rt, dim=-1).sum().item()
                    val_cos_s += F.cosine_similarity(cv, rs, dim=-1).sum().item()
                    val_agree += (zt == zs).float().mean(dim=-1).sum().item()
                    val_acc_t += (classifier(zt).argmax(-1) == lb).sum().item()
                    val_acc_s += (classifier(zs).argmax(-1) == lb).sum().item()
                    val_n += lb.shape[0]

            metrics = {
                "cos_text": val_cos_t / val_n,
                "cos_scorer": val_cos_s / val_n,
                "bit_agree": val_agree / val_n,
                "acc_text": val_acc_t / val_n,
                "acc_scorer": val_acc_s / val_n,
            }
            history.append({"epoch": epoch, "loss": epoch_loss / epoch_n, **metrics})
            print(f"    Epoch {epoch:>4d}  loss={epoch_loss/epoch_n:.4f}  "
                  f"agree={metrics['bit_agree']:.1%}  "
                  f"cos_t={metrics['cos_text']:.4f}  cos_s={metrics['cos_scorer']:.4f}  "
                  f"acc_t={metrics['acc_text']:.1%}  acc_s={metrics['acc_scorer']:.1%}")

    p2_final = history[-1] if history else {}

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: Operator Training
    # ═══════════════════════════════════════════════════════════════

    print(f"\n  Phase 3: Generating operator examples...")
    text_enc.eval(); scorer_enc.eval(); lattice.eval()

    # Encode all passages
    with torch.no_grad():
        all_z = lattice(scorer_enc(consensus_vecs.to(device)))

    trad_to_idx = {}
    for i, t in enumerate(traditions):
        trad_to_idx.setdefault(t, []).append(i)

    trad_to_family = {}
    for fam, trads in TRADITION_FAMILIES.items():
        for t in trads:
            trad_to_family[t] = fam

    rng = random.Random(42)
    za_list, zb_list, op_list, tv_list, tl_list = [], [], [], [], []

    # Fusion (op=0): same-family pairs
    for fam, trads in TRADITION_FAMILIES.items():
        avail = [t for t in trads if t in trad_to_idx]
        for i in range(len(avail)):
            for j in range(i+1, len(avail)):
                ia = rng.choice(trad_to_idx[avail[i]])
                ib = rng.choice(trad_to_idx[avail[j]])
                za_list.append(all_z[ia]); zb_list.append(all_z[ib])
                op_list.append(0)
                tv_list.append((consensus_vecs[ia] + consensus_vecs[ib]) / 2)
                tl_list.append(labels[ia])

    # Fission (op=1): known schisms
    for ta, tb in SCHISM_PAIRS:
        if ta in trad_to_idx and tb in trad_to_idx:
            for _ in range(3):
                ia = rng.choice(trad_to_idx[ta]); ib = rng.choice(trad_to_idx[tb])
                za_list.append(all_z[ia]); zb_list.append(all_z[ib])
                op_list.append(1)
                tv_list.append(consensus_vecs[ia]); tl_list.append(labels[ia])

    # Perturbation (op=2): cross-family
    for t in trad_to_idx:
        if t not in trad_to_family: continue
        fam = trad_to_family[t]
        distant = [t2 for t2 in trad_to_idx if t2 in trad_to_family and trad_to_family[t2] != fam]
        if distant:
            for _ in range(2):
                td = rng.choice(distant)
                ia = rng.choice(trad_to_idx[t]); ib = rng.choice(trad_to_idx[td])
                za_list.append(all_z[ia]); zb_list.append(all_z[ib])
                op_list.append(2)
                tv_list.append(consensus_vecs[ia]); tl_list.append(labels[ia])

    # None (op=3): same tradition
    for t, idxs in trad_to_idx.items():
        if len(idxs) >= 2:
            for _ in range(2):
                ia, ib = rng.sample(idxs, 2)
                za_list.append(all_z[ia]); zb_list.append(all_z[ib])
                op_list.append(3)
                tv_list.append(consensus_vecs[ia]); tl_list.append(labels[ia])

    za_t = torch.stack(za_list); zb_t = torch.stack(zb_list)
    op_t = torch.tensor(op_list, dtype=torch.long)
    tv_t = torch.stack(tv_list); tl_t = torch.tensor(tl_list, dtype=torch.long)

    for op, name in enumerate(["fusion", "fission", "perturbation", "none"]):
        print(f"    {name}: {(op_t == op).sum().item()}")
    print(f"    Total: {len(op_t)}")

    # Train/val split
    perm = torch.randperm(len(op_t))
    n_val = max(1, int(len(op_t) * 0.15))
    train_idx = perm[n_val:]
    val_idx = perm[:n_val]

    # Repeat training data
    n_rep = 30
    za_tr = za_t[train_idx].repeat(n_rep, 1)
    zb_tr = zb_t[train_idx].repeat(n_rep, 1)
    op_tr = op_t[train_idx].repeat(n_rep)
    tv_tr = tv_t[train_idx].repeat(n_rep, 1)
    tl_tr = tl_t[train_idx].repeat(n_rep)

    p3_train = DataLoader(TensorDataset(za_tr, zb_tr, op_tr, tv_tr, tl_tr),
                          batch_size=64, shuffle=True)
    p3_val = DataLoader(TensorDataset(za_t[val_idx], zb_t[val_idx], op_t[val_idx],
                                       tv_t[val_idx], tl_t[val_idx]),
                        batch_size=64, shuffle=False)

    # Operator model
    class FusionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(n_bits*2, 64), nn.GELU(), nn.Linear(64, n_bits), nn.Sigmoid())
        def forward(self, a, b):
            w = self.net(torch.cat([a, b], -1))
            soft = w * a + (1-w) * b
            hard = (soft > 0.5).float()
            return hard + soft - soft.detach()

    class FissionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(n_bits, 64), nn.GELU(), nn.Linear(64, n_bits), nn.Tanh())
        def forward(self, z):
            p = self.net(z)
            sa, sb = z + 0.3*p, z - 0.3*p
            ha, hb = (sa > 0.5).float(), (sb > 0.5).float()
            return ha + sa - sa.detach(), hb + sb - sb.detach()

    class PerturbNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(n_bits*2, 64), nn.GELU(), nn.Linear(64, n_bits), nn.Sigmoid())
        def forward(self, c, p):
            pull = self.net(torch.cat([c, p], -1))
            soft = (1-pull)*c + pull*p
            hard = (soft > 0.5).float()
            return hard + soft - soft.detach()

    class OpPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_bits*2+4, 64), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 4),
            )
        def forward(self, a, b):
            h = (a != b).float().sum(-1, keepdim=True)
            hn = h / n_bits
            ba = (a*b).sum(-1, keepdim=True) / n_bits
            bx = ((a+b)%2).sum(-1, keepdim=True) / n_bits
            return self.net(torch.cat([a, b, hn, ba, bx, h/n_bits], -1))

    fusion_net = FusionNet().to(device)
    fission_net = FissionNet().to(device)
    perturb_net = PerturbNet().to(device)
    op_pred = OpPredictor().to(device)
    op_decoder = Decoder().to(device)
    op_classifier = Classifier().to(device)

    p3_params = (list(fusion_net.parameters()) + list(fission_net.parameters()) +
                 list(perturb_net.parameters()) + list(op_pred.parameters()) +
                 list(op_decoder.parameters()) + list(op_classifier.parameters()))
    print(f"  Phase 3 model: {sum(p.numel() for p in p3_params):,} params")

    p3_epochs = config.get("p3_epochs", 200)
    p3_opt = AdamW(p3_params, lr=lr, weight_decay=1e-4)
    p3_sched = CosineAnnealingLR(p3_opt, T_max=p3_epochs, eta_min=lr*0.01)

    print(f"  Phase 3: {p3_epochs} epochs")
    p3_history = []

    for epoch in range(1, p3_epochs + 1):
        fusion_net.train(); fission_net.train(); perturb_net.train()
        op_pred.train(); op_decoder.train(); op_classifier.train()

        ep_loss, ep_correct, ep_n = 0.0, 0, 0
        for za, zb, ops, tvecs, tlabs in p3_train:
            za, zb, ops = za.to(device), zb.to(device), ops.to(device)
            tvecs, tlabs = tvecs.to(device), tlabs.to(device)

            logits = op_pred(za, zb)
            z_fused = fusion_net(za, zb)
            z_fa, z_fb = fission_net(za)
            z_pert = perturb_net(za, zb)

            loss = F.cross_entropy(logits, ops)

            # Fusion recon
            fm = (ops == 0).float()
            if fm.sum() > 0:
                cl = (1 - F.cosine_similarity(tvecs, op_decoder(z_fused), dim=-1))
                loss = loss + (cl * fm).sum() / (fm.sum() + 1e-8)

            # Fission distance
            fim = (ops == 1).float()
            if fim.sum() > 0:
                cd = (z_fa - z_fb).abs().mean(-1)
                fl = F.relu(0.2 - cd)
                loss = loss + (fl * fim).sum() / (fim.sum() + 1e-8)

            # Perturbation pull
            pm = (ops == 2).float()
            if pm.sum() > 0:
                od = (za - zb).abs().mean(-1)
                nd = (z_pert - zb).abs().mean(-1)
                pl = F.relu(nd - 0.7 * od)
                loss = loss + (pl * pm).sum() / (pm.sum() + 1e-8)

            # Classification
            loss = loss + 0.3 * F.cross_entropy(op_classifier(z_fused), tlabs)

            p3_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(p3_params, 1.0)
            p3_opt.step()

            ep_loss += loss.item() * za.shape[0]
            ep_correct += (logits.argmax(-1) == ops).sum().item()
            ep_n += za.shape[0]

        p3_sched.step()

        if epoch % 20 == 0 or epoch == 1 or epoch == p3_epochs:
            # Eval
            fusion_net.eval(); fission_net.eval(); perturb_net.eval()
            op_pred.eval(); op_decoder.eval()
            vc, vn = 0, 0
            with torch.no_grad():
                for za, zb, ops, tvecs, tlabs in p3_val:
                    za, zb, ops = za.to(device), zb.to(device), ops.to(device)
                    preds = op_pred(za, zb).argmax(-1)
                    vc += (preds == ops).sum().item()
                    vn += ops.shape[0]
            val_acc = vc / max(1, vn)
            p3_history.append({"epoch": epoch, "loss": ep_loss/ep_n,
                               "train_acc": ep_correct/ep_n, "val_acc": val_acc})
            print(f"    Epoch {epoch:>4d}  loss={ep_loss/ep_n:.4f}  "
                  f"train={ep_correct/ep_n:.1%}  val={val_acc:.1%}")

    p3_final = p3_history[-1] if p3_history else {}

    # ═══════════════════════════════════════════════════════════════
    # Collect results
    # ═══════════════════════════════════════════════════════════════
    result = {
        "config_idx": config_idx,
        "config": config,
        "phase2": {
            "final_metrics": p2_final,
            "history": history,
        },
        "phase3": {
            "final_metrics": p3_final,
            "history": p3_history,
        },
    }

    # Save to volume
    out_path = f"/data/config_{config_idx}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


@app.local_entrypoint()
def main():
    import time

    # ─── Load and serialize all data for Modal ────────────────────
    print("Loading corpus data...")

    # Use scaled corpus if available, else original
    scaled_consensus = "lattice_autoencoder/data/merged_consensus.json"
    original_consensus = "mlx-pipeline/multi_scorer_consensus.json"

    import os
    use_scaled = os.path.exists(scaled_consensus)
    cons_path = scaled_consensus if use_scaled else original_consensus
    print(f"  Using {'scaled' if use_scaled else 'original'} corpus: {cons_path}")

    with open("lattice_autoencoder/data/text_embeddings.json") as f:
        text_data = json.load(f)

    with open(cons_path) as f:
        cons_data = json.load(f)

    # Build text embed lookup by (tradition, source)
    text_lookup = {}
    for e in text_data["embeddings"]:
        text_lookup[(e["tradition"], e["source"])] = e["vector"]

    # Match consensus entries to text embeddings
    traditions_list = []
    text_embeds = []
    consensus_vecs = []
    raw_score_vecs = []  # unnormalized scores as scorer input
    scorer_variances = []

    for e in cons_data["embeddings"]:
        key = (e["tradition"], e["source"])
        if key not in text_lookup:
            continue  # skip entries without text embeddings
        traditions_list.append(e["tradition"])
        text_embeds.append(text_lookup[key])
        consensus_vecs.append([e["normalized"].get(a, 0.0) for a in AXES])
        raw_score_vecs.append([e["raw_scores"].get(a, 0.0) for a in AXES])
        var = e.get("scorer_variance", {})
        scorer_variances.append([var.get(a, 0.01) if isinstance(var, dict) else 0.01 for a in AXES])

    traditions_sorted = sorted(set(traditions_list))
    trad_to_idx = {t: i for i, t in enumerate(traditions_sorted)}
    labels = [trad_to_idx[t] for t in traditions_list]

    # Generate 4 synthetic scorer vectors from raw_scores + variance
    # This simulates per-model variation for the scorer encoder
    import random
    random.seed(42)
    scorer_vecs = []
    for model_idx in range(4):
        model_vecs = []
        for i in range(len(raw_score_vecs)):
            vec = []
            for j in range(12):
                # Add model-specific offset based on variance
                std = math.sqrt(scorer_variances[i][j]) if scorer_variances[i][j] > 0 else 0.02
                offset = random.gauss(0, std)
                val = max(0.0, min(1.0, raw_score_vecs[i][j] + offset))
                vec.append(val)
            norm = math.sqrt(sum(v*v for v in vec)) or 1.0
            model_vecs.append([v/norm for v in vec])
        scorer_vecs.append(model_vecs)

    corpus_data = {
        "text_embeds": text_embeds,
        "consensus_vecs": consensus_vecs,
        "labels": labels,
        "traditions": traditions_list,
        "scorer_vecs": scorer_vecs,
        "n_classes": len(traditions_sorted),
    }

    print(f"  {len(text_embeds)} passages, {len(traditions_sorted)} traditions, "
          f"{len(scorer_vecs)} scorers")

    # ─── Hyperparameter configs to sweep in parallel ──────────────
    # Based on previous best: hidden=192, lr=2e-3, align=2.5, tau=10
    # With 826 passages (6.5× more), use larger batches
    configs = [
        {"hidden_dim": 192, "lr": 2e-3, "align_weight": 2.5, "tau_max": 10.0,
         "p2_epochs": 200, "p3_epochs": 150, "batch_size": 64, "samples_per_passage": 30},
        {"hidden_dim": 256, "lr": 1e-3, "align_weight": 2.0, "tau_max": 10.0,
         "p2_epochs": 200, "p3_epochs": 150, "batch_size": 64, "samples_per_passage": 30},
    ]

    print(f"\nLaunching {len(configs)} parallel configs on Modal...")
    t0 = time.time()

    # Launch all in parallel
    futures = []
    for i, cfg in enumerate(configs):
        futures.append(train_config.spawn(i, cfg, corpus_data))

    # Collect results
    all_results = []
    for f in futures:
        result = f.get()
        all_results.append(result)
        idx = result["config_idx"]
        p2 = result["phase2"]["final_metrics"]
        p3 = result["phase3"]["final_metrics"]
        print(f"\n  Config {idx} done:")
        print(f"    Phase 2: agree={p2.get('bit_agree',0):.1%}  "
              f"cos_t={p2.get('cos_text',0):.4f}  cos_s={p2.get('cos_scorer',0):.4f}  "
              f"acc_t={p2.get('acc_text',0):.1%}  acc_s={p2.get('acc_scorer',0):.1%}")
        print(f"    Phase 3: op_acc={p3.get('val_acc',0):.1%}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"ALL CONFIGS COMPLETE in {elapsed:.0f}s")
    print(f"{'='*70}")

    # Find best config
    best_idx = max(range(len(all_results)),
                   key=lambda i: (all_results[i]["phase2"]["final_metrics"].get("bit_agree", 0) +
                                  all_results[i]["phase3"]["final_metrics"].get("val_acc", 0)))
    best = all_results[best_idx]
    print(f"\nBest config: #{best_idx}")
    print(f"  Config: {best['config']}")
    print(f"  Phase 2 bit agreement: {best['phase2']['final_metrics'].get('bit_agree',0):.1%}")
    print(f"  Phase 3 operator accuracy: {best['phase3']['final_metrics'].get('val_acc',0):.1%}")

    # Save all results locally
    out_dir = "lattice_autoencoder/runs/modal"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_dir}/sweep_results.json")
