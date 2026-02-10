#!/usr/bin/env python3
"""
Prophet Escape Experiment (Modal)
=================================
Tests §7.3 "Second Axial Age" prediction:
  Can prophet events + elevated mutation break monotheistic lock-in?

Protocol:
  Phase 1: Drive system to monotheism (γ=0.9, 3000 steps)
  Phase 2: Drop coercion to 0, vary prophet_rate and mutation_rate
  Phase 3: Measure whether N_eff recovers (escape) or stays locked (no escape)

Sweep grid:
  prophet_rate ∈ {0, 0.001, 0.005, 0.01, 0.02, 0.05}
  mutation_rate ∈ {0.08, 0.12, 0.16, 0.20, 0.25}
  n_runs = 20 per condition
"""

import modal
import json
import math
import random
import os
from pathlib import Path

app = modal.App("prophet-escape-sweep")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "matplotlib", "numpy"
)

# ─── Inline minimal SwarmKernel (same as hysteresis_modal.py) ───

AXES = [
    "authority", "transcendence", "care", "justice", "wisdom", "power",
    "fertility", "war", "death", "creation", "nature", "order"
]

def _rnd_unit(rng):
    v = {a: rng.random() for a in AXES}
    n = math.sqrt(sum(x*x for x in v.values())) or 1.0
    return {k: x/n for k, x in v.items()}

def _norm(v):
    return math.sqrt(sum(v[k]**2 for k in AXES))

def _cosine(a, b):
    d = sum(a[k]*b[k] for k in AXES)
    na = _norm(a); nb = _norm(b)
    return d/(na*nb) if na > 0 and nb > 0 else 0.0

def _add_scaled(a, b, s):
    for k in AXES:
        a[k] += b[k] * s

def _scale(a, s):
    return {k: a[k]*s for k in AXES}

def _normalize(v):
    n = _norm(v) or 1.0
    return {k: v[k]/n for k in AXES}

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


class _Agent:
    __slots__ = ('id', 'belief', 'w', 'assoc', 'freq')
    def __init__(self, id, belief, w=1.0):
        self.id = id
        self.belief = belief
        self.w = w
        self.assoc = {}
        self.freq = {}


class MiniKernel:
    """Minimal kernel with prophet + fission support."""
    def __init__(self, n_agents=80, coercion=0.0, mutation_rate=0.08,
                 prophet_rate=0.0, prophet_pull_fraction=0.15,
                 prophet_pull_strength=0.4, prophet_prestige=8.0,
                 fission_variance_threshold=0.15, fission_min_cluster_size=6,
                 cluster_threshold=0.4, seed=42):
        self.rng = random.Random(seed)
        self.n_agents = n_agents
        self.coercion = coercion
        self.mutation_rate = mutation_rate
        self.prophet_rate = prophet_rate
        self.prophet_pull_fraction = prophet_pull_fraction
        self.prophet_pull_strength = prophet_pull_strength
        self.prophet_prestige = prophet_prestige
        self.fission_variance_threshold = fission_variance_threshold
        self.fission_min_cluster_size = fission_min_cluster_size
        self.cluster_threshold = cluster_threshold
        self.t = 0
        self.agents = [_Agent(i, _rnd_unit(self.rng)) for i in range(n_agents)]
        self.centroids = []
        self.clusters = []

        # Interaction params
        self.learning_rate = 0.08
        self.penalty_rate = 0.02
        self.prestige_alpha = 0.20
        self.ritual_period = 50
        self.ritual_bonus = 0.10
        self.base_success_thresh = 0.58
        self.belief_influence = 0.15

        # Init associations
        theonyms = ["isis","thor","indra","yah","nana","osiris","zeus","odin",
                     "ra","anu","ishtar","baal","amun","enar","shen","kami"]
        for a in self.agents:
            for name in theonyms:
                a.assoc[name] = _rnd_unit(self.rng)
                a.freq[name] = 0

        # Prophet/fission tracking
        self.prophet_events = []
        self.fission_events = []

    def _sample_context(self):
        return _rnd_unit(self.rng)

    def _select_hearers(self, speaker):
        others = [a for a in self.agents if a.id != speaker.id]
        if self.coercion > 0:
            weights = [math.exp(_cosine(speaker.belief, a.belief) * (1 + 9*self.coercion)) for a in others]
        else:
            weights = [1.0] * len(others)
        return self.rng.choices(others, weights=weights, k=min(3, len(others)))

    def _produce(self, speaker, ctx_vec):
        scored = []
        total_freq = sum(speaker.freq.values()) + 1
        for form, vec in speaker.assoc.items():
            ctx_score = sum(vec[k]*ctx_vec[k] for k in AXES)
            belief_score = sum(vec[k]*speaker.belief[k] for k in AXES) * self.belief_influence
            freq_score = 0.1 * math.log((speaker.freq.get(form,0)+1)/total_freq)
            scored.append((form, ctx_score + belief_score + freq_score))
        msg = []
        for _ in range(3):
            if self.rng.random() < 0.10:
                msg.append(self.rng.choice([k for k,_ in scored]))
            else:
                mx = max(s for _,s in scored) if scored else 1.0
                exps = [(k, math.exp(s-mx)) for k,s in scored]
                z = sum(v for _,v in exps) or 1.0
                r = self.rng.random()*z
                acc = 0.0
                for k,v in exps:
                    acc += v
                    if acc >= r:
                        msg.append(k)
                        break
                else:
                    msg.append(scored[-1][0])
        return msg

    def _interpret(self, hearer, msg):
        v = {k: 0.0 for k in AXES}
        for tok in msg:
            if tok in hearer.assoc:
                _add_scaled(v, hearer.assoc[tok], 1.0)
        n = _norm(v)
        if n == 0:
            return _rnd_unit(self.rng)
        return {k: v[k]/n for k in AXES}

    def step(self):
        self.t += 1
        ctx_vec = self._sample_context()
        weights = [a.w for a in self.agents]
        speaker = self.rng.choices(self.agents, weights=weights, k=1)[0]
        hearers = self._select_hearers(speaker)
        msg = self._produce(speaker, ctx_vec)

        preds = [self._interpret(h, msg) for h in hearers]
        sim = sum(_cosine(ctx_vec, p) for p in preds) / max(1, len(preds))
        ritual = (self.t % self.ritual_period == 0)
        thresh = self.base_success_thresh + (self.ritual_bonus if ritual else 0.0)
        success = sim >= thresh

        lr = self.learning_rate if success else -self.penalty_rate
        for a in [speaker] + hearers:
            blend = {k: 0.8*ctx_vec[k] + 0.2*a.belief[k] for k in AXES}
            for tok in msg:
                if tok not in a.assoc:
                    a.assoc[tok] = _rnd_unit(self.rng)
                _add_scaled(a.assoc[tok], blend, lr)
                a.freq[tok] = a.freq.get(tok, 0) + 1

        for a in [speaker] + hearers:
            delta = self.prestige_alpha if success else -self.prestige_alpha * 0.3
            a.w = _clamp(a.w * (1.0 + delta), 0.1, 10.0)

        for a in [speaker] + hearers:
            if self.rng.random() < self.mutation_rate:
                for k in AXES:
                    a.belief[k] += self.rng.gauss(0, 0.05)
                a.belief = _normalize(a.belief)

        # Prophet events
        if self.prophet_rate > 0 and self.rng.random() < self.prophet_rate:
            prophet = self.rng.choice(self.agents)
            prophet.belief = _rnd_unit(self.rng)
            prophet.w = self.prophet_prestige
            n_pull = max(1, int(self.prophet_pull_fraction * len(self.agents)))
            others = [a for a in self.agents if a.id != prophet.id]
            others.sort(key=lambda a: _cosine(a.belief, prophet.belief), reverse=True)
            for a in others[:n_pull]:
                lam = self.prophet_pull_strength
                new_b = {k: (1-lam)*a.belief[k] + lam*prophet.belief[k] for k in AXES}
                a.belief = _normalize(new_b)
            self.prophet_events.append(self.t)

        # Clustering + attractor deepening + fission
        if self.t % 50 == 0:
            self.update_clusters()
            if self.coercion > 0 and self.centroids:
                eta_base = 0.05 * self.coercion
                for ci, cluster in enumerate(self.clusters):
                    if ci >= len(self.centroids):
                        break
                    centroid = self.centroids[ci]
                    for aid in cluster:
                        a = self.agents[aid]
                        eta = eta_base * (1.0 / (a.w + 0.1))
                        eta = min(eta, 0.3)
                        for k in AXES:
                            a.belief[k] += eta * (centroid[k] - a.belief[k])
                        a.belief = _normalize(a.belief)
            # Fission check
            self._maybe_fission()

    def update_clusters(self):
        effective_threshold = self.cluster_threshold + 0.3 * self.coercion
        self.centroids = []
        self.clusters = []
        for agent in self.agents:
            if not self.centroids:
                self.centroids.append(dict(agent.belief))
                self.clusters.append([agent.id])
                continue
            distances = [1 - _cosine(agent.belief, c) for c in self.centroids]
            min_dist = min(distances)
            best_idx = distances.index(min_dist)
            if min_dist < effective_threshold:
                self.clusters[best_idx].append(agent.id)
            else:
                self.centroids.append(dict(agent.belief))
                self.clusters.append([agent.id])

        new_centroids = []
        new_clusters = []
        for cluster in self.clusters:
            if not cluster:
                continue
            centroid = {k: 0.0 for k in AXES}
            total_w = 0.0
            for aid in cluster:
                w = self.agents[aid].w
                _add_scaled(centroid, self.agents[aid].belief, w)
                total_w += w
            if total_w > 0:
                new_centroids.append(_scale(centroid, 1.0/total_w))
            else:
                new_centroids.append(_scale(centroid, 1.0/len(cluster)))
            new_clusters.append(cluster)

        # Fusion
        merge_dist = 0.15 + 0.35 * self.coercion
        merged = True
        while merged:
            merged = False
            for i in range(len(new_centroids)):
                for j in range(i+1, len(new_centroids)):
                    if 1 - _cosine(new_centroids[i], new_centroids[j]) < merge_dist:
                        combined = new_clusters[i] + new_clusters[j]
                        centroid = {k: 0.0 for k in AXES}
                        total_w = 0.0
                        for aid in combined:
                            w = self.agents[aid].w
                            _add_scaled(centroid, self.agents[aid].belief, w)
                            total_w += w
                        if total_w > 0:
                            new_centroids[i] = _scale(centroid, 1.0/total_w)
                        new_clusters[i] = combined
                        del new_centroids[j]
                        del new_clusters[j]
                        merged = True
                        break
                if merged:
                    break
        self.centroids = new_centroids
        self.clusters = new_clusters

    def _maybe_fission(self):
        if not self.clusters or not self.centroids:
            return
        i = 0
        while i < len(self.clusters):
            cluster = self.clusters[i]
            if len(cluster) < self.fission_min_cluster_size:
                i += 1
                continue
            centroid = self.centroids[i]
            total_w = 0.0
            weighted_var = 0.0
            max_w = 0.0
            for aid in cluster:
                a = self.agents[aid]
                dist_sq = sum((a.belief[k] - centroid[k])**2 for k in AXES)
                weighted_var += a.w * dist_sq
                total_w += a.w
                max_w = max(max_w, a.w)
            if total_w == 0:
                i += 1
                continue
            sigma_sq = weighted_var / total_w
            mean_w = total_w / len(cluster)
            kappa = max_w / mean_w if mean_w > 0 else 1.0
            threshold = self.fission_variance_threshold * (1 + 1.0/max(kappa, 0.01))
            if sigma_sq > threshold:
                # Find two most distant agents
                max_dist = -1
                a1, a2 = cluster[0], cluster[-1]
                for ci_idx in range(len(cluster)):
                    for cj_idx in range(ci_idx+1, len(cluster)):
                        d = 1 - _cosine(self.agents[cluster[ci_idx]].belief,
                                        self.agents[cluster[cj_idx]].belief)
                        if d > max_dist:
                            max_dist = d
                            a1, a2 = cluster[ci_idx], cluster[cj_idx]
                seed1 = dict(self.agents[a1].belief)
                seed2 = dict(self.agents[a2].belief)
                g1, g2 = [], []
                for aid in cluster:
                    d1 = 1 - _cosine(self.agents[aid].belief, seed1)
                    d2 = 1 - _cosine(self.agents[aid].belief, seed2)
                    (g1 if d1 <= d2 else g2).append(aid)
                if g1 and g2:
                    c1 = {k: 0.0 for k in AXES}; w1 = 0.0
                    for aid in g1:
                        _add_scaled(c1, self.agents[aid].belief, self.agents[aid].w)
                        w1 += self.agents[aid].w
                    if w1 > 0: c1 = _scale(c1, 1.0/w1)
                    c2 = {k: 0.0 for k in AXES}; w2 = 0.0
                    for aid in g2:
                        _add_scaled(c2, self.agents[aid].belief, self.agents[aid].w)
                        w2 += self.agents[aid].w
                    if w2 > 0: c2 = _scale(c2, 1.0/w2)
                    self.clusters[i] = g1
                    self.centroids[i] = c1
                    self.clusters.append(g2)
                    self.centroids.append(c2)
                    self.fission_events.append(self.t)
            i += 1

    def measure(self):
        self.update_clusters()
        N = len(self.agents)
        sizes = [len(c) for c in self.clusters]
        n_eff = sum(1 for s in sizes if s >= 2)
        dominance = max(sizes)/N if N > 0 and sizes else 0.0
        entropy = 0.0
        for s in sizes:
            if s > 0:
                p = s/N
                entropy -= p * math.log(p)
        return {"dominance": dominance, "n_eff": n_eff, "entropy": entropy,
                "n_clusters": len(self.clusters)}


# ─── Modal functions ───

@app.function(image=image, timeout=600)
def run_single_condition(prophet_rate: float, mutation_rate: float,
                         run_id: int, n_agents: int = 80,
                         lock_in_steps: int = 5000,
                         escape_steps: int = 8000,
                         measurement_interval: int = 200):
    """
    Phase 1: Lock-in (γ=0.9, no prophets, corpus-calibrated θ=0.12)
    Phase 2: Escape attempt (γ=0.05 residual, with prophets + mutation)
    """
    seed = run_id * 1000 + int(prophet_rate * 10000) + int(mutation_rate * 100)

    # Phase 1: Drive to monotheism with corpus-calibrated params
    k = MiniKernel(n_agents=n_agents, coercion=0.9, mutation_rate=0.05,
                   prophet_rate=0.0, cluster_threshold=0.12,
                   prophet_prestige=4.0, seed=seed)
    k.belief_influence = 0.07  # corpus-calibrated
    for _ in range(lock_in_steps):
        k.step()

    lock_in_state = k.measure()

    # Phase 2: Drop coercion (residual 0.05 = institutional inertia §7.2)
    k.coercion = 0.05
    k.mutation_rate = mutation_rate
    k.prophet_rate = prophet_rate
    k.prophet_events = []
    k.fission_events = []

    trajectory = []
    for step in range(escape_steps):
        k.step()
        if (step + 1) % measurement_interval == 0:
            m = k.measure()
            m["step"] = step + 1
            trajectory.append(m)

    final = k.measure()
    return {
        "prophet_rate": prophet_rate,
        "mutation_rate": mutation_rate,
        "run_id": run_id,
        "lock_in": lock_in_state,
        "final": final,
        "trajectory": trajectory,
        "n_prophet_events": len(k.prophet_events),
        "n_fission_events": len(k.fission_events),
        "escaped": final["n_eff"] >= 3 and final["dominance"] < 0.5,
    }


@app.function(image=image, timeout=300)
def generate_plots(all_results: list, output_dir: str):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Parse results into grid
    prophet_rates = sorted(set(r["prophet_rate"] for r in all_results))
    mutation_rates = sorted(set(r["mutation_rate"] for r in all_results))

    # Escape probability heatmap
    escape_grid = np.zeros((len(mutation_rates), len(prophet_rates)))
    neff_grid = np.zeros_like(escape_grid)
    dom_grid = np.zeros_like(escape_grid)

    for mi, mu in enumerate(mutation_rates):
        for pi, pr in enumerate(prophet_rates):
            runs = [r for r in all_results
                    if r["prophet_rate"] == pr and r["mutation_rate"] == mu]
            if runs:
                escape_grid[mi, pi] = sum(1 for r in runs if r["escaped"]) / len(runs)
                neff_grid[mi, pi] = np.mean([r["final"]["n_eff"] for r in runs])
                dom_grid[mi, pi] = np.mean([r["final"]["dominance"] for r in runs])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Escape probability
    im0 = axes[0].imshow(escape_grid, aspect='auto', origin='lower',
                          cmap='RdYlGn', vmin=0, vmax=1)
    axes[0].set_xticks(range(len(prophet_rates)))
    axes[0].set_xticklabels([f"{p:.3f}" for p in prophet_rates], rotation=45)
    axes[0].set_yticks(range(len(mutation_rates)))
    axes[0].set_yticklabels([f"{m:.2f}" for m in mutation_rates])
    axes[0].set_xlabel("Prophet rate")
    axes[0].set_ylabel("Mutation rate μ")
    axes[0].set_title("Escape Probability")
    plt.colorbar(im0, ax=axes[0])
    # Annotate cells
    for mi in range(len(mutation_rates)):
        for pi in range(len(prophet_rates)):
            axes[0].text(pi, mi, f"{escape_grid[mi,pi]:.0%}",
                        ha='center', va='center', fontsize=8,
                        color='white' if escape_grid[mi,pi] < 0.5 else 'black')

    # Final N_eff
    im1 = axes[1].imshow(neff_grid, aspect='auto', origin='lower',
                          cmap='viridis', vmin=1)
    axes[1].set_xticks(range(len(prophet_rates)))
    axes[1].set_xticklabels([f"{p:.3f}" for p in prophet_rates], rotation=45)
    axes[1].set_yticks(range(len(mutation_rates)))
    axes[1].set_yticklabels([f"{m:.2f}" for m in mutation_rates])
    axes[1].set_xlabel("Prophet rate")
    axes[1].set_ylabel("Mutation rate μ")
    axes[1].set_title("Final N_eff (deity count)")
    plt.colorbar(im1, ax=axes[1])
    for mi in range(len(mutation_rates)):
        for pi in range(len(prophet_rates)):
            axes[1].text(pi, mi, f"{neff_grid[mi,pi]:.1f}",
                        ha='center', va='center', fontsize=8, color='white')

    # Final Dominance
    im2 = axes[2].imshow(dom_grid, aspect='auto', origin='lower',
                          cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[2].set_xticks(range(len(prophet_rates)))
    axes[2].set_xticklabels([f"{p:.3f}" for p in prophet_rates], rotation=45)
    axes[2].set_yticks(range(len(mutation_rates)))
    axes[2].set_yticklabels([f"{m:.2f}" for m in mutation_rates])
    axes[2].set_xlabel("Prophet rate")
    axes[2].set_ylabel("Mutation rate μ")
    axes[2].set_title("Final Dominance D")
    plt.colorbar(im2, ax=axes[2])
    for mi in range(len(mutation_rates)):
        for pi in range(len(prophet_rates)):
            axes[2].text(pi, mi, f"{dom_grid[mi,pi]:.2f}",
                        ha='center', va='center', fontsize=8,
                        color='white' if dom_grid[mi,pi] > 0.5 else 'black')

    fig.suptitle("Prophet Escape from Monotheistic Lock-In\n"
                 "(Phase 1: γ=0.9 for 3000 steps → Phase 2: γ=0 for 5000 steps)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    buf = __import__('io').BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()


@app.local_entrypoint()
def main():
    prophet_rates = [0.0, 0.001, 0.003, 0.005, 0.01, 0.02]
    mutation_rates = [0.05, 0.10, 0.15, 0.20, 0.25]
    n_runs = 20

    print(f"Prophet Escape Experiment")
    print(f"  {len(prophet_rates)} prophet rates × {len(mutation_rates)} mutation rates × {n_runs} runs")
    print(f"  = {len(prophet_rates) * len(mutation_rates) * n_runs} total conditions")
    print()

    # Launch all conditions in parallel
    futures = []
    for pr in prophet_rates:
        for mu in mutation_rates:
            for run_id in range(n_runs):
                futures.append(
                    run_single_condition.spawn(pr, mu, run_id)
                )

    print(f"Launched {len(futures)} Modal tasks...")

    all_results = []
    done = 0
    for future in futures:
        result = future.get()
        all_results.append(result)
        done += 1
        if done % 50 == 0:
            print(f"  {done}/{len(futures)} complete")

    print(f"\nAll {len(all_results)} runs complete.")

    # Save raw data
    out_dir = Path("sim/runs/prophet_escape")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "prophet_escape_data.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Raw data saved to {out_dir / 'prophet_escape_data.json'}")

    # Generate plots
    print("\nGenerating plots...")
    plot_bytes = generate_plots.remote(all_results, str(out_dir))
    plot_path = out_dir / "prophet_escape_plot.png"
    plot_path.write_bytes(plot_bytes)
    print(f"Plot saved to {plot_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("ESCAPE PROBABILITY MATRIX")
    print("=" * 70)
    print(f"{'':>12}", end="")
    for pr in prophet_rates:
        print(f" | pr={pr:.3f}", end="")
    print()
    print("-" * 70)

    for mu in mutation_rates:
        print(f"  μ={mu:.2f}  ", end="")
        for pr in prophet_rates:
            runs = [r for r in all_results
                    if r["prophet_rate"] == pr and r["mutation_rate"] == mu]
            esc = sum(1 for r in runs if r["escaped"]) / len(runs) if runs else 0
            print(f" |  {esc:5.0%}  ", end="")
        print()

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # No-prophet baseline
    no_prophet = [r for r in all_results if r["prophet_rate"] == 0]
    esc_no_prophet = sum(1 for r in no_prophet if r["escaped"]) / len(no_prophet) if no_prophet else 0
    print(f"  Baseline (no prophets): {esc_no_prophet:.0%} escape rate")

    # Best condition
    best_esc = 0
    best_cond = (0, 0)
    for pr in prophet_rates:
        for mu in mutation_rates:
            runs = [r for r in all_results
                    if r["prophet_rate"] == pr and r["mutation_rate"] == mu]
            esc = sum(1 for r in runs if r["escaped"]) / len(runs) if runs else 0
            if esc > best_esc:
                best_esc = esc
                best_cond = (pr, mu)
    print(f"  Best condition: prophet_rate={best_cond[0]}, μ={best_cond[1]} → {best_esc:.0%} escape")

    # Prophet-only (default mutation)
    prophet_only = [r for r in all_results
                    if r["mutation_rate"] == 0.08 and r["prophet_rate"] > 0]
    if prophet_only:
        esc_po = sum(1 for r in prophet_only if r["escaped"]) / len(prophet_only)
        print(f"  Prophets only (μ=0.08): {esc_po:.0%} escape rate")

    # Mutation-only (no prophets)
    mut_only = [r for r in all_results
                if r["prophet_rate"] == 0 and r["mutation_rate"] > 0.08]
    if mut_only:
        esc_mo = sum(1 for r in mut_only if r["escaped"]) / len(mut_only)
        print(f"  Mutation only (no prophets): {esc_mo:.0%} escape rate")

    # Both required?
    both = [r for r in all_results
            if r["prophet_rate"] > 0 and r["mutation_rate"] > 0.08]
    if both:
        esc_both = sum(1 for r in both if r["escaped"]) / len(both)
        print(f"  Both (prophets + elevated μ): {esc_both:.0%} escape rate")

    print()
