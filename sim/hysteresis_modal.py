#!/usr/bin/env python3
"""
Hysteresis Sweep on Modal — Parallel Execution

The ABM is CPU-bound, not GPU-bound. Modal helps by running all 30
independent runs in parallel (one container per run) instead of sequentially.

Usage:
  modal run sim/hysteresis_modal.py
"""

import modal
import json
import math
import os

app = modal.App("hysteresis-sweep")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("matplotlib", "numpy")
)

# ─── Inline the simulation kernel (Modal needs self-contained code) ──

AXES = [
    "authority", "transcendence", "care", "justice", "wisdom", "power",
    "fertility", "war", "death", "creation", "nature", "order"
]

DEITY_PRIORS_RAW = {
    "zeus": {"authority": 0.9, "transcendence": 0.8, "care": 0.3, "justice": 0.7, "wisdom": 0.6, "power": 0.9, "fertility": 0.2, "war": 0.8, "death": 0.1, "creation": 0.4, "nature": 0.3, "order": 0.8},
    "odin": {"authority": 0.8, "transcendence": 0.7, "care": 0.4, "justice": 0.6, "wisdom": 0.9, "power": 0.7, "fertility": 0.1, "war": 0.9, "death": 0.8, "creation": 0.3, "nature": 0.2, "order": 0.5},
    "amun": {"authority": 0.9, "transcendence": 0.9, "care": 0.6, "justice": 0.8, "wisdom": 0.8, "power": 0.8, "fertility": 0.3, "war": 0.2, "death": 0.1, "creation": 0.9, "nature": 0.1, "order": 0.9},
    "marduk": {"authority": 0.9, "transcendence": 0.6, "care": 0.5, "justice": 0.9, "wisdom": 0.7, "power": 0.9, "fertility": 0.1, "war": 0.8, "death": 0.3, "creation": 0.7, "nature": 0.1, "order": 0.9},
    "indra": {"authority": 0.8, "transcendence": 0.5, "care": 0.4, "justice": 0.7, "wisdom": 0.6, "power": 0.9, "fertility": 0.2, "war": 0.9, "death": 0.2, "creation": 0.3, "nature": 0.4, "order": 0.6},
    "shango": {"authority": 0.7, "transcendence": 0.4, "care": 0.3, "justice": 0.8, "wisdom": 0.5, "power": 0.8, "fertility": 0.1, "war": 0.7, "death": 0.2, "creation": 0.2, "nature": 0.6, "order": 0.5},
    "kami": {"authority": 0.3, "transcendence": 0.8, "care": 0.8, "justice": 0.4, "wisdom": 0.7, "power": 0.4, "fertility": 0.6, "war": 0.1, "death": 0.1, "creation": 0.5, "nature": 0.9, "order": 0.8},
    "manitou": {"authority": 0.2, "transcendence": 0.9, "care": 0.9, "justice": 0.3, "wisdom": 0.8, "power": 0.3, "fertility": 0.7, "war": 0.1, "death": 0.2, "creation": 0.6, "nature": 0.9, "order": 0.4},
    "apollo": {"authority": 0.6, "transcendence": 0.7, "care": 0.5, "justice": 0.6, "wisdom": 0.8, "power": 0.6, "fertility": 0.3, "war": 0.3, "death": 0.2, "creation": 0.7, "nature": 0.4, "order": 0.7},
    "freya": {"authority": 0.4, "transcendence": 0.5, "care": 0.8, "justice": 0.4, "wisdom": 0.6, "power": 0.5, "fertility": 0.9, "war": 0.6, "death": 0.4, "creation": 0.6, "nature": 0.7, "order": 0.3},
    "yah": {"authority": 0.9, "transcendence": 0.9, "care": 0.6, "justice": 0.9, "wisdom": 0.9, "power": 0.9, "fertility": 0.2, "war": 0.4, "death": 0.3, "creation": 0.9, "nature": 0.3, "order": 0.9},
    "baal": {"authority": 0.7, "transcendence": 0.6, "care": 0.4, "justice": 0.6, "wisdom": 0.5, "power": 0.8, "fertility": 0.6, "war": 0.6, "death": 0.3, "creation": 0.5, "nature": 0.7, "order": 0.6},
}

# Normalize deity priors
DEITY_PRIORS = {}
for name, vec in DEITY_PRIORS_RAW.items():
    n = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    DEITY_PRIORS[name] = {k: v / n for k, v in vec.items()}


# ─── Minimal inline kernel (just what we need for the sweep) ─────────

@app.function(image=image, timeout=3600, cpu=2)
def run_single_sweep(run_idx: int, seed_base: int, n_agents: int,
                     gamma_steps: list, equil_steps: int, meas_window: int):
    """Run one forward + reverse sweep. Called in parallel by Modal."""
    import random
    import statistics

    # --- Inline utility functions ---
    def _norm(a):
        return math.sqrt(sum(a[k] * a[k] for k in AXES))

    def _cosine(a, b):
        na, nb = _norm(a), _norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return sum(a[k] * b[k] for k in AXES) / (na * nb)

    def _normalize(a):
        n = _norm(a) or 1.0
        return {k: a[k] / n for k in AXES}

    def _rnd_unit(rng):
        v = {a: rng.random() for a in AXES}
        return _normalize(v)

    def _jitter(base, rng, noise=0.1):
        v = {k: base[k] + rng.gauss(0, noise) for k in AXES}
        return _normalize(v)

    def _add_scaled(a, b, s):
        for k in AXES:
            a[k] += b[k] * s

    def _scale(a, s):
        return {k: a[k] * s for k in AXES}

    def _clamp(x, lo, hi):
        return max(lo, min(hi, x))

    # --- Agent ---
    class Agent:
        __slots__ = ('id', 'belief', 'w', 'assoc', 'freq')
        def __init__(self, id, belief, w):
            self.id = id
            self.belief = belief
            self.w = w
            self.assoc = {}
            self.freq = {}

    # --- Minimal kernel ---
    class MiniKernel:
        def __init__(self, n_agents, coercion, seed):
            self.rng = random.Random(seed)
            self.N = n_agents
            self.coercion = coercion
            self.mutation_rate = 0.08
            self.learning_rate = 0.08
            self.penalty_rate = 0.02
            self.prestige_alpha = 0.20
            self.belief_influence = 0.15
            self.base_success_thresh = 0.58
            self.cluster_threshold = 0.4
            self.ritual_period = 50
            self.ritual_bonus = 0.10
            self.t = 0

            # Init agents from deity priors
            self.agents = []
            deity_names = list(DEITY_PRIORS.keys())
            for i in range(n_agents):
                chosen = self.rng.sample(deity_names, k=self.rng.randint(1, 2))
                belief = {k: 0.0 for k in AXES}
                for d in chosen:
                    _add_scaled(belief, DEITY_PRIORS[d], 1.0 / len(chosen))
                belief = _jitter(belief, self.rng, 0.15)
                self.agents.append(Agent(i, belief, 1.0))

            # Build small-world graph
            self.neighbors = {i: [] for i in range(n_agents)}
            k_half = 2
            for i in range(n_agents):
                for j in range(1, k_half + 1):
                    nb = (i + j) % n_agents
                    if nb not in self.neighbors[i]:
                        self.neighbors[i].append(nb)
                    if i not in self.neighbors[nb]:
                        self.neighbors[nb].append(i)
            # Rewire
            for i in range(n_agents):
                for j_idx in range(len(self.neighbors[i])):
                    if self.rng.random() < 0.1:
                        old = self.neighbors[i][j_idx]
                        candidates = [x for x in range(n_agents) if x != i and x not in self.neighbors[i]]
                        if candidates:
                            new = self.rng.choice(candidates)
                            self.neighbors[i][j_idx] = new
                            if old in self.neighbors and i in self.neighbors[old]:
                                self.neighbors[old].remove(i)
                            self.neighbors[new].append(i)

            # Seed theonyms
            theonyms = list(DEITY_PRIORS.keys())
            for a in self.agents:
                for name in theonyms:
                    a.assoc[name] = _jitter(DEITY_PRIORS[name], self.rng, 0.1)
                    a.freq[name] = 0

            self.clusters = []
            self.centroids = []

        def _sample_context(self):
            tasks = ["forage", "warn", "trade", "mourn", "build", "raid"]
            v = {k: 0.0 for k in AXES}
            task = self.rng.choice(tasks)
            if task == "forage": v["care"] += 0.4; v["power"] += 0.3; v["nature"] += 0.1
            elif task == "warn": v["authority"] += 0.4; v["wisdom"] += 0.2; v["justice"] += 0.2
            elif task == "trade": v["justice"] += 0.3; v["authority"] += 0.2; v["care"] += 0.2
            elif task == "mourn": v["death"] += 0.5; v["transcendence"] += 0.2; v["wisdom"] += 0.2
            elif task == "build": v["creation"] += 0.4; v["order"] += 0.2; v["power"] += 0.2
            elif task == "raid": v["war"] += 0.3; v["justice"] += 0.3; v["power"] += 0.2
            return _normalize(v)

        def _select_hearers(self, speaker):
            nbs = self.neighbors.get(speaker.id, [])
            if not nbs:
                others = [a for a in self.agents if a.id != speaker.id]
                return self.rng.sample(others, min(2, len(others)))
            if self.coercion > 0:
                weights = []
                for nid in nbs:
                    sim = _cosine(speaker.belief, self.agents[nid].belief)
                    weights.append(math.exp(sim * (1 + 9 * self.coercion)))
                chosen = self.rng.choices(nbs, weights=weights, k=min(2, len(nbs)))
            else:
                chosen = self.rng.sample(nbs, min(2, len(nbs)))
            return [self.agents[i] for i in chosen]

        def _produce(self, agent, ctx_vec):
            scored = []
            total_freq = sum(agent.freq.values()) + 1
            for form, vec in agent.assoc.items():
                ctx_score = sum(vec[k] * ctx_vec[k] for k in AXES)
                belief_score = sum(vec[k] * agent.belief[k] for k in AXES) * self.belief_influence
                freq_score = 0.1 * math.log((agent.freq.get(form, 0) + 1) / total_freq)
                scored.append((form, ctx_score + belief_score + freq_score))
            # Softmax choice (3 tokens)
            msg = []
            for _ in range(3):
                if self.rng.random() < 0.10:
                    msg.append(self.rng.choice([k for k, _ in scored]))
                else:
                    mx = max(s for _, s in scored) if scored else 1.0
                    exps = [(k, math.exp(s - mx)) for k, s in scored]
                    z = sum(v for _, v in exps) or 1.0
                    r = self.rng.random() * z
                    acc = 0.0
                    for k, v in exps:
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
            return {k: v[k] / n for k in AXES}

        def step(self):
            self.t += 1
            ctx_vec = self._sample_context()
            # Speaker selection (prestige-weighted)
            weights = [a.w for a in self.agents]
            speaker = self.rng.choices(self.agents, weights=weights, k=1)[0]
            hearers = self._select_hearers(speaker)
            msg = self._produce(speaker, ctx_vec)

            # Interpret & check success
            preds = [self._interpret(h, msg) for h in hearers]
            sim = sum(_cosine(ctx_vec, p) for p in preds) / max(1, len(preds))
            ritual = (self.t % self.ritual_period == 0)
            thresh = self.base_success_thresh + (self.ritual_bonus if ritual else 0.0)
            success = sim >= thresh

            # Learn
            lr = self.learning_rate if success else -self.penalty_rate
            for a in [speaker] + hearers:
                blend = {k: 0.8 * ctx_vec[k] + 0.2 * a.belief[k] for k in AXES}
                for tok in msg:
                    if tok not in a.assoc:
                        a.assoc[tok] = _rnd_unit(self.rng)
                    _add_scaled(a.assoc[tok], blend, lr)
                    a.freq[tok] = a.freq.get(tok, 0) + 1

            # Prestige
            for a in [speaker] + hearers:
                delta = self.prestige_alpha if success else -self.prestige_alpha * 0.3
                a.w = _clamp(a.w * (1.0 + delta), 0.1, 10.0)

            # Mutation
            for a in [speaker] + hearers:
                if self.rng.random() < self.mutation_rate:
                    for k in AXES:
                        a.belief[k] += self.rng.gauss(0, 0.05)
                    a.belief = _normalize(a.belief)

            # Clustering + attractor deepening (Definition 4a, §4.1)
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

        def update_clusters(self):
            # Coercion widens the absorption radius (§4.1: basin widening)
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

            # Prestige-weighted centroids (Definition 5)
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
                    new_centroids.append(_scale(centroid, 1.0 / total_w))
                else:
                    new_centroids.append(_scale(centroid, 1.0 / len(cluster)))
                new_clusters.append(cluster)

            # Fusion (§3.1): merge nearby centroids under coercion
            merge_dist = 0.15 + 0.35 * self.coercion
            merged = True
            while merged:
                merged = False
                for i in range(len(new_centroids)):
                    for j in range(i + 1, len(new_centroids)):
                        if 1 - _cosine(new_centroids[i], new_centroids[j]) < merge_dist:
                            # Merge j into i (prestige-weighted)
                            combined = new_clusters[i] + new_clusters[j]
                            centroid = {k: 0.0 for k in AXES}
                            total_w = 0.0
                            for aid in combined:
                                w = self.agents[aid].w
                                _add_scaled(centroid, self.agents[aid].belief, w)
                                total_w += w
                            if total_w > 0:
                                new_centroids[i] = _scale(centroid, 1.0 / total_w)
                            new_clusters[i] = combined
                            del new_centroids[j]
                            del new_clusters[j]
                            merged = True
                            break
                    if merged:
                        break

            self.centroids = new_centroids
            self.clusters = new_clusters

        def measure(self):
            self.update_clusters()
            N = len(self.agents)
            sizes = [len(c) for c in self.clusters]
            n_eff = sum(1 for s in sizes if s >= 2)
            dominance = max(sizes) / N if N > 0 and sizes else 0.0
            entropy = 0.0
            for s in sizes:
                if s > 0:
                    p = s / N
                    entropy -= p * math.log(p)
            return {"dominance": dominance, "n_eff": n_eff, "entropy": entropy}

    # --- Run the sweep ---
    seed = seed_base + run_idx * 137

    def do_sweep(direction, seed_offset):
        gammas = gamma_steps if direction == "forward" else list(reversed(gamma_steps))
        kernel = MiniKernel(n_agents, gammas[0], seed + seed_offset)
        results = []
        for gamma in gammas:
            kernel.coercion = gamma
            measurements = []
            for step in range(equil_steps):
                kernel.step()
                if step >= equil_steps - meas_window and step % 25 == 0:
                    measurements.append(kernel.measure())
            avg_d = sum(m["dominance"] for m in measurements) / len(measurements)
            avg_n = sum(m["n_eff"] for m in measurements) / len(measurements)
            avg_h = sum(m["entropy"] for m in measurements) / len(measurements)
            results.append({"gamma": gamma, "dominance": avg_d, "n_eff": avg_n, "entropy": avg_h})
            print(f"  Run {run_idx} {direction} γ={gamma:.2f}: D={avg_d:.3f}, N_eff={avg_n:.1f}")
        return results

    forward = do_sweep("forward", 0)
    reverse = do_sweep("reverse", 1)
    return {"run_idx": run_idx, "forward": forward, "reverse": reverse}


# ─── Orchestrator ────────────────────────────────────────────────────

@app.function(image=image, timeout=7200)
def generate_plots(forward_agg, reverse_agg, gamma_steps):
    """Generate hero plot from aggregated results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import io

    gammas = sorted(forward_agg.keys())

    d_f = [forward_agg[g]["dominance_mean"] for g in gammas]
    d_f_std = [forward_agg[g]["dominance_std"] for g in gammas]
    d_r = [reverse_agg[g]["dominance_mean"] for g in gammas]
    d_r_std = [reverse_agg[g]["dominance_std"] for g in gammas]

    neff_f = [forward_agg[g]["n_eff_mean"] for g in gammas]
    neff_f_std = [forward_agg[g]["n_eff_std"] for g in gammas]
    neff_r = [reverse_agg[g]["n_eff_mean"] for g in gammas]
    neff_r_std = [reverse_agg[g]["n_eff_std"] for g in gammas]

    ent_f = [forward_agg[g]["entropy_mean"] for g in gammas]
    ent_f_std = [forward_agg[g]["entropy_std"] for g in gammas]
    ent_r = [reverse_agg[g]["entropy_mean"] for g in gammas]
    ent_r_std = [reverse_agg[g]["entropy_std"] for g in gammas]

    # ─── Hero plot ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, (f_mean, f_std, r_mean, r_std, ylabel, title) in zip(axes, [
        (d_f, d_f_std, d_r, d_r_std, "Dominance D", "Hysteresis: Dominance"),
        (neff_f, neff_f_std, neff_r, neff_r_std, "Effective deity count N_eff", "Deity Count Collapse"),
        (ent_f, ent_f_std, ent_r, ent_r_std, "Belief entropy H", "Entropy: Diversity Lost"),
    ]):
        ax.fill_between(gammas,
                        [m - s for m, s in zip(f_mean, f_std)],
                        [m + s for m, s in zip(f_mean, f_std)],
                        alpha=0.2, color="#c0392b")
        ax.plot(gammas, f_mean, 'o-', color="#c0392b", linewidth=2.5,
                markersize=4, label="Forward (γ↑): polytheism → monotheism")

        ax.fill_between(gammas,
                        [m - s for m, s in zip(r_mean, r_std)],
                        [m + s for m, s in zip(r_mean, r_std)],
                        alpha=0.2, color="#2980b9")
        ax.plot(gammas, r_mean, 's-', color="#2980b9", linewidth=2.5,
                markersize=4, label="Reverse (γ↓): monotheism → ?")

        ax.set_xlabel("Coercion parameter γ", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

    # Annotate critical points on dominance panel
    ax0 = axes[0]
    gamma_c_plus = None
    for i, g in enumerate(gammas):
        if d_f[i] > 0.7:
            gamma_c_plus = g
            break
    gamma_c_minus = None
    for g in gammas:
        if reverse_agg[g]["dominance_mean"] > 0.7:
            gamma_c_minus = g
        else:
            break

    if gamma_c_plus is not None:
        ax0.axvline(x=gamma_c_plus, color="#c0392b", linestyle="--", alpha=0.5)
        ax0.annotate(f"γ$_c^+$ ≈ {gamma_c_plus:.2f}",
                     xy=(gamma_c_plus, 0.75), fontsize=10, color="#c0392b",
                     xytext=(gamma_c_plus + 0.08, 0.55),
                     arrowprops=dict(arrowstyle="->", color="#c0392b"))
    if gamma_c_minus is not None and gamma_c_minus != gamma_c_plus:
        ax0.axvline(x=gamma_c_minus, color="#2980b9", linestyle="--", alpha=0.5)
        ax0.annotate(f"γ$_c^-$ ≈ {gamma_c_minus:.2f}",
                     xy=(gamma_c_minus, 0.75), fontsize=10, color="#2980b9",
                     xytext=(max(0, gamma_c_minus - 0.15), 0.45),
                     arrowprops=dict(arrowstyle="->", color="#2980b9"))

    plt.tight_layout()

    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return buf.read()


@app.local_entrypoint()
def main():
    import statistics

    N_AGENTS = 80
    GAMMA_STEPS = [round(g * 0.05, 2) for g in range(21)]
    EQUIL_STEPS = 2000
    MEAS_WINDOW = 500
    N_RUNS = 30
    SEED_BASE = 42

    print("=" * 60)
    print("HYSTERESIS SWEEP — Modal Parallel Execution")
    print(f"N_AGENTS={N_AGENTS}, γ steps={len(GAMMA_STEPS)}, "
          f"equil={EQUIL_STEPS}/γ, runs={N_RUNS}")
    print("=" * 60)

    # Launch all runs in parallel
    print(f"\nLaunching {N_RUNS} parallel runs on Modal...")
    futures = []
    for run_idx in range(N_RUNS):
        futures.append(
            run_single_sweep.spawn(
                run_idx, SEED_BASE, N_AGENTS,
                GAMMA_STEPS, EQUIL_STEPS, MEAS_WINDOW
            )
        )

    # Collect results
    all_results = []
    for f in futures:
        result = f.get()
        all_results.append(result)
        print(f"  Run {result['run_idx']} complete.")

    # Aggregate
    def aggregate(all_runs, key):
        by_gamma = {}
        for run in all_runs:
            for point in run[key]:
                g = point["gamma"]
                if g not in by_gamma:
                    by_gamma[g] = {"dominance": [], "n_eff": [], "entropy": []}
                by_gamma[g]["dominance"].append(point["dominance"])
                by_gamma[g]["n_eff"].append(point["n_eff"])
                by_gamma[g]["entropy"].append(point["entropy"])
        result = {}
        for g in sorted(by_gamma.keys()):
            d = by_gamma[g]
            result[g] = {
                "gamma": g,
                "dominance_mean": statistics.mean(d["dominance"]),
                "dominance_std": statistics.stdev(d["dominance"]) if len(d["dominance"]) > 1 else 0,
                "n_eff_mean": statistics.mean(d["n_eff"]),
                "n_eff_std": statistics.stdev(d["n_eff"]) if len(d["n_eff"]) > 1 else 0,
                "entropy_mean": statistics.mean(d["entropy"]),
                "entropy_std": statistics.stdev(d["entropy"]) if len(d["entropy"]) > 1 else 0,
            }
        return result

    forward_agg = aggregate(all_results, "forward")
    reverse_agg = aggregate(all_results, "reverse")

    # Save raw data
    output_dir = "sim/runs/hysteresis_sweep"
    os.makedirs(output_dir, exist_ok=True)

    raw_data = {
        "params": {
            "n_agents": N_AGENTS,
            "gamma_steps": GAMMA_STEPS,
            "equilibration_steps": EQUIL_STEPS,
            "measurement_window": MEAS_WINDOW,
            "n_runs": N_RUNS,
        },
        "forward": {str(k): v for k, v in forward_agg.items()},
        "reverse": {str(k): v for k, v in reverse_agg.items()},
    }
    data_path = os.path.join(output_dir, "hysteresis_data.json")
    with open(data_path, "w") as f:
        json.dump(raw_data, f, indent=2)
    print(f"\nRaw data saved to {data_path}")

    # Generate plot on Modal
    print("\nGenerating hero plot...")
    plot_bytes = generate_plots.remote(forward_agg, reverse_agg, GAMMA_STEPS)
    hero_path = os.path.join(output_dir, "hero_plot.png")
    with open(hero_path, "wb") as f:
        f.write(plot_bytes)
    print(f"Hero plot saved to {hero_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'γ':>6} | {'D_fwd':>8} | {'D_rev':>8} | {'gap':>8} | {'Neff_fwd':>8} | {'Neff_rev':>8}")
    print("-" * 60)
    for g in GAMMA_STEPS:
        f = forward_agg[g]
        r = reverse_agg[g]
        gap = r["dominance_mean"] - f["dominance_mean"]
        print(f"{g:6.2f} | {f['dominance_mean']:8.3f} | {r['dominance_mean']:8.3f} | "
              f"{gap:+8.3f} | {f['n_eff_mean']:8.1f} | {r['n_eff_mean']:8.1f}")

    # Identify critical points
    gamma_c_plus = None
    for g in GAMMA_STEPS:
        if forward_agg[g]["dominance_mean"] > 0.7:
            gamma_c_plus = g
            break

    gamma_c_minus = None
    for g in reversed(GAMMA_STEPS):
        if reverse_agg[g]["dominance_mean"] < 0.7:
            gamma_c_minus = g
            break

    print(f"\nForward critical point γ_c+ ≈ {gamma_c_plus}")
    print(f"Reverse critical point γ_c- ≈ {gamma_c_minus}")
    if gamma_c_plus is not None and gamma_c_minus is not None:
        print(f"Hysteresis gap: Δγ = {gamma_c_plus - gamma_c_minus:.2f}")
        if gamma_c_minus < gamma_c_plus:
            print("✓ HYSTERESIS CONFIRMED: reverse threshold < forward threshold")
        else:
            print("~ No clear hysteresis — system may be reversible at these parameters")
    print()
