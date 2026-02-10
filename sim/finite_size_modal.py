#!/usr/bin/env python3
"""
Finite-Size Scaling Experiment (Modal)
======================================
Tests whether the polytheism→monotheism phase transition sharpens
with increasing system size N, as expected for a first-order transition.

Protocol:
  - N ∈ {80, 200, 500}
  - Forward sweep: γ from 0 to 1 (21 levels, 2000 steps each)
  - Reverse sweep: γ from 1 to 0
  - 15 runs per N value
  - Corpus-calibrated θ=0.12

Expected result for first-order transition:
  - D(γ) jump becomes sharper (steeper slope at γ_c)
  - Hysteresis gap narrows slightly but persists
  - Error bars shrink (less stochastic variation)
"""

import modal
import json
import math
import random
from pathlib import Path

app = modal.App("finite-size-scaling")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "matplotlib", "numpy"
)

# ─── Inline minimal kernel (same as hysteresis_modal / prophet_escape) ───

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
    def __init__(self, n_agents=80, coercion=0.0, mutation_rate=0.05,
                 cluster_threshold=0.12, seed=42):
        self.rng = random.Random(seed)
        self.n_agents = n_agents
        self.coercion = coercion
        self.mutation_rate = mutation_rate
        self.cluster_threshold = cluster_threshold
        self.t = 0
        self.agents = [_Agent(i, _rnd_unit(self.rng)) for i in range(n_agents)]
        self.centroids = []
        self.clusters = []
        self.learning_rate = 0.08
        self.penalty_rate = 0.02
        self.prestige_alpha = 0.20
        self.ritual_period = 50
        self.ritual_bonus = 0.10
        self.base_success_thresh = 0.58
        self.belief_influence = 0.07  # corpus-calibrated

        theonyms = ["isis","thor","indra","yah","nana","osiris","zeus","odin",
                     "ra","anu","ishtar","baal","amun","enar","shen","kami"]
        for a in self.agents:
            for name in theonyms:
                a.assoc[name] = _rnd_unit(self.rng)
                a.freq[name] = 0

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
        ctx_vec = _rnd_unit(self.rng)
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
        return {"dominance": dominance, "n_eff": n_eff, "entropy": entropy}


# ─── Modal functions ───

@app.function(image=image, timeout=1800)
def run_sweep(n_agents: int, run_id: int,
              gamma_levels: int = 21, steps_per_level: int = 2000):
    """Run forward + reverse hysteresis sweep for a given N."""
    gammas = [i / (gamma_levels - 1) for i in range(gamma_levels)]
    seed = run_id * 10000 + n_agents

    # Scale cluster threshold with N to correct for percolation:
    # In 12D, random vectors have mean pairwise cosine ~0. With N agents,
    # the probability of spurious clustering grows. We scale θ down slightly.
    # θ_eff = θ_base * (80/N)^0.25 — mild correction preserving the corpus value at N=80
    theta_eff = 0.12 * (80 / n_agents) ** 0.25

    k = MiniKernel(n_agents=n_agents, coercion=0.0,
                   mutation_rate=0.05, cluster_threshold=theta_eff, seed=seed)

    # Warm-up: let the system establish polytheistic baseline at γ=0
    warmup_steps = 1000 + n_agents * 5
    for _ in range(warmup_steps):
        k.step()

    forward = []
    for g in gammas:
        k.coercion = g
        for _ in range(steps_per_level):
            k.step()
        m = k.measure()
        m["gamma"] = g
        forward.append(m)

    reverse = []
    for g in reversed(gammas):
        k.coercion = g
        for _ in range(steps_per_level):
            k.step()
        m = k.measure()
        m["gamma"] = g
        reverse.append(m)

    return {
        "n_agents": n_agents,
        "run_id": run_id,
        "forward": forward,
        "reverse": list(reversed(reverse)),  # re-order by ascending γ
    }


@app.function(image=image, timeout=300)
def generate_plots(all_results: list):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agent_counts = sorted(set(r["n_agents"] for r in all_results))
    colors = {80: "#e74c3c", 200: "#3498db", 500: "#2ecc71"}
    labels = {80: "N=80", 200: "N=200", 500: "N=500"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = [("dominance", "Dominance D"), ("n_eff", "Effective Deity Count N_eff"),
               ("entropy", "Belief Entropy H")]

    for ax, (metric, title) in zip(axes, metrics):
        for N in agent_counts:
            runs = [r for r in all_results if r["n_agents"] == N]
            gammas = [pt["gamma"] for pt in runs[0]["forward"]]

            # Forward
            fwd_vals = np.array([[pt[metric] for pt in r["forward"]] for r in runs])
            fwd_mean = fwd_vals.mean(axis=0)
            fwd_std = fwd_vals.std(axis=0)
            ax.plot(gammas, fwd_mean, '-', color=colors[N], linewidth=2,
                    label=f'{labels[N]} fwd', alpha=0.9)
            ax.fill_between(gammas, fwd_mean - fwd_std, fwd_mean + fwd_std,
                           color=colors[N], alpha=0.15)

            # Reverse
            rev_vals = np.array([[pt[metric] for pt in r["reverse"]] for r in runs])
            rev_mean = rev_vals.mean(axis=0)
            rev_std = rev_vals.std(axis=0)
            ax.plot(gammas, rev_mean, '--', color=colors[N], linewidth=2,
                    label=f'{labels[N]} rev', alpha=0.9)
            ax.fill_between(gammas, rev_mean - rev_std, rev_mean + rev_std,
                           color=colors[N], alpha=0.10)

        ax.set_xlabel("Coercion γ", fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Finite-Size Scaling: Hysteresis at N = 80, 200, 500\n"
                 "(corpus-calibrated θ=0.12, solid=forward, dashed=reverse)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    buf = __import__('io').BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()


@app.local_entrypoint()
def main():
    agent_counts = [80, 200, 500]
    n_runs = 15

    total = len(agent_counts) * n_runs
    print(f"Finite-Size Scaling Experiment")
    print(f"  N ∈ {agent_counts}, {n_runs} runs each = {total} sweeps")
    print(f"  Corpus-calibrated: θ=0.12, μ=0.05, β=0.07")
    print()

    futures = []
    for N in agent_counts:
        for run_id in range(n_runs):
            futures.append(run_sweep.spawn(N, run_id))

    print(f"Launched {len(futures)} Modal tasks...")

    all_results = []
    done = 0
    for future in futures:
        result = future.get()
        all_results.append(result)
        done += 1
        if done % 10 == 0:
            print(f"  {done}/{len(futures)} complete")

    print(f"\nAll {len(all_results)} runs complete.")

    out_dir = Path("sim/runs/finite_size")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "finite_size_data.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Data saved to {out_dir / 'finite_size_data.json'}")

    print("\nGenerating plots...")
    plot_bytes = generate_plots.remote(all_results)
    plot_path = out_dir / "finite_size_plot.png"
    plot_path.write_bytes(plot_bytes)
    print(f"Plot saved to {plot_path}")

    # Summary
    print("\n" + "=" * 70)
    print("FINITE-SIZE SCALING SUMMARY")
    print("=" * 70)

    for N in agent_counts:
        runs = [r for r in all_results if r["n_agents"] == N]

        # Forward: find γ where D first exceeds 0.7
        fwd_gammas = [pt["gamma"] for pt in runs[0]["forward"]]
        fwd_dom = [sum(r["forward"][i]["dominance"] for r in runs) / len(runs)
                   for i in range(len(fwd_gammas))]

        gamma_c_fwd = None
        for i, d in enumerate(fwd_dom):
            if d >= 0.7:
                gamma_c_fwd = fwd_gammas[i]
                break

        # Reverse: find γ where D drops below 0.7
        rev_dom = [sum(r["reverse"][i]["dominance"] for r in runs) / len(runs)
                   for i in range(len(fwd_gammas))]

        gamma_c_rev = None
        for i in range(len(rev_dom) - 1, -1, -1):
            if rev_dom[i] >= 0.7:
                gamma_c_rev = fwd_gammas[i]
                break

        # Transition slope: max |dD/dγ| in forward sweep
        max_slope = 0
        for i in range(1, len(fwd_dom)):
            slope = abs(fwd_dom[i] - fwd_dom[i-1]) / (fwd_gammas[i] - fwd_gammas[i-1])
            max_slope = max(max_slope, slope)

        gap = (gamma_c_fwd - gamma_c_rev) if gamma_c_fwd and gamma_c_rev else None

        print(f"\n  N = {N}:")
        print(f"    γ_c+ (forward)  = {gamma_c_fwd}")
        print(f"    γ_c- (reverse)  = {gamma_c_rev}")
        print(f"    Hysteresis gap  = {gap:.2f}" if gap else "    Hysteresis gap  = N/A")
        print(f"    Max |dD/dγ|     = {max_slope:.2f}")
        print(f"    D at γ=0 (fwd)  = {fwd_dom[0]:.3f}")
        print(f"    D at γ=0 (rev)  = {rev_dom[0]:.3f}")

    print()
