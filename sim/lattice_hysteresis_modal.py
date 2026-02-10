#!/usr/bin/env python3
"""
Lattice Hysteresis Sweep on Modal — 8-Dot Braille + Hamming Centroids
=====================================================================

Same hysteresis sweep as hysteresis_modal.py, but with the key innovation:
centroids are computed as HAMMING MEANS on the 8-dot braille lattice
instead of arithmetic means in continuous space.

Hypothesis: Snap dynamics produce SHARPER phase transitions because
the discrete lattice eliminates continuous drift.

Usage:
  modal run sim/lattice_hysteresis_modal.py
"""

import modal
import json
import math
import os

app = modal.App("lattice-hysteresis-sweep")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("matplotlib", "numpy")
)

# ─── Inline constants ────────────────────────────────────────────────

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

DEITY_PRIORS = {}
for name, vec in DEITY_PRIORS_RAW.items():
    n = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    DEITY_PRIORS[name] = {k: v / n for k, v in vec.items()}


# ─── Single sweep function (runs in its own Modal container) ─────────

@app.function(image=image, timeout=3600, cpu=2)
def run_single_sweep(run_idx: int, seed_base: int, n_agents: int,
                     gamma_steps: list, equil_steps: int, meas_window: int,
                     use_hamming: bool = True):
    """
    Run one forward + reverse hysteresis sweep.

    If use_hamming=True, centroids are computed as Hamming means on the
    8-dot braille lattice (snap dynamics). Otherwise, arithmetic means
    in continuous space (baseline for comparison).
    """
    import random

    # ─── Inline utilities ─────────────────────────────────────────
    def _norm(a):
        return math.sqrt(sum(a[k] ** 2 for k in AXES))

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

    # ─── 8-dot Braille lattice (inlined) ──────────────────────────
    def encode_to_lattice(vec):
        """Encode continuous vec → list of 12 × 8-bit cells."""
        cells = []
        sorted_vals = sorted(vec.get(a, 0.0) for a in AXES)
        median_val = sorted_vals[len(AXES) // 2]
        for axis in AXES:
            value = vec.get(axis, 0.0)
            opposite = POLARITY_PAIRS.get(axis)
            opp_value = vec.get(opposite, 0.0) if opposite else 0.0
            pos_active = value > 0.3
            neg_active = opp_value > value + 0.1 if opposite else False
            tension = (pos_active and opp_value > 0.3 and
                       abs(value - opp_value) < 0.15) if opposite else False
            intensity = min(3, int(value * 4))
            dot4 = (intensity & 2) != 0
            dot5 = (intensity & 1) != 0
            rigid = value > 0.7
            salient = value > median_val
            momentum = False  # no prev_vec in sweep
            cells.append([pos_active, neg_active, tension, dot4, dot5,
                          rigid, salient, momentum])
        return cells

    def cells_to_bitstring(cells):
        bits = []
        for cell in cells:
            bits.extend(cell)
        return bits

    def bitstring_to_cells(bits):
        cells = []
        for i in range(0, len(bits), 8):
            cells.append(list(bits[i:i+8]))
        return cells

    def hamming_mean(all_cells, weights=None):
        """Majority-vote centroid on the 8-dot lattice."""
        n = len(all_cells)
        if n == 0:
            return [[False]*8 for _ in range(12)]
        all_bits = [cells_to_bitstring(c) for c in all_cells]
        n_bits = len(all_bits[0])
        result_bits = []
        for i in range(n_bits):
            if weights is None:
                ones = sum(1 for bs in all_bits if bs[i])
                result_bits.append(ones > n / 2)
            else:
                w_ones = sum(w for bs, w in zip(all_bits, weights) if bs[i])
                w_total = sum(weights)
                result_bits.append(w_ones > w_total / 2)
        return bitstring_to_cells(result_bits)

    def hamming_distance(cells_a, cells_b):
        ba = cells_to_bitstring(cells_a)
        bb = cells_to_bitstring(cells_b)
        return sum(1 for x, y in zip(ba, bb) if x != y)

    def decode_from_lattice(cells):
        """Decode lattice cells back to approximate continuous vector."""
        vec = {}
        for i, axis in enumerate(AXES):
            cell = cells[i]
            pos_active, neg_active, tension, dot4, dot5, rigid, salient, momentum = cell
            intensity = (2 if dot4 else 0) + (1 if dot5 else 0)
            value = (intensity + 0.5) / 4
            if not pos_active and neg_active:
                value *= 0.3
            if tension:
                value *= 0.85
            if rigid:
                value = max(value, 0.75)
            if salient:
                value *= 1.1
            vec[axis] = value
        return _normalize(vec)

    # ─── Agent ────────────────────────────────────────────────────
    class Agent:
        __slots__ = ('id', 'belief', 'w', 'assoc', 'freq', 'lattice')
        def __init__(self, id, belief, w):
            self.id = id
            self.belief = belief
            self.w = w
            self.assoc = {}
            self.freq = {}
            self.lattice = encode_to_lattice(belief)

    # ─── Kernel with dual centroid mode ───────────────────────────
    class LatticeKernel:
        def __init__(self, n_agents, coercion, seed, use_hamming):
            self.rng = random.Random(seed)
            self.N = n_agents
            self.coercion = coercion
            self.use_hamming = use_hamming
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

            # Init agents
            self.agents = []
            deity_names = list(DEITY_PRIORS.keys())
            for i in range(n_agents):
                chosen = self.rng.sample(deity_names, k=self.rng.randint(1, 2))
                belief = {k: 0.0 for k in AXES}
                for d in chosen:
                    _add_scaled(belief, DEITY_PRIORS[d], 1.0 / len(chosen))
                belief = _jitter(belief, self.rng, 0.15)
                self.agents.append(Agent(i, belief, 1.0))

            # Small-world graph
            self.neighbors = {i: [] for i in range(n_agents)}
            k_half = 2
            for i in range(n_agents):
                for j in range(1, k_half + 1):
                    nb = (i + j) % n_agents
                    if nb not in self.neighbors[i]:
                        self.neighbors[i].append(nb)
                    if i not in self.neighbors[nb]:
                        self.neighbors[nb].append(i)
            for i in range(n_agents):
                for j_idx in range(len(self.neighbors[i])):
                    if self.rng.random() < 0.1:
                        old = self.neighbors[i][j_idx]
                        candidates = [x for x in range(n_agents)
                                      if x != i and x not in self.neighbors[i]]
                        if candidates:
                            new = self.rng.choice(candidates)
                            self.neighbors[i][j_idx] = new
                            if old in self.neighbors and i in self.neighbors[old]:
                                self.neighbors[old].remove(i)
                            self.neighbors[new].append(i)

            # Seed theonyms
            for a in self.agents:
                for name in deity_names:
                    a.assoc[name] = _jitter(DEITY_PRIORS[name], self.rng, 0.1)
                    a.freq[name] = 0

            self.clusters = []
            self.centroids = []
            self.lattice_centroids = []  # 8-dot braille centroids
            self.total_flips = 0  # track cumulative cell flips

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
                blend = {k: 0.8 * ctx_vec[k] + 0.2 * a.belief[k] for k in AXES}
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
                    a.lattice = encode_to_lattice(a.belief)

            if self.t % 50 == 0:
                # Update all agent lattice projections
                for a in self.agents:
                    a.lattice = encode_to_lattice(a.belief)

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
                            a.lattice = encode_to_lattice(a.belief)

        def update_clusters(self):
            effective_threshold = self.cluster_threshold + 0.3 * self.coercion

            if self.use_hamming:
                # ─── HAMMING MODE: cluster on lattice, centroid = Hamming mean ───
                self.centroids = []
                self.clusters = []
                self.lattice_centroids = []

                for agent in self.agents:
                    if not self.lattice_centroids:
                        self.lattice_centroids.append(list(agent.lattice))
                        self.centroids.append(dict(agent.belief))
                        self.clusters.append([agent.id])
                        continue

                    # Use Hamming distance for clustering
                    distances = [hamming_distance(agent.lattice, lc) / 96.0
                                 for lc in self.lattice_centroids]
                    min_dist = min(distances)
                    best_idx = distances.index(min_dist)

                    if min_dist < effective_threshold:
                        self.clusters[best_idx].append(agent.id)
                    else:
                        self.lattice_centroids.append(list(agent.lattice))
                        self.centroids.append(dict(agent.belief))
                        self.clusters.append([agent.id])

                # Recompute centroids as Hamming means (SNAP)
                new_centroids = []
                new_lattice_centroids = []
                new_clusters = []
                for i, cluster in enumerate(self.clusters):
                    if not cluster:
                        continue
                    member_cells = [self.agents[aid].lattice for aid in cluster]
                    member_weights = [self.agents[aid].w for aid in cluster]

                    old_lc = self.lattice_centroids[i] if i < len(self.lattice_centroids) else None
                    new_lc = hamming_mean(member_cells, member_weights)

                    # Track flips
                    if old_lc is not None:
                        self.total_flips += hamming_distance(old_lc, new_lc)

                    new_lattice_centroids.append(new_lc)
                    # Decode for continuous-space operations
                    new_centroids.append(decode_from_lattice(new_lc))
                    new_clusters.append(cluster)

                # Fusion on lattice
                merge_dist_bits = int((0.15 + 0.35 * self.coercion) * 96)
                merged = True
                while merged:
                    merged = False
                    for i in range(len(new_lattice_centroids)):
                        for j in range(i + 1, len(new_lattice_centroids)):
                            if hamming_distance(new_lattice_centroids[i],
                                                new_lattice_centroids[j]) < merge_dist_bits:
                                combined = new_clusters[i] + new_clusters[j]
                                member_cells = [self.agents[aid].lattice for aid in combined]
                                member_weights = [self.agents[aid].w for aid in combined]
                                new_lattice_centroids[i] = hamming_mean(member_cells, member_weights)
                                new_centroids[i] = decode_from_lattice(new_lattice_centroids[i])
                                new_clusters[i] = combined
                                del new_lattice_centroids[j]
                                del new_centroids[j]
                                del new_clusters[j]
                                merged = True
                                break
                        if merged:
                            break

                self.centroids = new_centroids
                self.lattice_centroids = new_lattice_centroids
                self.clusters = new_clusters

            else:
                # ─── ARITHMETIC MODE: standard continuous centroids (baseline) ───
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
                        new_centroids.append(_scale(centroid, 1.0 / total_w))
                    else:
                        new_centroids.append(_scale(centroid, 1.0 / len(cluster)))
                    new_clusters.append(cluster)

                merge_dist = 0.15 + 0.35 * self.coercion
                merged_flag = True
                while merged_flag:
                    merged_flag = False
                    for i in range(len(new_centroids)):
                        for j in range(i + 1, len(new_centroids)):
                            if 1 - _cosine(new_centroids[i], new_centroids[j]) < merge_dist:
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
                                merged_flag = True
                                break
                        if merged_flag:
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
            return {
                "dominance": dominance,
                "n_eff": n_eff,
                "entropy": entropy,
                "total_flips": self.total_flips,
            }

    # ─── Run the sweep ────────────────────────────────────────────
    seed = seed_base + run_idx * 137

    def do_sweep(direction, seed_offset):
        gammas = gamma_steps if direction == "forward" else list(reversed(gamma_steps))
        kernel = LatticeKernel(n_agents, gammas[0], seed + seed_offset, use_hamming)
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
            total_flips = measurements[-1]["total_flips"] if measurements else 0
            results.append({
                "gamma": gamma, "dominance": avg_d, "n_eff": avg_n,
                "entropy": avg_h, "total_flips": total_flips,
            })
            print(f"  Run {run_idx} [{('HAM' if use_hamming else 'ARI')}] "
                  f"{direction} γ={gamma:.2f}: D={avg_d:.3f}, N_eff={avg_n:.1f}")
        return results

    forward = do_sweep("forward", 0)
    reverse = do_sweep("reverse", 1)
    return {"run_idx": run_idx, "use_hamming": use_hamming,
            "forward": forward, "reverse": reverse}


# ─── Plot generation ─────────────────────────────────────────────────

@app.function(image=image, timeout=3600)
def generate_comparison_plots(ham_fwd, ham_rev, arith_fwd, arith_rev, gamma_steps):
    """Generate comparison plot: Hamming vs Arithmetic centroids."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import io

    gammas = sorted(ham_fwd.keys())

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))

    for row, (fwd, rev, label, color_f, color_r) in enumerate([
        (arith_fwd, arith_rev, "Arithmetic (continuous)", "#c0392b", "#2980b9"),
        (ham_fwd, ham_rev, "Hamming (8-dot lattice)", "#e67e22", "#27ae60"),
    ]):
        d_f = [fwd[g]["dominance_mean"] for g in gammas]
        d_f_std = [fwd[g]["dominance_std"] for g in gammas]
        d_r = [rev[g]["dominance_mean"] for g in gammas]
        d_r_std = [rev[g]["dominance_std"] for g in gammas]

        neff_f = [fwd[g]["n_eff_mean"] for g in gammas]
        neff_r = [rev[g]["n_eff_mean"] for g in gammas]

        ent_f = [fwd[g]["entropy_mean"] for g in gammas]
        ent_r = [rev[g]["entropy_mean"] for g in gammas]

        for col, (f_mean, f_std, r_mean, r_std, ylabel, title) in enumerate([
            (d_f, d_f_std, d_r, d_r_std, "Dominance D", f"Hysteresis: {label}"),
            (neff_f, [0]*len(gammas), neff_r, [0]*len(gammas), "N_eff", f"Deity Count: {label}"),
            (ent_f, [0]*len(gammas), ent_r, [0]*len(gammas), "Entropy H", f"Entropy: {label}"),
        ]):
            ax = axes[row][col]
            ax.fill_between(gammas,
                            [m - s for m, s in zip(f_mean, f_std)],
                            [m + s for m, s in zip(f_mean, f_std)],
                            alpha=0.2, color=color_f)
            ax.plot(gammas, f_mean, 'o-', color=color_f, linewidth=2,
                    markersize=3, label="Forward (γ↑)")
            ax.fill_between(gammas,
                            [m - s for m, s in zip(r_mean, r_std)],
                            [m + s for m, s in zip(r_mean, r_std)],
                            alpha=0.2, color=color_r)
            ax.plot(gammas, r_mean, 's-', color=color_r, linewidth=2,
                    markersize=3, label="Reverse (γ↓)")
            ax.set_xlabel("Coercion γ", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.set_xlim(-0.02, 1.02)
            ax.grid(True, alpha=0.3)

    plt.suptitle("Hysteresis Sweep: Arithmetic vs 8-Dot Hamming Centroids",
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return buf.read()


# ─── Orchestrator ────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    import statistics

    N_AGENTS = 80
    GAMMA_STEPS = [round(g * 0.05, 2) for g in range(21)]  # 0.0 to 1.0
    EQUIL_STEPS = 2000
    MEAS_WINDOW = 500
    N_RUNS = 20  # 20 per mode × 2 modes = 40 total containers
    SEED_BASE = 42

    print("=" * 72)
    print("LATTICE HYSTERESIS SWEEP — Hamming vs Arithmetic Centroids")
    print(f"N_AGENTS={N_AGENTS}, γ steps={len(GAMMA_STEPS)}, "
          f"equil={EQUIL_STEPS}/γ, runs={N_RUNS}×2")
    print("=" * 72)

    # Launch all runs in parallel (both Hamming and Arithmetic)
    print(f"\nLaunching {N_RUNS * 2} parallel runs on Modal...")
    futures_ham = []
    futures_arith = []
    for run_idx in range(N_RUNS):
        futures_ham.append(
            run_single_sweep.spawn(
                run_idx, SEED_BASE, N_AGENTS,
                GAMMA_STEPS, EQUIL_STEPS, MEAS_WINDOW,
                use_hamming=True
            )
        )
        futures_arith.append(
            run_single_sweep.spawn(
                run_idx, SEED_BASE + 10000, N_AGENTS,
                GAMMA_STEPS, EQUIL_STEPS, MEAS_WINDOW,
                use_hamming=False
            )
        )

    # Collect
    ham_results = []
    for f in futures_ham:
        result = f.get()
        ham_results.append(result)
        print(f"  Hamming run {result['run_idx']} complete.")

    arith_results = []
    for f in futures_arith:
        result = f.get()
        arith_results.append(result)
        print(f"  Arithmetic run {result['run_idx']} complete.")

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

    ham_fwd = aggregate(ham_results, "forward")
    ham_rev = aggregate(ham_results, "reverse")
    arith_fwd = aggregate(arith_results, "forward")
    arith_rev = aggregate(arith_results, "reverse")

    # Save data
    output_dir = "sim/runs/lattice_hysteresis"
    os.makedirs(output_dir, exist_ok=True)

    raw_data = {
        "params": {
            "n_agents": N_AGENTS,
            "gamma_steps": GAMMA_STEPS,
            "equilibration_steps": EQUIL_STEPS,
            "measurement_window": MEAS_WINDOW,
            "n_runs": N_RUNS,
        },
        "hamming_forward": {str(k): v for k, v in ham_fwd.items()},
        "hamming_reverse": {str(k): v for k, v in ham_rev.items()},
        "arithmetic_forward": {str(k): v for k, v in arith_fwd.items()},
        "arithmetic_reverse": {str(k): v for k, v in arith_rev.items()},
    }
    data_path = os.path.join(output_dir, "lattice_hysteresis_data.json")
    with open(data_path, "w") as f:
        json.dump(raw_data, f, indent=2)
    print(f"\nData saved to {data_path}")

    # Generate comparison plot
    print("\nGenerating comparison plot...")
    plot_bytes = generate_comparison_plots.remote(
        ham_fwd, ham_rev, arith_fwd, arith_rev, GAMMA_STEPS
    )
    plot_path = os.path.join(output_dir, "hamming_vs_arithmetic.png")
    with open(plot_path, "wb") as f:
        f.write(plot_bytes)
    print(f"Plot saved to {plot_path}")

    # Summary table
    print(f"\n{'=' * 72}")
    print("COMPARISON: Hamming (8-dot) vs Arithmetic (continuous)")
    print(f"{'=' * 72}")
    print(f"\n{'γ':>6} | {'D_ham_f':>8} | {'D_ham_r':>8} | {'D_ari_f':>8} | {'D_ari_r':>8} | {'Δ_ham':>8} | {'Δ_ari':>8}")
    print("-" * 72)
    for g in GAMMA_STEPS:
        hf = ham_fwd[g]["dominance_mean"]
        hr = ham_rev[g]["dominance_mean"]
        af = arith_fwd[g]["dominance_mean"]
        ar = arith_rev[g]["dominance_mean"]
        print(f"{g:6.2f} | {hf:8.3f} | {hr:8.3f} | {af:8.3f} | {ar:8.3f} | "
              f"{hr - hf:+8.3f} | {ar - af:+8.3f}")

    # Identify critical points
    def find_critical(agg, threshold=0.7):
        for g in GAMMA_STEPS:
            if agg[g]["dominance_mean"] > threshold:
                return g
        return None

    ham_gc_plus = find_critical(ham_fwd)
    arith_gc_plus = find_critical(arith_fwd)

    print(f"\nHamming γ_c+ ≈ {ham_gc_plus}  |  Arithmetic γ_c+ ≈ {arith_gc_plus}")
    if ham_gc_plus and arith_gc_plus:
        if ham_gc_plus < arith_gc_plus:
            print("✓ Hamming centroids produce SHARPER phase transition (lower γ_c+)")
        elif ham_gc_plus == arith_gc_plus:
            print("= Same critical point — snap dynamics don't shift the threshold")
        else:
            print("~ Arithmetic transitions earlier — unexpected")
    print()
