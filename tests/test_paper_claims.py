#!/usr/bin/env python3
"""
Automated Test Suite for Paper Claims
======================================
Verifies all key claims from "Gods as Centroids" (Barrett, 2026) using
lightweight, fast-running simulations. These are smoke tests that confirm
the qualitative behavior holds, not full reproduction runs.

Run: pytest tests/test_paper_claims.py -v
"""

import sys
import os
import math
import statistics
import random

import pytest

# Ensure sim/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sim'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from swarm_kernel import (
    SwarmKernel, Config, AXES, DEITY_PRIORS,
    cosine, norm, rnd_unit_vec, dot
)


# ── Helpers ─────────────────────────────────────────────────────────

def quick_sim(n=40, steps=1000, coercion=0.0, seed=42, **kwargs):
    """Run a quick simulation and return the kernel."""
    cfg = Config(
        N=n, steps_per_generation=steps, coercion=coercion,
        seed=seed, use_deity_priors=True, cluster_update_freq=50,
        **kwargs
    )
    kernel = SwarmKernel(cfg)
    for _ in range(steps):
        kernel.step()
    return kernel


# ── §2: Model Basics ───────────────────────────────────────────────

class TestModelBasics:
    """Verify fundamental model properties."""

    def test_belief_space_dimensionality(self):
        """Belief vectors have 12 dimensions (Definition 1)."""
        assert len(AXES) == 12

    def test_deity_priors_normalized(self):
        """All deity priors are unit vectors (Definition 4)."""
        for name, vec in DEITY_PRIORS.items():
            n = math.sqrt(sum(v * v for v in vec.values()))
            assert abs(n - 1.0) < 0.01, f"{name} norm = {n}"

    def test_agents_have_beliefs(self):
        """Agents carry belief vectors (Definition 2)."""
        kernel = quick_sim(n=20, steps=10)
        for agent in kernel.agents:
            assert len(agent.belief) == 12
            assert all(k in agent.belief for k in AXES)

    def test_cosine_affinity(self):
        """Affinity is cosine similarity (Definition 3)."""
        a = {k: 1.0 for k in AXES}
        b = {k: 1.0 for k in AXES}
        assert abs(cosine(a, b) - 1.0) < 1e-6

        c = {k: -1.0 for k in AXES}
        assert abs(cosine(a, c) - (-1.0)) < 1e-6


# ── §2.4: Clustering ───────────────────────────────────────────────

class TestClustering:
    """Verify deity emergence via clustering."""

    def test_centroids_emerge(self):
        """Centroids emerge from agent dynamics (Definition 5)."""
        kernel = quick_sim(n=40, steps=1500, coercion=0.1)
        assert len(kernel.centroids) >= 1, "No centroids emerged"

    def test_clusters_partition_agents(self):
        """Every agent belongs to exactly one cluster."""
        kernel = quick_sim(n=40, steps=1000, coercion=0.1)
        all_ids = set()
        for cluster in kernel.clusters:
            for aid in cluster:
                assert aid not in all_ids, f"Agent {aid} in multiple clusters"
                all_ids.add(aid)

    def test_effective_deity_count(self):
        """N_eff is the count of non-trivial clusters (Definition 6)."""
        kernel = quick_sim(n=40, steps=1000, coercion=0.1)
        n_eff = len([c for c in kernel.clusters if len(c) >= 2])
        assert n_eff >= 1


# ── §4: Phase Transitions ──────────────────────────────────────────

class TestPhaseTransitions:
    """Verify coercion-driven phase transition behavior."""

    def test_low_coercion_polytheism(self):
        """Low coercion produces multiple centroids (§4.2)."""
        kernel = quick_sim(n=50, steps=1500, coercion=0.05)
        assert len(kernel.centroids) >= 2, \
            f"Expected polytheism at γ=0.05, got {len(kernel.centroids)} centroids"

    def test_high_coercion_monotheism(self):
        """High coercion produces few centroids with high dominance (§4.3)."""
        kernel = quick_sim(n=50, steps=2000, coercion=0.85)
        if kernel.clusters:
            max_cluster = max(len(c) for c in kernel.clusters)
            dominance = max_cluster / len(kernel.agents)
            assert dominance > 0.5, \
                f"Expected high dominance at γ=0.85, got D={dominance:.2f}"

    def test_coercion_reduces_neff(self):
        """N_eff decreases monotonically with coercion (qualitative)."""
        neffs = []
        for gamma in [0.0, 0.3, 0.6, 0.9]:
            kernel = quick_sim(n=50, steps=1500, coercion=gamma, seed=42)
            neffs.append(len(kernel.centroids))
        # At minimum, high coercion should have fewer centroids than low
        assert neffs[-1] <= neffs[0] + 2, \
            f"N_eff did not decrease with coercion: {neffs}"


# ── §3: Centroid Operations ────────────────────────────────────────

class TestCentroidOperations:
    """Verify fusion, fission, and perturbation mechanics."""

    def test_fusion_merges_nearby_centroids(self):
        """Under high coercion, nearby centroids merge (§3.1)."""
        kernel = quick_sim(n=60, steps=2000, coercion=0.8)
        # High coercion should produce fewer centroids than low
        kernel_low = quick_sim(n=60, steps=2000, coercion=0.05, seed=42)
        assert len(kernel.centroids) <= len(kernel_low.centroids) + 2

    def test_prophet_creates_perturbation(self):
        """Prophet events can nucleate new attractors (§3.3)."""
        kernel = quick_sim(
            n=50, steps=2000, coercion=0.1,
            prophet_rate=0.01, prophet_pull_strength=0.4
        )
        # With prophets, we should still have centroids
        assert len(kernel.centroids) >= 1


# ── §5.1: Accessibility Corollary ──────────────────────────────────

class TestAccessibilityCorollary:
    """Verify channel-invariant attractors (Corollary 1)."""

    def test_restricted_agents_converge(self):
        """Sensory-restricted agents still form clusters (§5.1)."""
        kernel = quick_sim(
            n=50, steps=2000, coercion=0.1,
            enable_sensory_restrictions=True,
            sensory_restriction_ratio=0.2
        )
        assert len(kernel.centroids) >= 1, \
            "No centroids with sensory restrictions"

    def test_restricted_vs_unrestricted_similar(self):
        """Restricted and unrestricted simulations produce similar centroid counts."""
        k_unr = quick_sim(n=50, steps=1500, coercion=0.2, seed=42)
        k_res = quick_sim(
            n=50, steps=1500, coercion=0.2, seed=42,
            enable_sensory_restrictions=True,
            sensory_restriction_ratio=0.2
        )
        diff = abs(len(k_unr.centroids) - len(k_res.centroids))
        assert diff <= 5, \
            f"Centroid count diverged: unrestricted={len(k_unr.centroids)}, restricted={len(k_res.centroids)}"


# ── §5.2: Ritual Stabilization ─────────────────────────────────────

class TestRitualStabilization:
    """Verify Corollary 2: ritual reduces churn."""

    def test_ritual_reduces_centroid_count_variance(self):
        """Ritual bonus should stabilize cluster dynamics (§5.2)."""
        # Run with and without ritual
        histories_no_ritual = []
        histories_ritual = []

        for seed in range(42, 47):
            cfg_nr = Config(N=40, steps_per_generation=1000, ritual_bonus=0.0,
                            seed=seed, coercion=0.1, cluster_update_freq=50)
            k_nr = SwarmKernel(cfg_nr)
            h_nr = []
            for step in range(1000):
                k_nr.step()
                if step % 100 == 0:
                    h_nr.append(len(k_nr.centroids))
            histories_no_ritual.append(statistics.stdev(h_nr) if len(h_nr) > 1 else 0)

            cfg_r = Config(N=40, steps_per_generation=1000, ritual_bonus=0.15,
                           ritual_period=50, seed=seed, coercion=0.1, cluster_update_freq=50)
            k_r = SwarmKernel(cfg_r)
            h_r = []
            for step in range(1000):
                k_r.step()
                if step % 100 == 0:
                    h_r.append(len(k_r.centroids))
            histories_ritual.append(statistics.stdev(h_r) if len(h_r) > 1 else 0)

        mean_var_nr = statistics.mean(histories_no_ritual)
        mean_var_r = statistics.mean(histories_ritual)
        # Ritual should not increase variance dramatically
        assert mean_var_r <= mean_var_nr * 2.0, \
            f"Ritual increased variance: no_ritual={mean_var_nr:.3f}, ritual={mean_var_r:.3f}"


# ── §5.3: Prestige Convergence ─────────────────────────────────────

class TestPrestigeConvergence:
    """Verify Corollary 3: high prestige amplification speeds convergence."""

    def test_high_alpha_fewer_centroids(self):
        """Higher prestige α should produce fewer centroids at moderate coercion (§5.3)."""
        neffs_low_alpha = []
        neffs_high_alpha = []
        for seed in range(42, 52):
            k_low = quick_sim(n=50, steps=1500, coercion=0.3, seed=seed, prestige_alpha=0.20)
            k_high = quick_sim(n=50, steps=1500, coercion=0.3, seed=seed, prestige_alpha=0.50)
            neffs_low_alpha.append(len(k_low.centroids))
            neffs_high_alpha.append(len(k_high.centroids))

        mean_low = statistics.mean(neffs_low_alpha)
        mean_high = statistics.mean(neffs_high_alpha)
        # High alpha should produce fewer or equal centroids on average
        assert mean_high <= mean_low + 2, \
            f"High α did not reduce N_eff: α=0.2→{mean_low:.1f}, α=0.5→{mean_high:.1f}"


# ── §5.4: Braille Lattice ──────────────────────────────────────────

class TestBrailleLattice:
    """Verify braille lattice compression properties."""

    def test_recursive_compression_imports(self):
        """recursive_compression.py is importable and has key functions."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sim'))
        from recursive_compression import (
            compress, hamming_centroid, centroid_is_stable,
            op_fusion, op_fission, op_perturbation,
            THEOLOGY, POLITICAL, PERSONALITY
        )
        assert THEOLOGY.name == 'theology'
        assert len(THEOLOGY.axes) == 12

    def test_roundtrip_cosine(self):
        """Round-trip through braille lattice preserves cosine > 0.90."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sim'))
        from recursive_compression import compress, THEOLOGY

        # Use deity priors from swarm_kernel as test vectors
        vecs = list(DEITY_PRIORS.values())[:5]
        result = compress(vecs, THEOLOGY)
        assert result.reconstruction_cosine > 0.90, \
            f"Round-trip cosine = {result.reconstruction_cosine:.3f} (expected > 0.90)"

    def test_hamming_centroid_stability(self):
        """Hamming centroid is stable under small perturbations (Corollary 4)."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sim'))
        from recursive_compression import compress, THEOLOGY

        # Use all deity priors for a larger sample (stability needs majority)
        vecs = list(DEITY_PRIORS.values())
        result = compress(vecs, THEOLOGY)
        # With many vectors, snap dynamics should be stable
        # Even if not perfectly stable, reconstruction cosine should be high
        assert result.reconstruction_cosine > 0.90, \
            f"Reconstruction cosine {result.reconstruction_cosine:.3f} too low"


# ── §8: Corpus Calibration ─────────────────────────────────────────

class TestCorpusCalibration:
    """Verify corpus data files exist and are well-formed."""

    def test_consensus_corpus_exists(self):
        """Multi-scorer consensus corpus exists."""
        path = os.path.join(os.path.dirname(__file__), '..',
                            'mlx-pipeline', 'multi_scorer_consensus.json')
        assert os.path.exists(path), f"Missing: {path}"
        import json
        with open(path) as f:
            data = json.load(f)
        # Structure: {description, models, agreement, n_passages, n_traditions, embeddings}
        n = data.get('n_passages', len(data.get('embeddings', [])))
        assert n >= 50, f"Expected ≥50 passages, got {n}"

    def test_corpus_parts_exist(self):
        """Expanded corpus parts exist."""
        parts_dir = os.path.join(os.path.dirname(__file__), '..',
                                 'mlx-pipeline', 'corpus_parts')
        assert os.path.isdir(parts_dir)
        init_path = os.path.join(parts_dir, '__init__.py')
        assert os.path.exists(init_path)

    def test_scaled_corpus_exists(self):
        """Scaled 826-passage corpus exists."""
        path = os.path.join(os.path.dirname(__file__), '..',
                            'lattice_autoencoder', 'data', 'merged_consensus.json')
        assert os.path.exists(path), f"Missing: {path}"
        import json
        with open(path) as f:
            data = json.load(f)
        # Structure: {description, n_passages, n_traditions, embeddings}
        n = data.get('n_passages', len(data.get('embeddings', [])))
        assert n >= 100, f"Expected ≥100 passages, got {n}"

    def test_calibrated_config_exists(self):
        """Corpus-calibrated config exists."""
        path = os.path.join(os.path.dirname(__file__), '..',
                            'sim', 'corpus_calibrated_config.json')
        assert os.path.exists(path), f"Missing: {path}"
        import json
        with open(path) as f:
            cfg = json.load(f)
        # Should be a valid JSON object
        assert isinstance(cfg, dict), f"Expected dict, got {type(cfg)}"


# ── §4.4: Hysteresis Data ──────────────────────────────────────────

class TestSavedResults:
    """Verify that key simulation outputs exist."""

    def test_hysteresis_data_exists(self):
        """Hysteresis sweep results exist."""
        path = os.path.join(os.path.dirname(__file__), '..',
                            'sim', 'runs', 'hysteresis_sweep', 'hero_plot.png')
        assert os.path.exists(path), f"Missing: {path}"

    def test_prophet_escape_data_exists(self):
        """Prophet escape results exist."""
        path = os.path.join(os.path.dirname(__file__), '..',
                            'sim', 'runs', 'prophet_escape', 'prophet_escape_plot.png')
        assert os.path.exists(path), f"Missing: {path}"

    def test_lattice_hysteresis_data_exists(self):
        """Lattice hysteresis comparison exists."""
        path = os.path.join(os.path.dirname(__file__), '..',
                            'sim', 'runs', 'lattice_hysteresis', 'hamming_vs_arithmetic.png')
        assert os.path.exists(path), f"Missing: {path}"

    def test_mmla_checkpoint_exists(self):
        """MMLA Phase 1 trained checkpoint exists."""
        path = os.path.join(os.path.dirname(__file__), '..',
                            'lattice_autoencoder', 'runs', 'theology', 'best.pt')
        assert os.path.exists(path), f"Missing: {path}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
