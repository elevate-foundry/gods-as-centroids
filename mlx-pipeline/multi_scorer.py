#!/usr/bin/env python3
"""
Multi-LLM Scorer for Expanded Corpus
=====================================
Scores all 126 passages using multiple LLMs to establish inter-scorer
agreement. Each model scores independently with the same prompt at
temperature=0.

Models:
  1. anthropic/claude-sonnet-4  (already done — loaded from existing results)
  2. openai/gpt-4o
  3. google/gemini-2.0-flash-001
  4. meta-llama/llama-3.3-70b-instruct

Output per model: mlx-pipeline/scores_{model_slug}.json
Final:            mlx-pipeline/multi_scorer_consensus.json
"""

import json
import math
import os
import sys
import time
import httpx
from pathlib import Path
from collections import defaultdict

# ─── Config ───────────────────────────────────────────────────────────

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not API_KEY:
    env_path = Path(__file__).parent.parent / "web" / ".env.local"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                API_KEY = line.split("=", 1)[1].strip()

API_URL = "https://openrouter.ai/api/v1/chat/completions"

AXES = [
    "authority", "transcendence", "care", "justice", "wisdom", "power",
    "fertility", "war", "death", "creation", "nature", "order"
]

MODELS = [
    ("openai/gpt-4o", "gpt4o"),
    ("google/gemini-2.0-flash-001", "gemini_flash"),
    ("meta-llama/llama-3.3-70b-instruct", "llama70b"),
]

BASE_DIR = Path(__file__).parent

# ─── Import corpus ────────────────────────────────────────────────────

sys.path.insert(0, str(BASE_DIR))
from corpus_parts import EXPANDED_CORPUS


# ─── LLM Scoring ─────────────────────────────────────────────────────

def call_llm(model: str, messages: list, max_tokens: int = 800) -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/elevate-foundry/gods-as-centroids",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    for attempt in range(3):
        try:
            resp = httpx.post(API_URL, json=payload, headers=headers, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"    API error (attempt {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
    return "[ERROR]"


def make_prompt(passage: dict) -> str:
    return f"""You are a computational theology engine. Score the following religious text on exactly 12 theological axes. Each score must be a float between 0.0 and 1.0.

TEXT: "{passage['text']}"
SOURCE: {passage['source']} ({passage['tradition']})

Score each axis based on how strongly this text expresses that concept:

1. authority — Divine command, sovereignty, hierarchy, obedience demanded
2. transcendence — Beyond the physical, otherworldly, metaphysical abstraction
3. care — Compassion, nurturing, mercy, love, protection of the vulnerable
4. justice — Moral law, punishment, reward, cosmic fairness, righteousness
5. wisdom — Knowledge, insight, understanding, enlightenment, truth-seeking
6. power — Raw divine force, omnipotence, cosmic might, dominion
7. fertility — Life-giving, abundance, reproduction, growth, prosperity
8. war — Conflict, struggle, martial virtue, conquest, destruction of enemies
9. death — Mortality, afterlife, underworld, destruction, endings
10. creation — Cosmogony, making, origination, bringing into being
11. nature — Earth, elements, animals, seasons, natural world, ecology
12. order — Cosmic structure, law, dharma, harmony, regularity, ritual

Respond with ONLY a JSON object mapping each axis name to its score. Example:
{{"authority": 0.8, "transcendence": 0.6, ...}}

Be precise. A score of 0.0 means the text has zero expression of that concept. A score of 1.0 means the text is maximally about that concept."""


def parse_scores(response: str) -> dict:
    """Parse LLM response into axis scores dict."""
    try:
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        # Some models wrap in extra text — find the JSON object
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            response = response[start:end]
        scores = json.loads(response)
        vec = {}
        for axis in AXES:
            val = float(scores.get(axis, 0.0))
            vec[axis] = max(0.0, min(1.0, val))
        return vec
    except (json.JSONDecodeError, ValueError) as e:
        return None


def normalize(vec: dict) -> dict:
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1
    return {a: vec[a] / norm for a in AXES}


# ─── Score one model ─────────────────────────────────────────────────

def score_model(model_id: str, slug: str):
    """Score all passages with a single model. Checkpoint/resume supported."""
    output_file = BASE_DIR / f"scores_{slug}.json"
    checkpoint_file = BASE_DIR / f"checkpoint_{slug}.json"

    # Already complete?
    if output_file.exists():
        with open(output_file) as f:
            data = json.load(f)
        if len(data.get("embeddings", [])) >= len(EXPANDED_CORPUS):
            print(f"  {slug}: already complete ({len(data['embeddings'])} passages)")
            return data

    # Resume from checkpoint
    embeddings = []
    start_idx = 0
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
        embeddings = checkpoint.get("embeddings", [])
        start_idx = len(embeddings)
        print(f"  {slug}: resuming from checkpoint ({start_idx}/{len(EXPANDED_CORPUS)})")

    corpus = EXPANDED_CORPUS
    print(f"  {slug}: scoring passages {start_idx+1} to {len(corpus)}")

    for i in range(start_idx, len(corpus)):
        passage = corpus[i]
        tradition = passage["tradition"]
        source = passage["source"][:45]
        print(f"    [{i+1}/{len(corpus)}] {tradition}: {source}")

        prompt = make_prompt(passage)
        response = call_llm(model_id, [{"role": "user", "content": prompt}], max_tokens=300)
        vec = parse_scores(response)

        if vec is None:
            print(f"      FAILED — retrying...")
            time.sleep(3)
            response = call_llm(model_id, [{"role": "user", "content": prompt}], max_tokens=300)
            vec = parse_scores(response)

        if vec is None:
            print(f"      FAILED again — using zeros")
            vec = {axis: 0.0 for axis in AXES}

        vec_norm = normalize(vec)
        embeddings.append({
            "tradition": passage["tradition"],
            "source": passage["source"],
            "expected_cluster": passage["expected_cluster"],
            "raw_scores": vec,
            "normalized": vec_norm,
        })

        # Checkpoint every 10
        if (i + 1) % 10 == 0:
            with open(checkpoint_file, "w") as f:
                json.dump({"embeddings": embeddings}, f, indent=2)
            print(f"      [checkpoint: {len(embeddings)} scored]")

        time.sleep(1.0)

    # Save final
    result = {
        "model": model_id,
        "slug": slug,
        "n_passages": len(embeddings),
        "n_traditions": len(set(e["tradition"] for e in embeddings)),
        "embeddings": embeddings,
    }
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    # Clean checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    print(f"  {slug}: DONE — {len(embeddings)} passages saved to {output_file.name}")
    return result


# ─── Inter-scorer agreement ──────────────────────────────────────────

def compute_agreement(all_scores: dict):
    """
    Compute inter-scorer agreement metrics.
    all_scores: {slug: [list of 126 raw_scores dicts]}
    """
    slugs = sorted(all_scores.keys())
    n_passages = len(next(iter(all_scores.values())))
    n_scorers = len(slugs)

    print(f"\n{'='*70}")
    print(f"INTER-SCORER AGREEMENT ({n_scorers} models × {n_passages} passages × {len(AXES)} axes)")
    print(f"{'='*70}")
    print(f"  Models: {', '.join(slugs)}")

    # 1. Per-axis Pearson correlation between all scorer pairs
    from itertools import combinations

    def pearson(x, y):
        n = len(x)
        mx, my = sum(x)/n, sum(y)/n
        sx = math.sqrt(sum((xi-mx)**2 for xi in x) / n)
        sy = math.sqrt(sum((yi-my)**2 for yi in y) / n)
        if sx == 0 or sy == 0:
            return 0.0
        return sum((xi-mx)*(yi-my) for xi, yi in zip(x, y)) / (n * sx * sy)

    print(f"\n─── Per-Axis Pearson Correlation (mean across all scorer pairs) ───\n")
    axis_correlations = {}
    for axis in AXES:
        pair_corrs = []
        for s1, s2 in combinations(slugs, 2):
            x = [all_scores[s1][i][axis] for i in range(n_passages)]
            y = [all_scores[s2][i][axis] for i in range(n_passages)]
            pair_corrs.append(pearson(x, y))
        mean_corr = sum(pair_corrs) / len(pair_corrs)
        axis_correlations[axis] = mean_corr
        bar = "█" * int(mean_corr * 40)
        print(f"  {axis:15s} r={mean_corr:.3f}  {bar}")

    overall_r = sum(axis_correlations.values()) / len(AXES)
    print(f"\n  Overall mean r = {overall_r:.3f}")

    # 2. Per-passage mean absolute deviation across scorers
    print(f"\n─── Per-Passage Scorer Disagreement (mean abs deviation) ───\n")
    passage_devs = []
    for i in range(n_passages):
        devs = []
        for axis in AXES:
            vals = [all_scores[s][i][axis] for s in slugs]
            mean_val = sum(vals) / len(vals)
            devs.extend(abs(v - mean_val) for v in vals)
        passage_devs.append(sum(devs) / len(devs))

    mean_dev = sum(passage_devs) / len(passage_devs)
    max_dev = max(passage_devs)
    min_dev = min(passage_devs)
    print(f"  Mean absolute deviation: {mean_dev:.4f}")
    print(f"  Min passage deviation:   {min_dev:.4f}")
    print(f"  Max passage deviation:   {max_dev:.4f}")

    # 3. Pairwise model correlations (overall)
    print(f"\n─── Pairwise Model Correlation (all axes flattened) ───\n")
    pairwise = {}
    for s1, s2 in combinations(slugs, 2):
        x = [all_scores[s1][i][a] for i in range(n_passages) for a in AXES]
        y = [all_scores[s2][i][a] for i in range(n_passages) for a in AXES]
        r = pearson(x, y)
        pairwise[(s1, s2)] = r
        print(f"  {s1:15s} × {s2:15s}  r={r:.3f}")

    mean_pairwise = sum(pairwise.values()) / len(pairwise)
    print(f"\n  Mean pairwise r = {mean_pairwise:.3f}")

    # 4. Krippendorff's alpha (simplified ordinal)
    # Using the ratio-based approximation: α ≈ 1 - (observed disagreement / expected disagreement)
    print(f"\n─── Krippendorff's Alpha (ratio approximation) ───\n")

    # Flatten all scores into a matrix: (n_scorers × n_items) for each axis
    all_vals = []
    for axis in AXES:
        for i in range(n_passages):
            for s in slugs:
                all_vals.append(all_scores[s][i][axis])

    grand_mean = sum(all_vals) / len(all_vals)
    total_var = sum((v - grand_mean)**2 for v in all_vals) / len(all_vals)

    # Within-unit variance (disagreement among scorers for same passage-axis)
    within_var = 0
    n_units = 0
    for axis in AXES:
        for i in range(n_passages):
            vals = [all_scores[s][i][axis] for s in slugs]
            unit_mean = sum(vals) / len(vals)
            within_var += sum((v - unit_mean)**2 for v in vals) / len(vals)
            n_units += 1
    within_var /= n_units

    alpha = 1 - (within_var / total_var) if total_var > 0 else 0
    print(f"  Krippendorff's α ≈ {alpha:.3f}")
    if alpha > 0.8:
        print(f"  Interpretation: GOOD agreement (α > 0.8)")
    elif alpha > 0.667:
        print(f"  Interpretation: ACCEPTABLE agreement (0.667 < α < 0.8)")
    else:
        print(f"  Interpretation: POOR agreement (α < 0.667)")

    # 5. Consensus scores (mean across all scorers)
    print(f"\n─── Consensus Scores (mean of {n_scorers} models) ───\n")
    consensus_embeddings = []
    corpus = EXPANDED_CORPUS
    for i in range(n_passages):
        consensus_raw = {}
        for axis in AXES:
            vals = [all_scores[s][i][axis] for s in slugs]
            consensus_raw[axis] = sum(vals) / len(vals)
        consensus_norm = normalize(consensus_raw)
        consensus_embeddings.append({
            "tradition": corpus[i]["tradition"],
            "source": corpus[i]["source"],
            "expected_cluster": corpus[i]["expected_cluster"],
            "raw_scores": {a: round(v, 4) for a, v in consensus_raw.items()},
            "normalized": {a: round(v, 4) for a, v in consensus_norm.items()},
            "scorer_variance": {a: round(
                sum((all_scores[s][i][a] - consensus_raw[a])**2 for s in slugs) / n_scorers, 6
            ) for a in AXES},
        })

    # Tradition centroids from consensus
    groups = defaultdict(list)
    for e in consensus_embeddings:
        groups[e["tradition"]].append(e["normalized"])

    for tradition in sorted(groups.keys()):
        vecs = groups[tradition]
        centroid = {a: sum(v[a] for v in vecs) / len(vecs) for a in AXES}
        top3 = sorted(AXES, key=lambda a: centroid[a], reverse=True)[:3]
        top3_str = ", ".join(f"{a}={centroid[a]:.2f}" for a in top3)
        print(f"  {tradition:25s} ({len(vecs):2d}) {top3_str}")

    return {
        "n_scorers": n_scorers,
        "n_passages": n_passages,
        "models": slugs,
        "overall_mean_r": round(overall_r, 4),
        "mean_pairwise_r": round(mean_pairwise, 4),
        "krippendorff_alpha": round(alpha, 4),
        "mean_abs_deviation": round(mean_dev, 4),
        "per_axis_correlation": {a: round(v, 4) for a, v in axis_correlations.items()},
        "pairwise_correlations": {f"{s1}_x_{s2}": round(r, 4) for (s1, s2), r in pairwise.items()},
        "consensus_embeddings": consensus_embeddings,
    }


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    if not API_KEY:
        print("ERROR: No OPENROUTER_API_KEY found")
        print("Set OPENROUTER_API_KEY env var or add to web/.env.local")
        return

    print("=" * 70)
    print("MULTI-LLM SCORER — INTER-SCORER AGREEMENT")
    print(f"  {len(EXPANDED_CORPUS)} passages × {len(MODELS) + 1} models")
    print("=" * 70)

    # Load existing Claude scores
    claude_path = BASE_DIR / "expanded_embeddings_results.json"
    if not claude_path.exists():
        print("ERROR: Run score_expanded_corpus.py first (Claude baseline)")
        return

    with open(claude_path) as f:
        claude_data = json.load(f)
    print(f"\n  Claude Sonnet 4: loaded {len(claude_data['embeddings'])} passages")

    # Score with each additional model
    all_results = {"claude": claude_data}

    for model_id, slug in MODELS:
        print(f"\n{'─'*70}")
        print(f"  Scoring with {model_id} ({slug})")
        print(f"{'─'*70}")
        result = score_model(model_id, slug)
        all_results[slug] = result

    # Collect raw scores from all models
    all_scores = {}
    for slug, result in all_results.items():
        all_scores[slug] = [e["raw_scores"] for e in result["embeddings"]]

    # Compute agreement
    agreement = compute_agreement(all_scores)

    # Save consensus results
    consensus_output = {
        "description": "Multi-LLM consensus scores for expanded religious corpus",
        "models": {slug: result.get("model", slug) for slug, result in all_results.items()},
        "agreement": {k: v for k, v in agreement.items() if k != "consensus_embeddings"},
        "n_passages": agreement["n_passages"],
        "n_traditions": len(set(e["tradition"] for e in agreement["consensus_embeddings"])),
        "embeddings": agreement["consensus_embeddings"],
    }

    consensus_path = BASE_DIR / "multi_scorer_consensus.json"
    with open(consensus_path, "w") as f:
        json.dump(consensus_output, f, indent=2)
    print(f"\n  Consensus results saved to {consensus_path.name}")

    print("\nDone!")


if __name__ == "__main__":
    main()
