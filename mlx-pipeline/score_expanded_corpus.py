#!/usr/bin/env python3
"""
Score Expanded Corpus on 12 Theological Axes
=============================================
Uses the same LLM scoring pipeline as real_embeddings.py but on the
expanded corpus (~130 passages from ~35 traditions).

Output: mlx-pipeline/expanded_embeddings_results.json
"""

import json
import math
import os
import sys
import time
import httpx
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not API_KEY:
    env_path = Path(__file__).parent.parent / "web" / ".env.local"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                API_KEY = line.split("=", 1)[1].strip()

API_URL = "https://openrouter.ai/api/v1/chat/completions"
SCORER_MODEL = "anthropic/claude-sonnet-4"

AXES = [
    "authority", "transcendence", "care", "justice", "wisdom", "power",
    "fertility", "war", "death", "creation", "nature", "order"
]

OUTPUT_FILE = Path(__file__).parent / "expanded_embeddings_results.json"
CHECKPOINT_FILE = Path(__file__).parent / "expanded_scoring_checkpoint.json"


# ─── Import corpus ────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent))
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
            resp = httpx.post(API_URL, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
    return "[ERROR]"


def score_passage(passage: dict) -> dict:
    """Score a religious passage on all 12 theological axes."""
    prompt = f"""You are a computational theology engine. Score the following religious text on exactly 12 theological axes. Each score must be a float between 0.0 and 1.0.

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

    response = call_llm(SCORER_MODEL, [{"role": "user", "content": prompt}], max_tokens=300)

    try:
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        scores = json.loads(response)
        vec = {}
        for axis in AXES:
            val = float(scores.get(axis, 0.0))
            vec[axis] = max(0.0, min(1.0, val))
        return vec
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  Parse error: {e}")
        print(f"  Raw: {response[:200]}")
        return None


def normalize(vec: dict) -> dict:
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1
    return {a: vec[a] / norm for a in AXES}


def main():
    if not API_KEY:
        print("ERROR: No OPENROUTER_API_KEY found")
        print("Set OPENROUTER_API_KEY env var or add to web/.env.local")
        return

    corpus = EXPANDED_CORPUS
    print("=" * 70)
    print("EXPANDED CORPUS SCORING")
    print(f"  {len(corpus)} passages from {len(set(p['tradition'] for p in corpus))} traditions")
    print(f"  Scorer: {SCORER_MODEL}")
    print("=" * 70)

    # Load checkpoint if exists (resume from where we left off)
    embeddings = []
    start_idx = 0
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            checkpoint = json.load(f)
        embeddings = checkpoint.get("embeddings", [])
        start_idx = len(embeddings)
        print(f"\n  Resuming from checkpoint: {start_idx}/{len(corpus)} already scored")

    # Tradition summary
    traditions = {}
    for p in corpus:
        traditions[p["tradition"]] = traditions.get(p["tradition"], 0) + 1
    print(f"\n  Traditions ({len(traditions)}):")
    for t, count in sorted(traditions.items()):
        marker = "✓" if any(e["tradition"] == t for e in embeddings) else " "
        print(f"    {marker} {t}: {count} passages")

    # Score remaining passages
    print(f"\n--- Scoring passages {start_idx+1} to {len(corpus)} ---\n")

    for i in range(start_idx, len(corpus)):
        passage = corpus[i]
        print(f"  [{i+1}/{len(corpus)}] {passage['tradition']}: {passage['source'][:50]}")

        vec = score_passage(passage)
        if vec is None:
            print(f"    FAILED — retrying once...")
            time.sleep(3)
            vec = score_passage(passage)
        if vec is None:
            print(f"    FAILED again — using zeros")
            vec = {axis: 0.0 for axis in AXES}

        vec_norm = normalize(vec)
        embeddings.append({
            "tradition": passage["tradition"],
            "source": passage["source"],
            "expected_cluster": passage["expected_cluster"],
            "raw_scores": vec,
            "normalized": vec_norm,
        })

        # Checkpoint every 10 passages
        if (i + 1) % 10 == 0:
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump({"embeddings": embeddings}, f, indent=2)
            print(f"    [checkpoint saved: {len(embeddings)} scored]")

        # Rate limiting — 1.5s between calls
        time.sleep(1.5)

    # Save final results
    print(f"\n  All {len(embeddings)} passages scored.")

    # Summary by tradition
    print("\n--- Tradition Summary ---\n")
    tradition_data = {}
    for e in embeddings:
        t = e["tradition"]
        if t not in tradition_data:
            tradition_data[t] = []
        tradition_data[t].append(e)

    for t in sorted(tradition_data.keys()):
        entries = tradition_data[t]
        n = len(entries)
        # Compute centroid
        centroid = {a: 0.0 for a in AXES}
        for e in entries:
            for a in AXES:
                centroid[a] += e["normalized"][a]
        centroid = {a: centroid[a] / n for a in AXES}
        # Top 3 axes
        top3 = sorted(AXES, key=lambda a: centroid[a], reverse=True)[:3]
        top3_str = ", ".join(f"{a}={centroid[a]:.2f}" for a in top3)
        print(f"  {t:25s} ({n:2d} passages) top: {top3_str}")

    # Save
    results = {
        "model": SCORER_MODEL,
        "n_passages": len(embeddings),
        "n_traditions": len(tradition_data),
        "traditions": list(sorted(tradition_data.keys())),
        "embeddings": embeddings,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {OUTPUT_FILE}")

    # Clean up checkpoint
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print(f"  Checkpoint cleaned up")

    print("\nDone!")


if __name__ == "__main__":
    main()
