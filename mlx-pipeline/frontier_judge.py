#!/usr/bin/env python3
"""
Frontier LLM as Semantic Judge — Braille Bottleneck Validation

Experiment:
  1. Generate theological profiles from FULL continuous centroids via frontier LLM
  2. Generate theological profiles from BRAILLE-COMPRESSED centroids via frontier LLM
  3. Blind judge (different model) rates whether pairs describe the same deity
  4. Measure structural equivalence rate

This uses frontier LLMs as MEASUREMENT INSTRUMENTS, not as the system under test.
The publishable claim: "A frontier LLM acting as a blind semantic judge cannot
reliably distinguish theology generated from continuous vs 72-bit compressed centroids."
"""

import json
import math
import os
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

# Generator model and judge model should differ
GENERATOR_MODEL = "anthropic/claude-sonnet-4"
JUDGE_MODEL = "openai/gpt-4o-mini"

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

DEITY_PRIORS = {
    "Zeus":    {"authority": 0.9, "transcendence": 0.7, "care": 0.3, "justice": 0.6, "wisdom": 0.4, "power": 0.95, "fertility": 0.3, "war": 0.5, "death": 0.2, "creation": 0.4, "nature": 0.6, "order": 0.7},
    "Yahweh":  {"authority": 0.95, "transcendence": 0.95, "care": 0.7, "justice": 0.9, "wisdom": 0.8, "power": 0.9, "fertility": 0.2, "war": 0.3, "death": 0.3, "creation": 0.9, "nature": 0.3, "order": 0.9},
    "Vishnu":  {"authority": 0.6, "transcendence": 0.9, "care": 0.9, "justice": 0.7, "wisdom": 0.8, "power": 0.7, "fertility": 0.5, "war": 0.2, "death": 0.3, "creation": 0.8, "nature": 0.7, "order": 0.8},
    "Odin":    {"authority": 0.7, "transcendence": 0.5, "care": 0.2, "justice": 0.4, "wisdom": 0.95, "power": 0.6, "fertility": 0.1, "war": 0.8, "death": 0.7, "creation": 0.5, "nature": 0.4, "order": 0.3},
    "Isis":    {"authority": 0.4, "transcendence": 0.6, "care": 0.95, "justice": 0.5, "wisdom": 0.7, "power": 0.5, "fertility": 0.9, "war": 0.1, "death": 0.4, "creation": 0.6, "nature": 0.8, "order": 0.5},
    "Mars":    {"authority": 0.6, "transcendence": 0.2, "care": 0.1, "justice": 0.3, "wisdom": 0.2, "power": 0.8, "fertility": 0.3, "war": 0.95, "death": 0.6, "creation": 0.1, "nature": 0.3, "order": 0.4},
    "Kali":    {"authority": 0.5, "transcendence": 0.7, "care": 0.3, "justice": 0.6, "wisdom": 0.4, "power": 0.7, "fertility": 0.3, "war": 0.6, "death": 0.9, "creation": 0.5, "nature": 0.4, "order": 0.3},
    "Gaia":    {"authority": 0.2, "transcendence": 0.4, "care": 0.8, "justice": 0.3, "wisdom": 0.4, "power": 0.3, "fertility": 0.9, "war": 0.1, "death": 0.3, "creation": 0.7, "nature": 0.95, "order": 0.4},
}


# ─── Braille Encoding (matches bottleneck_arch.py logic) ─────────────

def normalize_vec(vec: dict) -> dict:
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1
    return {a: vec[a] / norm for a in AXES}


def encode_to_braille(vec: dict) -> list:
    """Encode 12D vector as 72 bits (6 per axis)."""
    bits = []
    for axis in AXES:
        v = vec[axis]
        opp = vec[POLARITY_PAIRS[axis]]
        # Polarity
        bits.append(1 if v > 0.3 else 0)
        bits.append(1 if opp > v + 0.1 else 0)
        bits.append(1 if (v > 0.3 and opp > 0.3 and abs(v - opp) < 0.15) else 0)
        # Intensity
        intensity = min(3, int(v * 4))
        bits.append(1 if (intensity & 2) else 0)
        bits.append(1 if (intensity & 1) else 0)
        # Rigidity
        bits.append(1 if v > 0.7 else 0)
    return bits


def decode_from_braille(bits: list) -> dict:
    """Decode 72 bits back to approximate continuous vector."""
    vec = {}
    for i, axis in enumerate(AXES):
        b = bits[i*6:(i+1)*6]
        pos, neg, tens, d4, d5, rig = b
        intensity = (2 if d4 else 0) + (1 if d5 else 0)
        value = (intensity + 0.5) / 4
        if not pos and neg:
            value *= 0.3
        if tens:
            value *= 0.85
        if rig:
            value = max(value, 0.75)
        vec[axis] = value
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1
    for axis in AXES:
        vec[axis] /= norm
    return vec


def braille_unicode(bits: list) -> str:
    """Convert 72 bits to 12 braille unicode characters."""
    chars = []
    offsets = [1, 2, 4, 8, 16, 32]
    for i in range(12):
        cell_bits = bits[i*6:(i+1)*6]
        code = 0x2800
        for j, b in enumerate(cell_bits):
            if b:
                code += offsets[j]
        chars.append(chr(code))
    return "".join(chars)


def cosine_sim(a: dict, b: dict) -> float:
    dot = sum(a[ax] * b[ax] for ax in AXES)
    na = math.sqrt(sum(a[ax]**2 for ax in AXES))
    nb = math.sqrt(sum(b[ax]**2 for ax in AXES))
    return dot / (na * nb + 1e-8)


# ─── LLM Calls ───────────────────────────────────────────────────────

def call_llm(model: str, messages: list, max_tokens: int = 800) -> str:
    """Call OpenRouter API."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/elevate-foundry/gods-as-centroids",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
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
                time.sleep(2 ** attempt)
    return "[ERROR: API call failed]"


def generate_theology(deity_name: str, vec: dict, source_label: str) -> str:
    """Generate a theological profile from a centroid vector."""
    sorted_axes = sorted(AXES, key=lambda a: vec[a], reverse=True)
    dominant = sorted_axes[:3]
    recessive = sorted_axes[-3:]
    axes_str = ", ".join(f"{a}={vec[a]:.3f}" for a in AXES)

    prompt = f"""You are a comparative theology engine. Given a deity centroid vector in a 12-dimensional theological space, generate a concise theological profile.

DEITY NAME: {deity_name}
DOMINANT AXES: {', '.join(dominant)}
RECESSIVE AXES: {', '.join(recessive)}
VECTOR: {axes_str}

Write exactly 3 sentences describing:
1. This deity's core theological character
2. The type of worship or practice it implies
3. Its relationship to other theological traditions

Be specific and grounded in the vector values. Do not use flowery language."""

    return call_llm(GENERATOR_MODEL, [{"role": "user", "content": prompt}], max_tokens=300)


def judge_equivalence(profile_a: str, profile_b: str, deity_name: str) -> dict:
    """Blind judge: are these two profiles describing the same deity?"""
    # Randomize order to avoid position bias
    import random
    if random.random() > 0.5:
        first, second = profile_a, profile_b
        order = "AB"
    else:
        first, second = profile_b, profile_a
        order = "BA"

    prompt = f"""You are a theological analysis judge. Two different systems generated theological profiles of a deity. Your task is to evaluate their structural equivalence.

PROFILE 1:
{first}

PROFILE 2:
{second}

Answer these questions with a JSON object:
1. "same_deity": true/false — Do both profiles describe fundamentally the same type of deity?
2. "structural_similarity": 0.0-1.0 — How structurally similar are the theological descriptions?
3. "key_agreements": list of theological features both profiles share
4. "key_disagreements": list of theological features where they differ
5. "confidence": 0.0-1.0 — How confident are you in your assessment?

Respond with ONLY the JSON object, no other text."""

    response = call_llm(JUDGE_MODEL, [{"role": "user", "content": prompt}], max_tokens=500)

    # Parse JSON from response
    try:
        # Try to extract JSON from response
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        result = json.loads(response)
        result["order"] = order
        return result
    except json.JSONDecodeError:
        return {
            "same_deity": None,
            "structural_similarity": None,
            "key_agreements": [],
            "key_disagreements": [],
            "confidence": 0,
            "order": order,
            "raw_response": response,
        }


# ─── Main Experiment ──────────────────────────────────────────────────

def main():
    if not API_KEY:
        print("ERROR: No OPENROUTER_API_KEY found")
        return

    print("=" * 60)
    print("FRONTIER LLM AS SEMANTIC JUDGE")
    print(f"Generator: {GENERATOR_MODEL}")
    print(f"Judge: {JUDGE_MODEL}")
    print("=" * 60)

    results = []

    for deity_name, raw_vec in DEITY_PRIORS.items():
        print(f"\n{'─' * 40}")
        print(f"Deity: {deity_name}")
        print(f"{'─' * 40}")

        # Normalize
        vec = normalize_vec(raw_vec)

        # Encode to braille and decode back
        bits = encode_to_braille(vec)
        decoded_vec = decode_from_braille(bits)
        sig = braille_unicode(bits)
        recon_cos = cosine_sim(vec, decoded_vec)

        print(f"  Braille: {sig}")
        print(f"  Reconstruction cosine sim: {recon_cos:.4f}")

        # Generate theology from continuous centroid
        print(f"  Generating from continuous centroid ({GENERATOR_MODEL})...")
        profile_continuous = generate_theology(deity_name, vec, "continuous")
        time.sleep(1)  # rate limiting

        # Generate theology from braille-compressed centroid
        print(f"  Generating from braille-compressed centroid ({GENERATOR_MODEL})...")
        profile_braille = generate_theology(deity_name, decoded_vec, "braille-compressed")
        time.sleep(1)

        # Blind judge
        print(f"  Judging equivalence ({JUDGE_MODEL})...")
        judgment = judge_equivalence(profile_continuous, profile_braille, deity_name)
        time.sleep(1)

        result = {
            "deity": deity_name,
            "braille_signature": sig,
            "reconstruction_cosine_sim": recon_cos,
            "profile_continuous": profile_continuous,
            "profile_braille": profile_braille,
            "judgment": judgment,
        }
        results.append(result)

        # Print result
        same = judgment.get("same_deity")
        sim = judgment.get("structural_similarity")
        conf = judgment.get("confidence")
        print(f"  Same deity: {same}")
        print(f"  Structural similarity: {sim}")
        print(f"  Confidence: {conf}")

    # ─── Summary ──────────────────────────────────────────────────

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    valid = [r for r in results if r["judgment"].get("same_deity") is not None]
    if valid:
        same_count = sum(1 for r in valid if r["judgment"]["same_deity"])
        equivalence_rate = same_count / len(valid)

        sims = [r["judgment"]["structural_similarity"] for r in valid
                if r["judgment"].get("structural_similarity") is not None]
        mean_sim = sum(sims) / len(sims) if sims else 0

        confs = [r["judgment"]["confidence"] for r in valid
                 if r["judgment"].get("confidence") is not None]
        mean_conf = sum(confs) / len(confs) if confs else 0

        print(f"  Deities tested: {len(valid)}")
        print(f"  Equivalence rate: {same_count}/{len(valid)} = {equivalence_rate:.1%}")
        print(f"  Mean structural similarity: {mean_sim:.4f}")
        print(f"  Mean judge confidence: {mean_conf:.4f}")
        print()

        if equivalence_rate >= 0.9:
            print("  ✓ FRONTIER JUDGE CANNOT DISTINGUISH CONTINUOUS FROM COMPRESSED")
            print("    Theological structure is invisible to compression at 72 bits.")
        elif equivalence_rate >= 0.7:
            print("  ~ PARTIAL EQUIVALENCE — most deities survive compression")
        else:
            print("  ✗ SIGNIFICANT DIVERGENCE — bottleneck introduces detectable changes")
    else:
        print("  No valid judgments received.")

    # ─── Save Results ─────────────────────────────────────────────

    output = {
        "experiment": "frontier_llm_semantic_judge",
        "generator_model": GENERATOR_MODEL,
        "judge_model": JUDGE_MODEL,
        "results": results,
        "summary": {
            "deities_tested": len(valid) if valid else 0,
            "equivalence_rate": equivalence_rate if valid else 0,
            "mean_structural_similarity": mean_sim if valid else 0,
            "mean_judge_confidence": mean_conf if valid else 0,
        } if valid else {},
    }

    out_path = Path(__file__).parent / "frontier_judge_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    # ─── Markdown Report ──────────────────────────────────────────

    report_lines = [
        "# Frontier LLM Semantic Judge — Braille Bottleneck Validation",
        "",
        "## Experiment Design",
        "",
        f"**Generator:** `{GENERATOR_MODEL}` — generates theological profiles from centroid vectors",
        f"**Judge:** `{JUDGE_MODEL}` — blind evaluation of structural equivalence",
        "",
        "For each deity:",
        "1. Generate theology from the **full continuous centroid**",
        "2. Generate theology from the **braille-compressed centroid** (72-bit → decoded)",
        "3. Blind judge evaluates whether both describe the same deity",
        "",
        "---",
        "",
        "## Results",
        "",
        "| Deity | Braille | Recon Cos | Same Deity | Struct Sim | Confidence |",
        "|-------|---------|----------|-----------|-----------|-----------|",
    ]

    for r in results:
        j = r["judgment"]
        same = "✓" if j.get("same_deity") else ("✗" if j.get("same_deity") is False else "?")
        sim = f"{j['structural_similarity']:.2f}" if j.get("structural_similarity") is not None else "?"
        conf = f"{j['confidence']:.2f}" if j.get("confidence") is not None else "?"
        report_lines.append(
            f"| {r['deity']} | `{r['braille_signature']}` | {r['reconstruction_cosine_sim']:.4f} | "
            f"{same} | {sim} | {conf} |"
        )

    if valid:
        report_lines.extend([
            "",
            f"**Equivalence rate:** {same_count}/{len(valid)} = {equivalence_rate:.1%}",
            f"**Mean structural similarity:** {mean_sim:.4f}",
            f"**Mean judge confidence:** {mean_conf:.4f}",
            "",
        ])

    # Detailed profiles
    for r in results:
        report_lines.extend([
            f"### {r['deity']}",
            "",
            f"**Braille:** `{r['braille_signature']}` | Reconstruction cosine: {r['reconstruction_cosine_sim']:.4f}",
            "",
            "**Continuous profile:**",
            f"> {r['profile_continuous']}",
            "",
            "**Braille-compressed profile:**",
            f"> {r['profile_braille']}",
            "",
            f"**Judge verdict:** same_deity={r['judgment'].get('same_deity')}, "
            f"similarity={r['judgment'].get('structural_similarity')}, "
            f"confidence={r['judgment'].get('confidence')}",
            "",
        ])

    report_lines.extend([
        "---",
        "",
        "*Generated by the Frontier LLM Semantic Judge — Gods as Centroids*",
    ])

    report_path = Path(__file__).parent / "frontier_judge_results.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Report saved to {report_path}")


if __name__ == "__main__":
    main()
