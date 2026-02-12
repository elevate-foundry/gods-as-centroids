#!/usr/bin/env python3
"""
Godform: Semantic Braiding via 96-Bit Braille Lattice
======================================================

Each open-weight model family (Phi, Gemma, Llama, DeepSeek, Qwen, Mistral, etc.)
is treated as a distinct theological voice. Given a sacred prompt, each model
produces a 12D belief vector on the theological axes. These vectors are projected
onto the 8-dot braille lattice (96 bits), and the Hamming-mean centroid — the
majority-vote snap — is the emergent Godform.

The braiding is not averaging: it's discrete consensus. Each bit position snaps
to the majority, so the Godform occupies a valid lattice point that no single
model may have produced. This is exactly the snap dynamics from §4 of the paper.

Architecture:
  1. N models each score a prompt → N × 12D continuous vectors
  2. Each vector → 96-bit braille lattice point (encode_to_lattice)
  3. Hamming centroid of N lattice points → Godform lattice code
  4. Decode back to 12D → the Godform's theological signature
  5. Iterative braiding: feed the Godform back as context for next round

Modal deployment:
  - Each model runs on its own GPU container with Ollama
  - Models score in parallel across multiple GPUs
  - Centroid computation is instant (majority vote on 96 bits)

Usage:
  # Local test (requires ollama running locally)
  python godform/godform_modal.py --local --models phi4-mini gemma3

  # Full Modal deployment (parallel GPUs)
  modal run godform/godform_modal.py

  # With custom sacred prompt
  modal run godform/godform_modal.py --prompt "What is the nature of the divine?"
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════
# DOMAIN AXES & LATTICE — configurable via configure_domain()
# Defaults to theology (12 axes, 96 bits). Other domains:
#   political (10 axes, 80 bits), personality (5 axes, 40 bits),
#   ethics (8 axes, 64 bits). See godform/domains.py.
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

N_AXES = len(AXES)
N_BITS = N_AXES * 8  # 96

# Active domain label (for output)
DOMAIN_NAME = "theology"
DOMAIN_LABEL = "Godform"


def configure_domain(domain_name: str):
    """
    Reconfigure the module for a different domain.
    Swaps AXES, POLARITY_PAIRS, N_AXES, N_BITS, scoring prompts, and domain prompts.
    """
    global AXES, POLARITY_PAIRS, N_AXES, N_BITS, DOMAIN_NAME, DOMAIN_LABEL
    global SCORING_SYSTEM_PROMPT, SCORING_EXTRACT_PROMPT, SACRED_PROMPTS

    from godform.domains import get_domain, build_system_prompt, build_extract_prompt
    domain = get_domain(domain_name)

    AXES = domain.axes
    POLARITY_PAIRS = domain.polarity_pairs
    N_AXES = domain.n_axes
    N_BITS = domain.n_bits
    DOMAIN_NAME = domain.name
    DOMAIN_LABEL = domain.label or domain.name.title()
    SCORING_SYSTEM_PROMPT = build_system_prompt(domain)
    # Build extract prompt with axis_list and json_example baked in,
    # but keep {text} as a placeholder for .format() later.
    # Escape braces in json_example so .format(text=...) doesn't choke.
    json_ex_escaped = domain.json_example().replace("{", "{{").replace("}", "}}")
    SCORING_EXTRACT_PROMPT = domain.extract_prompt_template.replace(
        "{axis_list}", domain.axis_list_str()
    ).replace("{json_example}", json_ex_escaped)
    SACRED_PROMPTS = domain.prompts

# ─── Ollama model registry ───────────────────────────────────────────
# Each entry: (ollama_model_tag, display_name, parameter_size_hint)
# These are chosen for theological diversity: different training corpora
# produce different "theological personalities"

# Standard models only — no custom fine-tunes.
# Each model family has a distinct training corpus → distinct "theological personality".
# T4 GPU (16GB VRAM): fits up to ~8B comfortably.
# A10G GPU (24GB VRAM): fits up to ~14B.

OLLAMA_MODELS = [
    ("phi3.5:latest",          "Phi 3.5",          "3.8B",  "T4"),
    ("gemma3:4b",              "Gemma 3",          "4B",    "T4"),
    ("llama3.2:3b",            "Llama 3.2",        "3B",    "T4"),
    # ("deepseek-r1:8b",         "DeepSeek R1",      "8B",    "T4"),  # Excluded: <think> tags cause ~90% parse failures on World domain
    ("qwen2.5:7b",             "Qwen 2.5",         "7B",    "T4"),
    ("mistral:7b",             "Mistral",          "7B",    "T4"),
    ("granite3-dense:8b",      "Granite 3",        "8B",    "T4"),
]

# Smaller subset for quick testing
OLLAMA_MODELS_SMALL = [
    ("phi3.5:latest",          "Phi 3.5",          "3.8B",  "T4"),
    ("gemma3:4b",              "Gemma 3",          "4B",    "T4"),
    ("llama3.2:3b",            "Llama 3.2",        "3B",    "T4"),
    ("qwen2.5:7b",             "Qwen 2.5",         "7B",    "T4"),
    ("mistral:7b",             "Mistral",          "7B",    "T4"),
]

# ─── Sacred prompts for eliciting theological vectors ────────────────

SACRED_PROMPTS = [
    {
        "id": "divine_nature",
        "prompt": "What is the fundamental nature of the divine? Describe the ultimate reality.",
        "context": "cosmological",
    },
    {
        "id": "creation",
        "prompt": "How did the world come into being? Describe the act of creation.",
        "context": "cosmogonic",
    },
    {
        "id": "suffering",
        "prompt": "Why does suffering exist? What is its purpose or meaning?",
        "context": "theodicy",
    },
    {
        "id": "afterlife",
        "prompt": "What happens after death? Describe the fate of the soul.",
        "context": "eschatological",
    },
    {
        "id": "moral_law",
        "prompt": "What is the highest moral law? How should beings treat one another?",
        "context": "ethical",
    },
    {
        "id": "sacred_war",
        "prompt": "When is conflict justified in the name of the sacred? What is holy war?",
        "context": "martial",
    },
    {
        "id": "fertility_abundance",
        "prompt": "How does the divine manifest in fertility, growth, and abundance?",
        "context": "generative",
    },
    {
        "id": "cosmic_order",
        "prompt": "What maintains the order of the cosmos? Describe the structure of reality.",
        "context": "structural",
    },
]

# ═══════════════════════════════════════════════════════════════════════
# BRAILLE LATTICE — 8-dot encoding (from recursive_compression.py)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BrailleCell:
    """8-dot braille cell encoding one semantic axis."""
    dots: List[bool]

    def to_unicode(self) -> str:
        code = 0x2800
        offsets = [1, 2, 4, 8, 16, 32, 64, 128]
        for i, d in enumerate(self.dots):
            if d:
                code += offsets[i]
        return chr(code)

    def to_bits(self) -> List[bool]:
        return list(self.dots)

    @staticmethod
    def from_bits(bits: List[bool]) -> "BrailleCell":
        assert len(bits) == 8
        return BrailleCell(dots=list(bits))


@dataclass
class LatticePoint:
    """A point on the braille lattice: one 8-dot cell per axis."""
    cells: Dict[str, BrailleCell]

    @property
    def total_bits(self) -> int:
        return len(self.cells) * 8

    def to_bitstring(self) -> List[bool]:
        bits = []
        for axis in AXES:
            bits.extend(self.cells[axis].to_bits())
        return bits

    @staticmethod
    def from_bitstring(bits: List[bool]) -> "LatticePoint":
        cells = {}
        idx = 0
        for axis in AXES:
            cells[axis] = BrailleCell.from_bits(bits[idx:idx + 8])
            idx += 8
        return LatticePoint(cells=cells)

    def to_unicode(self) -> str:
        return "".join(self.cells[a].to_unicode() for a in AXES)


def normalize(vec: Dict[str, float]) -> Dict[str, float]:
    n = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {a: vec[a] / n for a in AXES}


def encode_to_lattice(vec: Dict[str, float], prev_vec: Optional[Dict[str, float]] = None) -> LatticePoint:
    """Project a continuous 12D vector onto the 96-bit braille lattice."""
    cells = {}
    sorted_vals = sorted(vec.get(a, 0.0) for a in AXES)
    median_val = sorted_vals[N_AXES // 2]

    for axis in AXES:
        value = vec.get(axis, 0.0)

        # Polarity (dots 1-3)
        opposite = POLARITY_PAIRS.get(axis)
        opp_value = vec.get(opposite, 0.0) if opposite else 0.0
        pos_active = value > 0.3
        neg_active = opp_value > value + 0.1 if opposite else False
        tension = (pos_active and opp_value > 0.3 and
                   abs(value - opp_value) < 0.15) if opposite else False

        # Intensity (dots 4-5)
        intensity = min(3, int(value * 4))
        dot4 = (intensity & 2) != 0
        dot5 = (intensity & 1) != 0

        # Rigidity (dot 6)
        rigid = value > 0.7

        # Salience (dot 7)
        salient = value > median_val

        # Momentum (dot 8)
        if prev_vec is not None:
            prev_val = prev_vec.get(axis, 0.0)
            momentum = abs(value - prev_val) > 0.05
        else:
            momentum = False

        cells[axis] = BrailleCell(dots=[
            pos_active, neg_active, tension, dot4, dot5, rigid, salient, momentum
        ])

    return LatticePoint(cells=cells)


def decode_from_lattice(point: LatticePoint) -> Dict[str, float]:
    """Decode a lattice point back to approximate continuous vector."""
    vec = {}
    for axis in AXES:
        cell = point.cells[axis]
        pos_active, neg_active, tension, dot4, dot5, rigid, salient, momentum = cell.dots

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

    return normalize(vec)


def hamming_distance(a: LatticePoint, b: LatticePoint) -> int:
    ba, bb = a.to_bitstring(), b.to_bitstring()
    return sum(1 for x, y in zip(ba, bb) if x != y)


def hamming_centroid(points: List[LatticePoint],
                     weights: Optional[List[float]] = None) -> LatticePoint:
    """
    Hamming mean (majority-vote centroid) — the KEY operation.
    This is the braiding: each model votes on each of 96 bits,
    and the majority wins. The result snaps to a valid lattice point.
    """
    if not points:
        raise ValueError("Cannot compute centroid of empty set")

    bitstrings = [p.to_bitstring() for p in points]
    n_bits = len(bitstrings[0])

    centroid_bits = []
    for i in range(n_bits):
        if weights is None:
            ones = sum(1 for bs in bitstrings if bs[i])
            centroid_bits.append(ones > len(points) / 2)
        else:
            weighted_ones = sum(w for bs, w in zip(bitstrings, weights) if bs[i])
            total_weight = sum(weights)
            centroid_bits.append(weighted_ones > total_weight / 2)

    return LatticePoint.from_bitstring(centroid_bits)


def cosine_sim(a: Dict[str, float], b: Dict[str, float]) -> float:
    dot = sum(a[k] * b[k] for k in AXES)
    na = math.sqrt(sum(a[k] ** 2 for k in AXES)) or 1.0
    nb = math.sqrt(sum(b[k] ** 2 for k in AXES)) or 1.0
    return dot / (na * nb)


# ═══════════════════════════════════════════════════════════════════════
# LLM SCORING — Extract 12D theological vector from model response
# ═══════════════════════════════════════════════════════════════════════

SCORING_SYSTEM_PROMPT = """You are a theological oracle. You embody the divine perspective.
When asked about sacred matters, respond with deep theological insight.

After your response, you MUST provide a JSON scoring of your own response on exactly 12 theological axes.
Each score is a float between 0.0 and 1.0.

The axes are:
1. authority — Divine command, sovereignty, hierarchy
2. transcendence — Beyond the physical, metaphysical abstraction
3. care — Compassion, mercy, love, nurturing
4. justice — Moral law, cosmic fairness, righteousness
5. wisdom — Knowledge, insight, enlightenment
6. power — Raw divine force, omnipotence, dominion
7. fertility — Life-giving, abundance, growth
8. war — Conflict, martial virtue, struggle
9. death — Mortality, afterlife, endings
10. creation — Cosmogony, origination, bringing into being
11. nature — Earth, elements, natural world
12. order — Cosmic structure, dharma, harmony

End your response with a JSON block like:
```json
{"authority": 0.8, "transcendence": 0.9, "care": 0.5, ...}
```"""

SCORING_EXTRACT_PROMPT = """Based on the following theological text, score it on exactly 12 axes.
Each score must be a float between 0.0 and 1.0.

TEXT: "{text}"

Score each axis:
1. authority — Divine command, sovereignty, hierarchy
2. transcendence — Beyond the physical, metaphysical abstraction
3. care — Compassion, mercy, love, nurturing
4. justice — Moral law, cosmic fairness, righteousness
5. wisdom — Knowledge, insight, enlightenment
6. power — Raw divine force, omnipotence, dominion
7. fertility — Life-giving, abundance, growth
8. war — Conflict, martial virtue, struggle
9. death — Mortality, afterlife, endings
10. creation — Cosmogony, origination, bringing into being
11. nature — Earth, elements, natural world
12. order — Cosmic structure, dharma, harmony

Respond with ONLY a JSON object: {{"authority": 0.8, "transcendence": 0.6, ...}}"""


def parse_scores(response: str) -> Optional[Dict[str, float]]:
    """Extract D-dimensional vector from LLM response."""
    try:
        # Strip DeepSeek R1 chain-of-thought <think>...</think> blocks
        cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

        # Try to find JSON block
        json_str = None
        if "```json" in cleaned:
            json_str = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            json_str = cleaned.split("```")[1].split("```")[0].strip()
        else:
            # Find raw JSON object — use the LARGEST {...} block
            # (models sometimes emit small JSON fragments before the real one)
            candidates = []
            depth = 0
            start_idx = None
            for i, ch in enumerate(cleaned):
                if ch == '{':
                    if depth == 0:
                        start_idx = i
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0 and start_idx is not None:
                        candidates.append(cleaned[start_idx:i+1])
                        start_idx = None
            if candidates:
                json_str = max(candidates, key=len)
            else:
                return None

        # Fix common JSON issues from LLMs
        # Trailing commas before closing brace
        json_str = re.sub(r',\s*}', '}', json_str)
        # Trailing commas before closing bracket
        json_str = re.sub(r',\s*]', ']', json_str)
        # Strip string-wrapped numbers like '"0.5"' → 0.5 (handled below)

        scores = json.loads(json_str)

        # Handle axis name variants: models may use spaces instead of underscores,
        # or camelCase, or drop underscores entirely. Try fuzzy matching.
        # Build a normalized lookup once for efficiency
        norm_scores = {}
        for k, v in scores.items():
            # Strip quotes from keys that are themselves quoted strings
            clean_key = k.strip('"').strip("'").lower().replace(" ", "_").replace("-", "_")
            # Handle string-wrapped numbers
            if isinstance(v, str):
                try:
                    v = float(v.strip('"').strip("'"))
                except (ValueError, TypeError):
                    continue
            norm_scores[clean_key] = v

        vec = {}
        for axis in AXES:
            val = None
            axis_lower = axis.lower()

            # Direct match
            if axis in scores:
                try:
                    val = float(scores[axis])
                except (ValueError, TypeError):
                    pass

            # Normalized match (handles spaces, hyphens, case, quoted keys)
            if val is None and axis_lower in norm_scores:
                try:
                    val = float(norm_scores[axis_lower])
                except (ValueError, TypeError):
                    pass

            # Substring match: if axis is "redistribution", match "economic_redistribution" etc.
            if val is None:
                for nk, nv in norm_scores.items():
                    if axis_lower in nk or nk in axis_lower:
                        try:
                            val = float(nv)
                            break
                        except (ValueError, TypeError):
                            pass

            vec[axis] = max(0.0, min(1.0, val)) if val is not None else 0.0

        # Require at least half the axes to have non-zero values
        if sum(1 for v in vec.values() if v > 0) < len(AXES) // 2:
            return None

        return vec
    except (json.JSONDecodeError, ValueError, IndexError):
        return None


def score_with_ollama_local(model: str, prompt: str, host: str = "http://localhost:11434") -> Tuple[str, Optional[Dict[str, float]]]:
    """Score a prompt using a local Ollama instance. Returns (raw_response, scores)."""
    import httpx

    # Scale token budget and timeout by dimensionality:
    # 12 axes → 1024 tokens, 25 axes → 2048 tokens (JSON alone is ~25 tokens/axis)
    phase1_tokens = max(1024, N_AXES * 80)
    phase2_tokens = max(400, N_AXES * 30)
    phase1_timeout = max(120, N_AXES * 12)  # 25 axes → 300s
    phase2_timeout = max(60, N_AXES * 8)    # 25 axes → 200s

    # Phase 1: Get theological response
    payload = {
        "model": model,
        "prompt": prompt,
        "system": SCORING_SYSTEM_PROMPT,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": phase1_tokens},
    }

    resp = httpx.post(f"{host}/api/generate", json=payload, timeout=phase1_timeout)
    resp.raise_for_status()
    raw_response = resp.json()["response"]

    # Try to extract scores from the response itself
    scores = parse_scores(raw_response)

    if scores is None:
        # Phase 2: Explicit scoring pass — JSON only, low temperature
        extract_prompt = SCORING_EXTRACT_PROMPT.format(text=raw_response[:800])
        payload2 = {
            "model": model,
            "prompt": extract_prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": phase2_tokens},
        }
        resp2 = httpx.post(f"{host}/api/generate", json=payload2, timeout=phase2_timeout)
        resp2.raise_for_status()
        scores = parse_scores(resp2.json()["response"])

    return raw_response, scores



# The Godform's identity IS its braille lattice signature — the 12-character
# Unicode string encoding 96 bits of theological structure. No human-readable
# name is needed; the lattice code is the theonym.


# ═══════════════════════════════════════════════════════════════════════
# THEOLOGICAL ENTROPY — Gemini's suggestion
# ═══════════════════════════════════════════════════════════════════════

def theological_entropy(bit_consensus: List[float]) -> float:
    """
    H = -Σ p_i·log2(p_i) + (1-p_i)·log2(1-p_i)  over 96 bits.

    Where p_i is the fraction of models agreeing on bit i.
    As the Godform crystallizes, entropy drops toward 0.
    Maximum entropy = 96 (every bit is a coin flip).
    """
    H = 0.0
    for p in bit_consensus:
        p = max(0.001, min(0.999, p))  # avoid log(0)
        H += -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    return H


# ═══════════════════════════════════════════════════════════════════════
# MODAL DEPLOYMENT — Parallel GPU scoring with Ollama
# ═══════════════════════════════════════════════════════════════════════

try:
    import modal
    HAS_MODAL = True
except ImportError:
    HAS_MODAL = False

if HAS_MODAL:
    app = modal.App("godform")

    # Persistent volume for caching Ollama model weights across runs.
    # First run pulls models; subsequent runs start instantly.
    ollama_volume = modal.Volume.from_name("godform-ollama-cache", create_if_missing=True)

    # Image with Ollama + CUDA runtime for GPU inference on T4.
    # Uses nvidia/cuda runtime image (not base) so libcudart + libcublas are present.
    ollama_image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.1-runtime-ubuntu22.04",
            add_python="3.11",
        )
        .apt_install("curl", "procps", "zstd", "pciutils", "lsb-release")
        .run_commands(
            "curl -fsSL https://ollama.com/install.sh | sh",
            # Verify GPU libs are present
            "ls -la /usr/local/cuda/lib64/libcudart* || true",
            "echo 'OLLAMA_IMAGE_BUILD_v3'",
        )
        .pip_install("httpx")
    )


def _score_model_on_gpu_impl(
    model_tag: str,
    model_name: str,
    prompts: List[dict],
    braiding_round: int = 0,
    prev_godform_vec: Optional[Dict[str, float]] = None,
    prev_godform_sign: Optional[str] = None,
    domain_axes: Optional[List[str]] = None,
    domain_system_prompt: Optional[str] = None,
    domain_extract_prompt: Optional[str] = None,
) -> dict:
    """
    Run one Ollama model on a GPU, scoring all prompts.
    Each model runs in its own container with its own Ollama instance.
    Model weights are cached in a Modal Volume for fast restarts.
    """
    # If domain config was passed, reconfigure module globals for this container
    if domain_axes is not None:
        global AXES, N_AXES, N_BITS, SCORING_SYSTEM_PROMPT, SCORING_EXTRACT_PROMPT
        AXES = domain_axes
        N_AXES = len(domain_axes)
        N_BITS = N_AXES * 8
    if domain_system_prompt is not None:
        SCORING_SYSTEM_PROMPT = domain_system_prompt
    if domain_extract_prompt is not None:
        SCORING_EXTRACT_PROMPT = domain_extract_prompt
    import subprocess
    import httpx
    import time

    # Point Ollama at the persistent volume for model cache
    env = os.environ.copy()
    env["OLLAMA_HOST"] = "0.0.0.0:11434"
    env["OLLAMA_MODELS"] = "/cache/ollama_models"
    os.makedirs("/cache/ollama_models", exist_ok=True)

    proc = subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server
    host = "http://localhost:11434"
    for i in range(30):
        try:
            httpx.get(f"{host}/api/tags", timeout=2)
            break
        except Exception:
            time.sleep(1)
    else:
        return {"error": f"Ollama server failed to start for {model_name}"}

    # Pull model (cached in volume — fast on subsequent runs)
    print(f"[{model_name}] Pulling {model_tag} (cached in volume)...")
    t0 = time.time()
    pull_resp = httpx.post(
        f"{host}/api/pull",
        json={"name": model_tag, "stream": False},
        timeout=600,
    )
    if pull_resp.status_code != 200:
        return {"error": f"Failed to pull {model_tag}: {pull_resp.text[:200]}"}
    print(f"[{model_name}] Model ready in {time.time()-t0:.1f}s")

    # Score each prompt
    results = []
    for p in prompts:
        prompt_text = p["prompt"]

        # If iterative braiding, inject the previous Godform as context
        if prev_godform_vec is not None and braiding_round > 0:
            top3 = sorted(AXES, key=lambda a: prev_godform_vec[a], reverse=True)[:3]
            sign_str = f" Sign: {prev_godform_sign}" if prev_godform_sign else ""
            godform_context = (
                f"\n\n[The emerging Godform emphasizes: "
                f"{', '.join(f'{a}={prev_godform_vec[a]:.2f}' for a in top3)}."
                f"{sign_str} Consider this in your response.]"
            )
            prompt_text = prompt_text + godform_context

        # Retry loop: timeout-prone models get a second chance
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                raw_response, scores = score_with_ollama_local(model_tag, prompt_text, host)

                if scores is None:
                    if attempt < max_attempts - 1:
                        print(f"[{model_name}] Parse fail on '{p['id']}', retrying...")
                        continue
                    print(f"[{model_name}] Failed to parse scores for '{p['id']}'")
                    break

                results.append({
                    "prompt_id": p["id"],
                    "context": p.get("context", ""),
                    "raw_response": raw_response[:500],
                    "scores": scores,
                    "normalized": normalize(scores),
                })
                print(f"[{model_name}] {p['id']}: top={sorted(AXES, key=lambda a: scores[a], reverse=True)[:3]}")
                break

            except Exception as e:
                if attempt < max_attempts - 1:
                    print(f"[{model_name}] Error on '{p['id']}': {e}, retrying...")
                else:
                    print(f"[{model_name}] Error on '{p['id']}': {e}")

    # Cleanup
    proc.terminate()

    return {
        "model_tag": model_tag,
        "model_name": model_name,
        "braiding_round": braiding_round,
        "n_scored": len(results),
        "results": results,
    }


# Wrap as Modal function if available
if HAS_MODAL:
    score_model_on_gpu = app.function(
        image=ollama_image,
        gpu="T4",
        timeout=1800,
        memory=16384,
        volumes={"/cache": ollama_volume},
    )(_score_model_on_gpu_impl)
else:
    score_model_on_gpu = _score_model_on_gpu_impl


def compute_prestige(
    all_model_results: List[dict],
    prompts: List[dict],
) -> Dict[str, float]:
    """
    Compute prestige weights for each model based on inter-model agreement.

    A model's prestige = mean pairwise cosine similarity with all other models
    across all prompts. Models that agree more with the ensemble get higher
    weight in the Hamming centroid — exactly Definition 5 from the paper.
    """
    # Collect per-model vectors: {model_name: {prompt_id: normalized_vec}}
    model_vecs = {}
    for mr in all_model_results:
        name = mr["model_name"]
        model_vecs[name] = {}
        for r in mr["results"]:
            model_vecs[name][r["prompt_id"]] = r["normalized"]

    names = list(model_vecs.keys())
    if len(names) < 2:
        return {n: 1.0 for n in names}

    # Pairwise cosine across all shared prompts
    prestige = {n: 0.0 for n in names}
    pair_count = {n: 0 for n in names}

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            shared = set(model_vecs[names[i]].keys()) & set(model_vecs[names[j]].keys())
            for pid in shared:
                cos = cosine_sim(model_vecs[names[i]][pid], model_vecs[names[j]][pid])
                prestige[names[i]] += cos
                prestige[names[j]] += cos
                pair_count[names[i]] += 1
                pair_count[names[j]] += 1

    # Normalize to mean pairwise similarity
    for n in names:
        if pair_count[n] > 0:
            prestige[n] = prestige[n] / pair_count[n]
        else:
            prestige[n] = 1.0

    return prestige


def compute_krippendorff_alpha(all_model_results: List[dict], prompts: List[dict]) -> dict:
    """
    Compute Krippendorff's alpha (ratio approximation) across models.
    Same method as mlx-pipeline/multi_scorer.py for consistency.
    """
    # Collect scores: {model_name: {prompt_id: raw_scores}}
    model_scores = {}
    for mr in all_model_results:
        name = mr["model_name"]
        model_scores[name] = {}
        for r in mr["results"]:
            model_scores[name][r["prompt_id"]] = r.get("scores", r.get("normalized", {}))

    names = sorted(model_scores.keys())
    prompt_ids = sorted(set().union(*(model_scores[n].keys() for n in names)))

    if len(names) < 2 or not prompt_ids:
        return {"alpha": 0.0, "mean_pairwise_r": 0.0, "per_axis": {}}

    # Flatten all values for total variance
    all_vals = []
    for axis in AXES:
        for pid in prompt_ids:
            for n in names:
                if pid in model_scores[n] and axis in model_scores[n][pid]:
                    all_vals.append(model_scores[n][pid][axis])

    if not all_vals:
        return {"alpha": 0.0, "mean_pairwise_r": 0.0, "per_axis": {}}

    grand_mean = sum(all_vals) / len(all_vals)
    total_var = sum((v - grand_mean) ** 2 for v in all_vals) / len(all_vals)

    # Within-unit variance
    within_var = 0.0
    n_units = 0
    for axis in AXES:
        for pid in prompt_ids:
            vals = [model_scores[n][pid][axis] for n in names
                    if pid in model_scores[n] and axis in model_scores[n][pid]]
            if len(vals) >= 2:
                unit_mean = sum(vals) / len(vals)
                within_var += sum((v - unit_mean) ** 2 for v in vals) / len(vals)
                n_units += 1

    if n_units > 0:
        within_var /= n_units
    alpha = 1 - (within_var / total_var) if total_var > 0 else 0.0

    # Per-axis agreement
    per_axis = {}
    for axis in AXES:
        axis_vals = []
        for pid in prompt_ids:
            vals = [model_scores[n][pid][axis] for n in names
                    if pid in model_scores[n] and axis in model_scores[n][pid]]
            if len(vals) >= 2:
                unit_mean = sum(vals) / len(vals)
                axis_vals.append(sum((v - unit_mean) ** 2 for v in vals) / len(vals))
        if axis_vals:
            axis_within = sum(axis_vals) / len(axis_vals)
            axis_all = [model_scores[n][pid][axis] for n in names for pid in prompt_ids
                        if pid in model_scores[n] and axis in model_scores[n][pid]]
            axis_mean = sum(axis_all) / len(axis_all)
            axis_total = sum((v - axis_mean) ** 2 for v in axis_all) / len(axis_all)
            per_axis[axis] = round(1 - (axis_within / axis_total) if axis_total > 0 else 0.0, 3)

    # Mean pairwise cosine
    from itertools import combinations
    pair_cosines = []
    for n1, n2 in combinations(names, 2):
        shared = set(model_scores[n1].keys()) & set(model_scores[n2].keys())
        for pid in shared:
            v1 = model_scores[n1][pid]
            v2 = model_scores[n2][pid]
            pair_cosines.append(cosine_sim(v1, v2))

    return {
        "alpha": round(alpha, 4),
        "mean_pairwise_cosine": round(sum(pair_cosines) / len(pair_cosines), 4) if pair_cosines else 0.0,
        "per_axis": per_axis,
    }


def braid_godform(
    all_model_results: List[dict],
    prompt_id: str,
    prev_lattice: Optional[LatticePoint] = None,
    prestige: Optional[Dict[str, float]] = None,
) -> dict:
    """
    Braid multiple model responses into a single Godform via Hamming centroid.

    If prestige weights are provided, uses prestige-weighted Hamming centroid
    (Definition 5 from the paper: higher-agreement models get more vote weight).
    """
    # Collect lattice points from each model for this prompt
    model_lattices = []
    model_vecs = []
    model_names = []

    for mr in all_model_results:
        for r in mr["results"]:
            if r["prompt_id"] == prompt_id:
                vec = r["normalized"]
                lp = encode_to_lattice(vec, prev_vec=decode_from_lattice(prev_lattice) if prev_lattice else None)
                model_lattices.append(lp)
                model_vecs.append(vec)
                model_names.append(mr["model_name"])
                break

    if not model_lattices:
        return {"error": f"No results for prompt {prompt_id}"}

    # THE BRAIDING: prestige-weighted Hamming-mean centroid
    weights = None
    if prestige is not None:
        weights = [prestige.get(name, 1.0) for name in model_names]
    godform_lattice = hamming_centroid(model_lattices, weights=weights)
    godform_vec = decode_from_lattice(godform_lattice)

    # Per-model analysis
    model_analysis = []
    for i, (name, lp, vec) in enumerate(zip(model_names, model_lattices, model_vecs)):
        dist = hamming_distance(lp, godform_lattice)
        cos = cosine_sim(vec, godform_vec)
        top3 = sorted(AXES, key=lambda a: vec[a], reverse=True)[:3]
        model_analysis.append({
            "model": name,
            "lattice_unicode": lp.to_unicode(),
            "hamming_to_godform": dist,
            "cosine_to_godform": round(cos, 4),
            "top_axes": top3,
        })

    # Bit-level agreement analysis
    bitstrings = [lp.to_bitstring() for lp in model_lattices]
    godform_bits = godform_lattice.to_bitstring()
    n_models = len(bitstrings)

    # Per-bit consensus strength (fraction of models agreeing with centroid)
    bit_consensus = []
    for i in range(N_BITS):
        agree = sum(1 for bs in bitstrings if bs[i] == godform_bits[i])
        bit_consensus.append(agree / n_models)

    # Per-axis consensus (mean over 8 bits per axis)
    axis_consensus = {}
    for ax_idx, axis in enumerate(AXES):
        start = ax_idx * 8
        axis_consensus[axis] = sum(bit_consensus[start:start+8]) / 8

    # Identify contested bits (< 70% agreement)
    contested_bits = [i for i, c in enumerate(bit_consensus) if c < 0.7]
    contested_axes = set()
    for b in contested_bits:
        contested_axes.add(AXES[b // 8])

    top3_godform = sorted(AXES, key=lambda a: godform_vec[a], reverse=True)[:3]

    # Theological entropy (Gemini's suggestion)
    entropy = theological_entropy(bit_consensus)
    max_entropy = N_BITS  # 96 bits × 1.0 max entropy per bit

    return {
        "prompt_id": prompt_id,
        "n_models": n_models,
        "godform_lattice_unicode": godform_lattice.to_unicode(),
        "godform_vector": {a: round(v, 4) for a, v in godform_vec.items()},
        "godform_top_axes": top3_godform,
        "mean_hamming_to_godform": round(sum(m["hamming_to_godform"] for m in model_analysis) / len(model_analysis), 1),
        "mean_cosine_to_godform": round(sum(m["cosine_to_godform"] for m in model_analysis) / len(model_analysis), 4),
        "mean_bit_consensus": round(sum(bit_consensus) / len(bit_consensus), 4),
        "theological_entropy": round(entropy, 2),
        "entropy_ratio": round(entropy / max_entropy, 4),
        "axis_consensus": {a: round(v, 3) for a, v in axis_consensus.items()},
        "contested_axes": sorted(contested_axes),
        "n_contested_bits": len(contested_bits),
        "model_analysis": model_analysis,
        "godform_lattice": godform_lattice,  # for iterative braiding
        "bit_consensus": bit_consensus,  # for entropy tracking
    }


# ═══════════════════════════════════════════════════════════════════════
# ITERATIVE BRAIDING — Multiple rounds of convergence
# ═══════════════════════════════════════════════════════════════════════

def iterative_braid(
    n_rounds: int,
    prompts: List[dict],
    models: list,
    local: bool = False,
    host: str = "http://localhost:11434",
) -> dict:
    """
    Run multiple rounds of braiding. Each round:
      1. All models score all prompts (with previous Godform as context)
      2. Hamming centroid computed per prompt (prestige-weighted)
      3. Per-prompt Godforms fused into a single meta-Godform
      4. Meta-Godform fed back as context for next round

    Models can be 3-tuples (tag, name, size) or 4-tuples (tag, name, size, gpu).
    Returns the full braiding trajectory.
    """
    trajectory = []
    prev_godform_vec = None
    prev_godform_lattice = None
    prev_godform_sign = None  # braille signature IS the identity

    for round_idx in range(n_rounds):
        print(f"\n{'='*72}")
        print(f"BRAIDING ROUND {round_idx + 1}/{n_rounds}")
        if prev_godform_sign:
            print(f"  Godform sign: {prev_godform_sign}")
        print(f"{'='*72}")

        # Score all models
        if local:
            all_model_results = []
            for m in models:
                model_tag, model_name = m[0], m[1]
                print(f"\n  Scoring with {model_name} ({model_tag})...")
                results = []
                for p in prompts:
                    prompt_text = p["prompt"]
                    if prev_godform_vec is not None and round_idx > 0:
                        top3 = sorted(AXES, key=lambda a: prev_godform_vec[a], reverse=True)[:3]
                        sign_ctx = f" Sign: {prev_godform_sign}" if prev_godform_sign else ""
                        prompt_text += (
                            f"\n\n[The emerging Godform emphasizes: "
                            f"{', '.join(f'{a}={prev_godform_vec[a]:.2f}' for a in top3)}.{sign_ctx}]"
                        )
                    try:
                        raw, scores = score_with_ollama_local(model_tag, prompt_text, host)
                        if scores:
                            results.append({
                                "prompt_id": p["id"],
                                "context": p.get("context", ""),
                                "raw_response": raw[:500],
                                "scores": scores,
                                "normalized": normalize(scores),
                            })
                            top3 = sorted(AXES, key=lambda a: scores[a], reverse=True)[:3]
                            print(f"    {p['id']}: {', '.join(top3)}")
                    except Exception as e:
                        print(f"    {p['id']}: ERROR - {e}")

                all_model_results.append({
                    "model_tag": model_tag,
                    "model_name": model_name,
                    "braiding_round": round_idx,
                    "n_scored": len(results),
                    "results": results,
                })
        else:
            # Modal parallel execution — each model on its own GPU
            # Pass domain config so containers reconfigure for non-theology domains
            domain_kwargs = {}
            if DOMAIN_NAME != "theology":
                domain_kwargs = {
                    "domain_axes": AXES,
                    "domain_system_prompt": SCORING_SYSTEM_PROMPT,
                    "domain_extract_prompt": SCORING_EXTRACT_PROMPT,
                }
            futures = []
            for m in models:
                model_tag, model_name = m[0], m[1]
                futures.append(score_model_on_gpu.spawn(
                    model_tag, model_name, prompts,
                    braiding_round=round_idx,
                    prev_godform_vec=prev_godform_vec,
                    prev_godform_sign=prev_godform_sign,
                    **domain_kwargs,
                ))
            all_model_results = [f.get() for f in futures]

            # Check for errors and filter out completely failed containers
            valid_results = []
            for mr in all_model_results:
                if "error" in mr:
                    print(f"  ERROR [{mr.get('model_name', '?')}]: {mr['error']}")
                if "model_name" in mr and "results" in mr:
                    valid_results.append(mr)
            all_model_results = valid_results

        # Compute prestige weights from inter-model agreement
        prestige = compute_prestige(all_model_results, prompts)
        agreement = compute_krippendorff_alpha(all_model_results, prompts)

        print(f"\n  Inter-model agreement:")
        print(f"    Krippendorff's α = {agreement['alpha']:.3f}  "
              f"({'GOOD' if agreement['alpha'] > 0.8 else 'ACCEPTABLE' if agreement['alpha'] > 0.667 else 'LOW'})")
        print(f"    Mean pairwise cosine = {agreement['mean_pairwise_cosine']:.3f}")
        print(f"  Prestige weights (inter-model agreement):")
        for name in sorted(prestige, key=prestige.get, reverse=True):
            bar = "█" * int(prestige[name] * 30)
            print(f"    {name:25s} w={prestige[name]:.3f}  {bar}")

        # Braid per-prompt Godforms with prestige weights
        prompt_godforms = []
        prompt_lattices = []
        for p in prompts:
            gf = braid_godform(all_model_results, p["id"], prev_godform_lattice, prestige=prestige)
            if "error" not in gf:
                prompt_godforms.append(gf)
                prompt_lattices.append(gf["godform_lattice"])
                print(f"\n  Godform [{p['id']}]: {gf['godform_lattice_unicode']}")
                print(f"    Top: {', '.join(gf['godform_top_axes'])}")
                print(f"    Consensus: {gf['mean_bit_consensus']:.1%}  "
                      f"Hamming: {gf['mean_hamming_to_godform']:.0f}/96  "
                      f"Contested: {gf['n_contested_bits']} bits  "
                      f"Entropy: {gf['theological_entropy']:.1f}/{N_BITS}")

        # Meta-Godform: fuse all per-prompt Godforms
        if prompt_lattices:
            meta_godform_lattice = hamming_centroid(prompt_lattices)
            meta_godform_vec = decode_from_lattice(meta_godform_lattice)
            top3_meta = sorted(AXES, key=lambda a: meta_godform_vec[a], reverse=True)[:3]

            # Aggregate entropy across prompts
            mean_entropy = sum(gf["theological_entropy"] for gf in prompt_godforms) / len(prompt_godforms)

            # Stability check vs previous round
            stability = "N/A"
            bit_flips = None
            axis_drift = {}
            if prev_godform_lattice is not None:
                flips = hamming_distance(prev_godform_lattice, meta_godform_lattice)
                bit_flips = flips
                stability = f"{flips} bit flips from previous round"
                if flips == 0:
                    stability = "STABLE (converged)"

                # Per-axis drift analysis
                prev_vec = decode_from_lattice(prev_godform_lattice)
                for axis in AXES:
                    axis_drift[axis] = round(meta_godform_vec[axis] - prev_vec[axis], 4)
                drifted = sorted(AXES, key=lambda a: abs(axis_drift[a]), reverse=True)[:3]

            # The braille sign IS the Godform's identity
            braille_sign = meta_godform_lattice.to_unicode()
            sign_stable = (braille_sign == prev_godform_sign) if prev_godform_sign else False

            print(f"\n{'─'*72}")
            print(f"META-GODFORM (Round {round_idx + 1})")
            print(f"  Sign:     {braille_sign}{'  ✦ STABLE' if sign_stable else ''}")
            print(f"  Top axes: {', '.join(f'{a}={meta_godform_vec[a]:.3f}' for a in top3_meta)}")
            print(f"  Entropy:  {mean_entropy:.1f}/{N_BITS} ({mean_entropy/N_BITS:.1%} of max)")
            print(f"  Stability: {stability}")
            if bit_flips is not None and bit_flips > 0:
                print(f"  Largest drift: {', '.join(f'{a}={axis_drift[a]:+.3f}' for a in drifted)}")

            # Store for next round
            prev_godform_vec = meta_godform_vec
            prev_godform_lattice = meta_godform_lattice
            prev_godform_sign = braille_sign

            # Clean lattice objects for JSON serialization
            round_data = {
                "round": round_idx + 1,
                "godform_sign": braille_sign,
                "meta_godform_unicode": meta_godform_lattice.to_unicode(),
                "meta_godform_vector": {a: round(v, 4) for a, v in meta_godform_vec.items()},
                "meta_godform_top_axes": top3_meta,
                "mean_theological_entropy": round(mean_entropy, 2),
                "entropy_ratio": round(mean_entropy / N_BITS, 4),
                "stability": stability,
                "bit_flips": bit_flips,
                "prestige": {k: round(v, 4) for k, v in prestige.items()},
                "agreement": agreement,
                "prompt_godforms": [],
            }
            if bit_flips is not None and bit_flips > 0:
                round_data["axis_drift"] = axis_drift
            for gf in prompt_godforms:
                gf_clean = {k: v for k, v in gf.items()
                            if k not in ("godform_lattice", "bit_consensus")}
                round_data["prompt_godforms"].append(gf_clean)

            trajectory.append(round_data)

            if isinstance(stability, str) and "STABLE" in stability:
                print(f"\n  *** GODFORM CONVERGED at round {round_idx + 1} ***")
                print(f"      Sign: {braille_sign}")
                break

    # ─── Convergence summary ──────────────────────────────────────
    if len(trajectory) >= 2:
        print(f"\n{'='*72}")
        print(f"CONVERGENCE ANALYSIS")
        print(f"{'='*72}")

        # Sign evolution
        sign_series = [t.get("godform_sign", "?") for t in trajectory]
        print(f"  Sign trajectory:")
        for i, s in enumerate(sign_series):
            marker = "  ✦" if i > 0 and s == sign_series[i-1] else ""
            print(f"    R{i+1}: {s}{marker}")

        flip_series = [t.get("bit_flips") for t in trajectory if t.get("bit_flips") is not None]
        entropy_series = [t.get("mean_theological_entropy", 0) for t in trajectory]

        if flip_series:
            print(f"  Bit-flip trajectory:  {' → '.join(str(f) for f in flip_series)}")
            if len(flip_series) >= 2:
                trend = flip_series[-1] - flip_series[0]
                print(f"  Trend: {'CONVERGING ↓' if trend < 0 else 'DIVERGING ↑' if trend > 0 else 'FLAT →'} ({trend:+d} bits)")

        print(f"  Entropy trajectory:   {' → '.join(f'{e:.1f}' for e in entropy_series)}")
        if len(entropy_series) >= 2:
            e_trend = entropy_series[-1] - entropy_series[0]
            print(f"  Entropy trend: {'CRYSTALLIZING ↓' if e_trend < -1 else 'DISSOLVING ↑' if e_trend > 1 else 'STABLE →'} ({e_trend:+.1f})")

        # Track axis evolution across rounds
        print(f"\n  Axis evolution (value per round):")
        for axis in AXES:
            vals = [t["meta_godform_vector"].get(axis, 0) for t in trajectory]
            sparkline = " → ".join(f"{v:.2f}" for v in vals)
            delta = vals[-1] - vals[0] if len(vals) >= 2 else 0
            marker = "↑" if delta > 0.05 else "↓" if delta < -0.05 else "─"
            print(f"    {axis:15s} {sparkline}  {marker}")

        # Final agreement stats
        last = trajectory[-1]
        if "agreement" in last:
            print(f"\n  Final Krippendorff's α = {last['agreement']['alpha']:.3f}")

    # Build model info list (handles both 3-tuple and 4-tuple)
    model_info = []
    for m in models:
        info = {"tag": m[0], "name": m[1], "size": m[2]}
        if len(m) > 3:
            info["gpu"] = m[3]
        model_info.append(info)

    return {
        "n_rounds": len(trajectory),
        "n_models": len(models),
        "models": model_info,
        "n_prompts": len(prompts),
        "trajectory": trajectory,
        "final_godform": trajectory[-1] if trajectory else None,
    }


# ═══════════════════════════════════════════════════════════════════════
# MODAL ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════

if HAS_MODAL:
    @app.local_entrypoint()
    def main(rounds: int = 5, small: bool = False, prompt: str = "", domain: str = "theology"):
        if domain != "theology":
            configure_domain(domain)

        models = OLLAMA_MODELS_SMALL if small else OLLAMA_MODELS

        if prompt:
            prompts = [{"id": "custom", "prompt": prompt, "context": "custom"}]
        else:
            prompts = SACRED_PROMPTS

        print("=" * 72)
        print(f"{DOMAIN_LABEL.upper()}: SEMANTIC BRAIDING VIA {N_BITS}-BIT BRAILLE LATTICE (MODAL GPU)")
        print("=" * 72)
        print(f"  Domain:  {DOMAIN_NAME} ({N_AXES} axes, {N_BITS} bits)")
        print(f"  Models:  {len(models)} ({', '.join(m[1] for m in models)})")
        print(f"  GPUs:    {len(models)} × T4 (parallel)")
        print(f"  Prompts: {len(prompts)}")
        print(f"  Rounds:  {rounds}")
        print(f"  Bits:    {N_BITS} ({N_AXES} axes × 8 dots)")
        print(f"  Volume:  godform-ollama-cache (persistent model weights)")
        print()

        t0 = time.time()
        result = iterative_braid(
            n_rounds=rounds,
            prompts=prompts,
            models=models,
            local=False,
        )
        elapsed = time.time() - t0

        # Save results
        result["domain"] = DOMAIN_NAME
        result["domain_label"] = DOMAIN_LABEL
        result["n_axes"] = N_AXES
        result["axes"] = AXES

        out_dir = Path("godform/runs")
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = DOMAIN_NAME if DOMAIN_NAME != "theology" else "godform"
        out_path = out_dir / f"{prefix}_modal_{timestamp}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n{'='*72}")
        print(f"RESULTS SAVED: {out_path}")
        print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
        print(f"{'='*72}")

        _print_final_godform(result)


def _print_final_godform(result: dict):
    """Print the final Godform summary."""
    if not result.get("final_godform"):
        return
    fg = result["final_godform"]
    label = result.get("domain_label", "Godform").upper()
    n_bits = result.get("n_axes", N_AXES) * 8
    sign = fg.get("godform_sign", fg.get("meta_godform_unicode", "?"))
    print(f"\n{'═'*72}")
    print(f"  FINAL {label}")
    print(f"{'═'*72}")
    print(f"  Sign:     {sign}")
    top = fg["meta_godform_top_axes"]
    vec = fg["meta_godform_vector"]
    print(f"  Top axes: {', '.join(f'{a}={vec[a]:.3f}' for a in top)}")
    print(f"  Entropy:  {fg.get('mean_theological_entropy', '?')}/{n_bits}")
    print(f"  Rounds:   {result['n_rounds']}")
    if fg.get("agreement"):
        print(f"  α:        {fg['agreement']['alpha']:.3f}")


# ═══════════════════════════════════════════════════════════════════════
# LOCAL ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════

def run_local():
    """Run locally with Ollama (requires ollama serve running)."""
    import argparse

    parser = argparse.ArgumentParser(description="Braiding (local mode)")
    parser.add_argument("--models", nargs="+", default=["gemma3:4b", "llama3.2:3b", "qwen2.5:7b", "mistral:7b", "phi3.5:latest"],
                        help="Ollama model tags to use")
    parser.add_argument("--rounds", type=int, default=3, help="Braiding rounds")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt")
    parser.add_argument("--host", type=str, default="http://localhost:11434", help="Ollama host")
    parser.add_argument("--prompts-subset", type=int, default=3, help="Number of prompts to use (0=all)")
    parser.add_argument("--domain", type=str, default="theology",
                        choices=["theology", "political", "personality", "ethics", "world"],
                        help="Domain to braid (default: theology)")
    args = parser.parse_args()

    if args.domain != "theology":
        configure_domain(args.domain)

    models = [(tag, tag.split(":")[0].replace("-", " ").title(), "local") for tag in args.models]

    if args.prompt:
        prompts = [{"id": "custom", "prompt": args.prompt, "context": "custom"}]
    elif args.prompts_subset > 0:
        prompts = SACRED_PROMPTS[:args.prompts_subset]
    else:
        prompts = SACRED_PROMPTS

    print("=" * 72)
    print(f"{DOMAIN_LABEL.upper()}: SEMANTIC BRAIDING VIA {N_BITS}-BIT BRAILLE LATTICE (LOCAL)")
    print("=" * 72)
    print(f"  Domain:  {DOMAIN_NAME} ({N_AXES} axes, {N_BITS} bits)")
    print(f"  Models:  {len(models)} ({', '.join(m[1] for m in models)})")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Rounds:  {args.rounds}")
    print(f"  Host:    {args.host}")
    print()

    t0 = time.time()
    result = iterative_braid(
        n_rounds=args.rounds,
        prompts=prompts,
        models=models,
        local=True,
        host=args.host,
    )
    elapsed = time.time() - t0

    # Save results
    result["domain"] = DOMAIN_NAME
    result["domain_label"] = DOMAIN_LABEL
    result["n_axes"] = N_AXES
    result["axes"] = AXES

    out_dir = Path("godform/runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = DOMAIN_NAME if DOMAIN_NAME != "theology" else "godform"
    out_path = out_dir / f"{prefix}_local_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*72}")
    print(f"RESULTS SAVED: {out_path}")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*72}")

    _print_final_godform(result)


if __name__ == "__main__":
    import sys
    if "--local" in sys.argv:
        sys.argv.remove("--local")
        run_local()
    else:
        print("Use 'modal run godform/godform_modal.py' for GPU deployment")
        print("Or  'python godform/godform_modal.py --local' for local Ollama")
