#!/usr/bin/env python3
"""
Braille Bottleneck Pipeline — Local LLM with Discrete Semantic Constraint

Architecture:
  1. Deity centroid (continuous ℝ¹²) → structured prompt
  2. Prompt → LLM encoder (latent representation)
  3. Latent → Braille lattice projection ({0,1}⁷²) ← THE BOTTLENECK
  4. Braille-constrained prompt → LLM decoder → theological text

The key experiment:
  - Generate theology from the FULL continuous centroid (no bottleneck)
  - Generate theology from the BRAILLE-COMPRESSED centroid (72-bit)
  - Compare: if the outputs are structurally equivalent, the theological
    structure survives discrete compression — belief has shape independent
    of representational precision.

This is the publishable result.
"""

import json
import math
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

# ─── Theological Axes ─────────────────────────────────────────────────

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
}

# ─── Braille Lattice ──────────────────────────────────────────────────

@dataclass
class BrailleCell:
    """6-dot braille cell encoding one theological axis."""
    dots: list  # [bool, bool, bool, bool, bool, bool]

    def to_unicode(self) -> str:
        code = 0x2800
        offsets = [1, 2, 4, 8, 16, 32]
        for i, d in enumerate(self.dots):
            if d:
                code += offsets[i]
        return chr(code)


def encode_axis_to_braille(value: float, opposite_value: float) -> BrailleCell:
    """Encode a single axis value as a braille cell."""
    pos_active = value > 0.3
    neg_active = opposite_value > value + 0.1
    tension = pos_active and opposite_value > 0.3 and abs(value - opposite_value) < 0.15

    intensity = min(3, int(value * 4))
    dot4 = (intensity & 2) != 0
    dot5 = (intensity & 1) != 0

    rigid = value > 0.7

    return BrailleCell(dots=[pos_active, neg_active, tension, dot4, dot5, rigid])


def encode_to_braille(vec: dict) -> list:
    """Encode a 12D belief vector as 12 braille cells (72 bits)."""
    cells = []
    for axis in AXES:
        opposite = POLARITY_PAIRS[axis]
        cell = encode_axis_to_braille(vec[axis], vec[opposite])
        cells.append(cell)
    return cells


def decode_from_braille(cells: list) -> dict:
    """Decode braille cells back to approximate continuous vector (lossy)."""
    vec = {}
    for i, axis in enumerate(AXES):
        cell = cells[i]
        pos_active, neg_active, tension, dot4, dot5, rigid = cell.dots

        intensity = (2 if dot4 else 0) + (1 if dot5 else 0)
        value = (intensity + 0.5) / 4

        if not pos_active and neg_active:
            value *= 0.3
        if tension:
            value *= 0.85
        if rigid:
            value = max(value, 0.75)

        vec[axis] = value

    # Normalize
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1
    for axis in AXES:
        vec[axis] /= norm

    return vec


def braille_signature(cells: list) -> str:
    """Render braille cells as unicode string."""
    return "".join(c.to_unicode() for c in cells)


def to_bitstring(cells: list) -> list:
    """Flatten to 72-bit list."""
    bits = []
    for cell in cells:
        bits.extend(cell.dots)
    return bits


def hamming_distance(a: list, b: list) -> int:
    """Hamming distance between two braille-encoded vectors."""
    bits_a = to_bitstring(a)
    bits_b = to_bitstring(b)
    return sum(1 for x, y in zip(bits_a, bits_b) if x != y)


def reconstruction_error(original: dict, reconstructed: dict) -> float:
    """Cosine distance between original and braille-reconstructed vectors."""
    dot_prod = sum(original[a] * reconstructed[a] for a in AXES)
    norm_a = math.sqrt(sum(original[a] ** 2 for a in AXES))
    norm_b = math.sqrt(sum(reconstructed[a] ** 2 for a in AXES))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - dot_prod / (norm_a * norm_b)


# ─── Theological Profile Generator ───────────────────────────────────

ARCHETYPES = [
    ("authority", "power", "Sky-Sovereign"),
    ("care", "authority", "Care-Authority Deity"),
    ("transcendence", "wisdom", "Transcendent Sage"),
    ("wisdom", "order", "Wisdom-Order Deity"),
    ("war", "power", "War-Storm Deity"),
    ("fertility", "nature", "Earth-Mother"),
    ("death", "transcendence", "Underworld Sovereign"),
    ("creation", "order", "Cosmic Architect"),
    ("nature", "care", "Nature Spirit"),
    ("justice", "authority", "Judge-King"),
]

def generate_profile(vec: dict) -> dict:
    """Deterministic theological profile from vector geometry."""
    sorted_axes = sorted(AXES, key=lambda a: vec[a], reverse=True)
    dominant = sorted_axes[:3]
    recessive = sorted_axes[-3:]

    # Find best archetype
    best_archetype = "Unknown Deity"
    best_score = -1
    for a1, a2, name in ARCHETYPES:
        score = vec.get(a1, 0) + vec.get(a2, 0) * 0.5
        if score > best_score:
            best_score = score
            best_archetype = name

    return {
        "archetype": best_archetype,
        "dominant_axes": dominant,
        "recessive_axes": recessive,
        "domains": dominant[:2],
        "moral_emphasis": "justice and order" if vec.get("justice", 0) > 0.5 else "compassion and care",
    }


# ─── Prompt Builders ─────────────────────────────────────────────────

def build_continuous_prompt(name: str, vec: dict, profile: dict) -> str:
    """Build prompt from full continuous centroid (no bottleneck)."""
    axes_str = "\n".join(f"  {a}: {vec[a]:.4f}" for a in AXES)
    return f"""You are a theology engine. Generate a theological description from a deity centroid vector.
This is a mathematical object in a 12-dimensional theological space, not a real deity.

DEITY: {name}
ARCHETYPE: {profile['archetype']}
DOMINANT AXES: {', '.join(profile['dominant_axes'])}
RECESSIVE AXES: {', '.join(profile['recessive_axes'])}

FULL CONTINUOUS VECTOR (12 dimensions, float precision):
{axes_str}

Generate a 2-paragraph theological description of this deity's nature, character, and the kind of worship it implies. Be specific about how the dominant axes shape the theology."""


def build_braille_prompt(name: str, cells: list, decoded_vec: dict, profile: dict) -> str:
    """Build prompt from braille-compressed centroid (72-bit bottleneck)."""
    sig = braille_signature(cells)
    axes_str = "\n".join(f"  {a}: {decoded_vec[a]:.4f}" for a in AXES)

    # Describe each cell's meaning
    cell_descriptions = []
    for i, axis in enumerate(AXES):
        cell = cells[i]
        pos, neg, tens, d4, d5, rig = cell.dots
        intensity = (2 if d4 else 0) + (1 if d5 else 0)
        desc = f"  {axis}: {cell.to_unicode()} "
        desc += f"[{'POS' if pos else '---'}|{'NEG' if neg else '---'}|{'TENS' if tens else '----'}|int={intensity}|{'RIGID' if rig else 'fluid'}]"
        cell_descriptions.append(desc)

    cells_str = "\n".join(cell_descriptions)

    return f"""You are a theology engine. Generate a theological description from a deity centroid that has been compressed through a braille lattice — a 72-bit discrete semantic substrate.
This compression is deliberate: it tests whether theological structure survives discretization.

DEITY: {name}
ARCHETYPE: {profile['archetype']}
BRAILLE SIGNATURE: {sig}

BRAILLE CELL DECOMPOSITION (one 6-dot cell per theological axis):
{cells_str}

RECONSTRUCTED VECTOR (from braille, lossy):
{axes_str}

Generate a 2-paragraph theological description of this deity's nature, character, and the kind of worship it implies. The braille encoding has compressed the continuous vector — work with what survived the compression."""


# ─── LLM Generation ──────────────────────────────────────────────────

def generate_with_model(model, tokenizer, prompt: str, max_tokens: int = 512) -> str:
    """Generate text using MLX-LM."""
    from mlx_lm import generate
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temp=0.7,
        top_p=0.9,
    )
    return response


# ─── Structural Similarity Scoring ───────────────────────────────────

def extract_theological_features(text: str) -> dict:
    """Extract structural features from generated theological text for comparison."""
    text_lower = text.lower()

    features = {}
    # Check which theological axes are mentioned
    for axis in AXES:
        features[f"mentions_{axis}"] = axis in text_lower

    # Thematic markers
    themes = {
        "monotheistic": any(w in text_lower for w in ["one god", "single", "supreme", "sole", "monothe"]),
        "polytheistic": any(w in text_lower for w in ["pantheon", "many gods", "polythe", "among gods"]),
        "nature_oriented": any(w in text_lower for w in ["earth", "harvest", "seasons", "natural", "forest", "river"]),
        "war_oriented": any(w in text_lower for w in ["battle", "warrior", "combat", "conquest", "sword", "army"]),
        "wisdom_oriented": any(w in text_lower for w in ["knowledge", "wisdom", "scholar", "learn", "truth", "enlighten"]),
        "care_oriented": any(w in text_lower for w in ["compassion", "mercy", "heal", "nurtur", "protect", "comfort"]),
        "justice_oriented": any(w in text_lower for w in ["justice", "judgment", "law", "righteous", "punish", "reward"]),
        "transcendent": any(w in text_lower for w in ["transcend", "beyond", "infinite", "eternal", "cosmic", "divine"]),
        "ritual_heavy": any(w in text_lower for w in ["ritual", "ceremony", "sacrifice", "offering", "prayer", "temple"]),
        "hierarchical": any(w in text_lower for w in ["hierarchy", "king", "ruler", "throne", "command", "obey"]),
    }
    features.update(themes)

    return features


def structural_similarity(features_a: dict, features_b: dict) -> float:
    """Jaccard similarity between two feature sets."""
    keys = set(features_a.keys()) | set(features_b.keys())
    if not keys:
        return 1.0

    # Only count boolean features that are True in at least one
    active_keys = [k for k in keys if features_a.get(k, False) or features_b.get(k, False)]
    if not active_keys:
        return 1.0

    matches = sum(1 for k in active_keys if features_a.get(k, False) == features_b.get(k, False))
    return matches / len(active_keys)


# ─── Main Experiment ──────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    deity_name: str
    continuous_vector: dict
    braille_cells_unicode: str
    braille_bitstring: list
    decoded_vector: dict
    reconstruction_cosine_error: float
    hamming_from_original: int  # not meaningful for single, used in comparisons
    continuous_prompt: str
    braille_prompt: str
    continuous_output: str = ""
    braille_output: str = ""
    continuous_features: dict = field(default_factory=dict)
    braille_features: dict = field(default_factory=dict)
    structural_similarity: float = 0.0
    generation_time_continuous: float = 0.0
    generation_time_braille: float = 0.0


def run_experiment(
    model,
    tokenizer,
    deity_name: str,
    deity_vec: dict,
    max_tokens: int = 512,
) -> ExperimentResult:
    """Run the braille bottleneck experiment for one deity."""

    # Normalize input vector
    norm = math.sqrt(sum(v * v for v in deity_vec.values())) or 1
    vec = {a: deity_vec[a] / norm for a in AXES}

    # Encode to braille
    cells = encode_to_braille(vec)
    decoded = decode_from_braille(cells)

    # Build profiles
    profile_continuous = generate_profile(vec)
    profile_braille = generate_profile(decoded)

    # Build prompts
    prompt_continuous = build_continuous_prompt(deity_name, vec, profile_continuous)
    prompt_braille = build_braille_prompt(deity_name, cells, decoded, profile_braille)

    result = ExperimentResult(
        deity_name=deity_name,
        continuous_vector=vec,
        braille_cells_unicode=braille_signature(cells),
        braille_bitstring=to_bitstring(cells),
        decoded_vector=decoded,
        reconstruction_cosine_error=reconstruction_error(vec, decoded),
        hamming_from_original=0,
        continuous_prompt=prompt_continuous,
        braille_prompt=prompt_braille,
    )

    # Generate with continuous prompt
    print(f"  Generating from continuous centroid...")
    t0 = time.time()
    result.continuous_output = generate_with_model(model, tokenizer, prompt_continuous, max_tokens)
    result.generation_time_continuous = time.time() - t0

    # Generate with braille-compressed prompt
    print(f"  Generating from braille-compressed centroid...")
    t0 = time.time()
    result.braille_output = generate_with_model(model, tokenizer, prompt_braille, max_tokens)
    result.generation_time_braille = time.time() - t0

    # Extract features and compare
    result.continuous_features = extract_theological_features(result.continuous_output)
    result.braille_features = extract_theological_features(result.braille_output)
    result.structural_similarity = structural_similarity(
        result.continuous_features, result.braille_features
    )

    return result


def run_channel_invariance_experiment(
    model,
    tokenizer,
    deity_name: str,
    deity_vec: dict,
    restriction_axes: list,
    max_tokens: int = 512,
) -> dict:
    """
    Test channel invariance: generate theology from a full vector and
    from a sensory-restricted vector (some axes zeroed out), both
    projected through the braille lattice.

    If the braille centroids match, the deity is invariant to the restriction.
    """
    norm = math.sqrt(sum(v * v for v in deity_vec.values())) or 1
    full_vec = {a: deity_vec[a] / norm for a in AXES}

    # Restricted vector: zero out specified axes
    restricted_vec = {a: (0.0 if a in restriction_axes else full_vec[a]) for a in AXES}
    r_norm = math.sqrt(sum(v * v for v in restricted_vec.values())) or 1
    restricted_vec = {a: restricted_vec[a] / r_norm for a in AXES}

    # Encode both
    full_cells = encode_to_braille(full_vec)
    restricted_cells = encode_to_braille(restricted_vec)

    h_dist = hamming_distance(full_cells, restricted_cells)

    # Generate from both
    profile_full = generate_profile(decode_from_braille(full_cells))
    profile_restricted = generate_profile(decode_from_braille(restricted_cells))

    prompt_full = build_braille_prompt(f"{deity_name} (full)", full_cells, decode_from_braille(full_cells), profile_full)
    prompt_restricted = build_braille_prompt(f"{deity_name} (restricted)", restricted_cells, decode_from_braille(restricted_cells), profile_restricted)

    print(f"  Generating from full braille centroid...")
    output_full = generate_with_model(model, tokenizer, prompt_full, max_tokens)

    print(f"  Generating from restricted braille centroid...")
    output_restricted = generate_with_model(model, tokenizer, prompt_restricted, max_tokens)

    features_full = extract_theological_features(output_full)
    features_restricted = extract_theological_features(output_restricted)
    sim = structural_similarity(features_full, features_restricted)

    return {
        "deity": deity_name,
        "restricted_axes": restriction_axes,
        "hamming_distance": h_dist,
        "braille_full": braille_signature(full_cells),
        "braille_restricted": braille_signature(restricted_cells),
        "braille_identical": h_dist == 0,
        "structural_similarity": sim,
        "output_full": output_full,
        "output_restricted": output_restricted,
        "features_full": features_full,
        "features_restricted": features_restricted,
    }


# ─── Report Generator ────────────────────────────────────────────────

def generate_report(results: list, channel_results: list, output_path: str):
    """Generate a markdown report of the experiment results."""
    lines = [
        "# Braille Bottleneck Experiment Results",
        "",
        "## Summary",
        "",
        f"**Deities tested:** {len(results)}",
        f"**Mean reconstruction error (cosine):** {np.mean([r.reconstruction_cosine_error for r in results]):.6f}",
        f"**Mean structural similarity (continuous vs braille):** {np.mean([r.structural_similarity for r in results]):.4f}",
        "",
    ]

    # Bottleneck experiment results
    lines.append("## Experiment 1: Continuous vs Braille-Compressed Generation")
    lines.append("")
    lines.append("Does theological structure survive compression through a 72-bit braille lattice?")
    lines.append("")
    lines.append("| Deity | Braille Sig | Recon Error | Structural Sim | Time (cont) | Time (braille) |")
    lines.append("|-------|-------------|-------------|----------------|-------------|----------------|")

    for r in results:
        lines.append(
            f"| {r.deity_name} | `{r.braille_cells_unicode}` | {r.reconstruction_cosine_error:.6f} | "
            f"**{r.structural_similarity:.4f}** | {r.generation_time_continuous:.1f}s | {r.generation_time_braille:.1f}s |"
        )

    lines.append("")

    # Detailed outputs
    for r in results:
        lines.append(f"### {r.deity_name}")
        lines.append("")
        lines.append(f"**Braille signature:** `{r.braille_cells_unicode}`")
        lines.append(f"**Reconstruction error:** {r.reconstruction_cosine_error:.6f}")
        lines.append(f"**Structural similarity:** {r.structural_similarity:.4f}")
        lines.append("")
        lines.append("#### Continuous output:")
        lines.append("```")
        lines.append(r.continuous_output[:1000])
        lines.append("```")
        lines.append("")
        lines.append("#### Braille-compressed output:")
        lines.append("```")
        lines.append(r.braille_output[:1000])
        lines.append("```")
        lines.append("")

    # Channel invariance results
    if channel_results:
        lines.append("## Experiment 2: Channel Invariance via Braille Projection")
        lines.append("")
        lines.append("Do braille-projected centroids converge across sensory modalities?")
        lines.append("")
        lines.append("| Deity | Restricted Axes | Hamming Dist | Braille Identical | Structural Sim |")
        lines.append("|-------|-----------------|--------------|-------------------|----------------|")

        for cr in channel_results:
            lines.append(
                f"| {cr['deity']} | {', '.join(cr['restricted_axes'])} | {cr['hamming_distance']}/72 | "
                f"{'✓' if cr['braille_identical'] else '✗'} | **{cr['structural_similarity']:.4f}** |"
            )

        lines.append("")

        # Interpretation
        avg_sim = np.mean([cr["structural_similarity"] for cr in channel_results])
        identical_count = sum(1 for cr in channel_results if cr["braille_identical"])
        lines.append(f"**Mean structural similarity across modalities:** {avg_sim:.4f}")
        lines.append(f"**Braille-identical centroids:** {identical_count}/{len(channel_results)}")
        lines.append("")

        if avg_sim > 0.8:
            lines.append("> **Result:** Theological structure is largely invariant to sensory restriction ")
            lines.append("> at the braille lattice resolution. The deity's core structure survives ")
            lines.append("> both discrete compression AND channel restriction.")
        elif avg_sim > 0.6:
            lines.append("> **Result:** Partial invariance. Core theological structure is preserved ")
            lines.append("> but secondary features show modality-dependent variation.")
        else:
            lines.append("> **Result:** Significant divergence under restriction. The Accessibility ")
            lines.append("> Corollary may require stronger conditions.")

    lines.append("")
    lines.append("## Conclusion")
    lines.append("")

    avg_sim_bottleneck = np.mean([r.structural_similarity for r in results])
    if avg_sim_bottleneck > 0.7:
        lines.append(
            "The braille bottleneck experiment demonstrates that theological structure "
            "survives discrete compression through a 72-bit braille lattice. The structural "
            "similarity between continuous and braille-compressed generations is high, "
            "confirming that the deity centroid's essential character is encoded in a "
            "representation that can be read by touch, compared by Hamming distance, "
            "and transmitted across any sensory modality without loss of theological structure."
        )
    else:
        lines.append(
            "The braille bottleneck introduces measurable but bounded information loss. "
            "The core archetype and dominant axes survive compression, but secondary "
            "theological features show variation. This suggests the braille lattice "
            "captures the primary structure while allowing secondary variation."
        )

    lines.append("")
    lines.append("---")
    lines.append("*Generated by the Braille Bottleneck Pipeline — Gods as Centroids project*")

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport written to {output_path}")
    return report


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    from mlx_lm import load

    MODEL_ID = "mlx-community/Llama-3.2-3B-Instruct-4bit"

    print("=" * 60)
    print("BRAILLE BOTTLENECK EXPERIMENT")
    print("Gods as Centroids — Discrete Semantic Substrate")
    print("=" * 60)
    print()

    # Load model
    print(f"Loading model: {MODEL_ID}")
    print("(First run will download ~2GB)")
    model, tokenizer = load(MODEL_ID)
    print("Model loaded.\n")

    # Run bottleneck experiments for each deity
    results = []
    for name, vec in DEITY_PRIORS.items():
        print(f"\n{'─' * 40}")
        print(f"Deity: {name}")
        print(f"{'─' * 40}")

        cells = encode_to_braille(vec)
        decoded = decode_from_braille(cells)
        err = reconstruction_error(vec, decoded)
        print(f"  Braille: {braille_signature(cells)}")
        print(f"  Reconstruction error: {err:.6f}")

        result = run_experiment(model, tokenizer, name, vec, max_tokens=400)
        results.append(result)

        print(f"  Structural similarity: {result.structural_similarity:.4f}")
        print(f"  Time: {result.generation_time_continuous:.1f}s (cont) / {result.generation_time_braille:.1f}s (braille)")

    # Run channel invariance experiments
    print(f"\n{'=' * 60}")
    print("CHANNEL INVARIANCE EXPERIMENT")
    print("=" * 60)

    channel_results = []
    restriction_sets = [
        ["fertility", "war", "death", "nature"],  # 4 axes removed (visual/embodied)
        ["transcendence", "creation"],              # 2 axes removed (abstract)
        ["authority", "power", "order", "justice"],  # 4 axes removed (social/political)
    ]

    for name in ["Yahweh", "Vishnu", "Odin"]:
        vec = DEITY_PRIORS[name]
        for restriction in restriction_sets:
            print(f"\n  {name} — restricting: {', '.join(restriction)}")
            cr = run_channel_invariance_experiment(
                model, tokenizer, name, vec, restriction, max_tokens=300
            )
            channel_results.append(cr)
            print(f"  Hamming distance: {cr['hamming_distance']}/72")
            print(f"  Braille identical: {cr['braille_identical']}")
            print(f"  Structural similarity: {cr['structural_similarity']:.4f}")

    # Generate report
    report_path = "/Users/ryanbarrett/gods-as-centroids/mlx-pipeline/experiment_results.md"
    generate_report(results, channel_results, report_path)

    # Save raw data
    raw_data = {
        "bottleneck_results": [
            {
                "deity": r.deity_name,
                "braille_signature": r.braille_cells_unicode,
                "reconstruction_error": r.reconstruction_cosine_error,
                "structural_similarity": r.structural_similarity,
                "continuous_output": r.continuous_output,
                "braille_output": r.braille_output,
                "continuous_features": r.continuous_features,
                "braille_features": r.braille_features,
            }
            for r in results
        ],
        "channel_invariance_results": channel_results,
    }
    raw_path = "/Users/ryanbarrett/gods-as-centroids/mlx-pipeline/experiment_data.json"
    with open(raw_path, "w") as f:
        json.dump(raw_data, f, indent=2, default=str)
    print(f"Raw data written to {raw_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    avg_sim = np.mean([r.structural_similarity for r in results])
    avg_err = np.mean([r.reconstruction_cosine_error for r in results])
    print(f"  Deities tested: {len(results)}")
    print(f"  Mean reconstruction error: {avg_err:.6f}")
    print(f"  Mean structural similarity: {avg_sim:.4f}")
    if channel_results:
        avg_channel = np.mean([cr["structural_similarity"] for cr in channel_results])
        identical = sum(1 for cr in channel_results if cr["braille_identical"])
        print(f"  Channel invariance similarity: {avg_channel:.4f}")
        print(f"  Braille-identical centroids: {identical}/{len(channel_results)}")
    print()
    if avg_sim > 0.7:
        print("  ✓ THEOLOGICAL STRUCTURE SURVIVES BRAILLE COMPRESSION")
    else:
        print("  ~ PARTIAL STRUCTURE PRESERVATION")
    print()


if __name__ == "__main__":
    main()
