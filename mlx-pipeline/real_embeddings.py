#!/usr/bin/env python3
"""
Real Embeddings Experiment — Braille Bottleneck on Actual Religious Texts

Instead of hand-designed deity priors, we:
1. Curate canonical passages from real religious traditions
2. Use a frontier LLM to score each passage on 12 theological axes
3. Compress the resulting embeddings through the 72-bit braille bottleneck
4. Measure whether real theological relationships survive compression
5. Compare against baselines (random binary projection, PCA+quantize)

This is the experiment that makes the paper real.
"""

import json
import math
import os
import time
import random
import numpy as np
import httpx
from pathlib import Path
from itertools import combinations

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

# ─── Corpus: Real Religious Texts ─────────────────────────────────────
# Each entry: tradition, source text, canonical passage
# Selected for theological density and diversity

CORPUS = [
    # ─── Abrahamic: Judaism ───
    {
        "tradition": "Judaism",
        "source": "Torah (Exodus 20:1-6)",
        "text": "I am the LORD your God, who brought you out of Egypt, out of the land of slavery. You shall have no other gods before me. You shall not make for yourself an image in the form of anything in heaven above or on the earth beneath or in the waters below. You shall not bow down to them or worship them; for I, the LORD your God, am a jealous God, punishing the children for the sin of the parents to the third and fourth generation of those who hate me, but showing love to a thousand generations of those who love me and keep my commandments.",
        "expected_cluster": "abrahamic_monotheism",
    },
    {
        "tradition": "Judaism",
        "source": "Psalms 23",
        "text": "The LORD is my shepherd, I lack nothing. He makes me lie down in green pastures, he leads me beside quiet waters, he refreshes my soul. He guides me along the right paths for his name's sake. Even though I walk through the darkest valley, I will fear no evil, for you are with me; your rod and your staff, they comfort me.",
        "expected_cluster": "abrahamic_monotheism",
    },
    {
        "tradition": "Judaism",
        "source": "Deuteronomy 6:4-9 (Shema)",
        "text": "Hear, O Israel: The LORD our God, the LORD is one. Love the LORD your God with all your heart and with all your soul and with all your strength. These commandments that I give you today are to be on your hearts. Impress them on your children. Talk about them when you sit at home and when you walk along the road, when you lie down and when you get up.",
        "expected_cluster": "abrahamic_monotheism",
    },

    # ─── Abrahamic: Christianity ───
    {
        "tradition": "Christianity",
        "source": "Gospel of John 1:1-5",
        "text": "In the beginning was the Word, and the Word was with God, and the Word was God. He was with God in the beginning. Through him all things were made; without him nothing was made that has been made. In him was life, and that life was the light of all mankind. The light shines in the darkness, and the darkness has not overcome it.",
        "expected_cluster": "abrahamic_monotheism",
    },
    {
        "tradition": "Christianity",
        "source": "Sermon on the Mount (Matthew 5:3-12)",
        "text": "Blessed are the poor in spirit, for theirs is the kingdom of heaven. Blessed are those who mourn, for they will be comforted. Blessed are the meek, for they will inherit the earth. Blessed are those who hunger and thirst for righteousness, for they will be filled. Blessed are the merciful, for they will be shown mercy. Blessed are the pure in heart, for they will see God. Blessed are the peacemakers, for they will be called children of God.",
        "expected_cluster": "abrahamic_monotheism",
    },
    {
        "tradition": "Christianity",
        "source": "Romans 8:38-39",
        "text": "For I am convinced that neither death nor life, neither angels nor demons, neither the present nor the future, nor any powers, neither height nor depth, nor anything else in all creation, will be able to separate us from the love of God that is in Christ Jesus our Lord.",
        "expected_cluster": "abrahamic_monotheism",
    },

    # ─── Abrahamic: Islam ───
    {
        "tradition": "Islam",
        "source": "Quran, Al-Fatiha (1:1-7)",
        "text": "In the name of God, the Most Gracious, the Most Merciful. Praise be to God, Lord of all the worlds. The Most Gracious, the Most Merciful. Master of the Day of Judgment. You alone we worship, and You alone we ask for help. Guide us on the Straight Path, the path of those who have received Your grace; not the path of those who have brought down wrath upon themselves, nor of those who have gone astray.",
        "expected_cluster": "abrahamic_monotheism",
    },
    {
        "tradition": "Islam",
        "source": "Quran, Al-Ikhlas (112:1-4)",
        "text": "Say: He is God, the One. God, the Eternal, the Absolute. He begets not, nor was He begotten. And there is none comparable to Him.",
        "expected_cluster": "abrahamic_monotheism",
    },
    {
        "tradition": "Islam",
        "source": "Quran, Al-Baqarah 2:255 (Ayat al-Kursi)",
        "text": "God! There is no deity except Him, the Ever-Living, the Sustainer of existence. Neither drowsiness overtakes Him nor sleep. To Him belongs whatever is in the heavens and whatever is on the earth. Who is it that can intercede with Him except by His permission? He knows what is before them and what will be after them, and they encompass not a thing of His knowledge except for what He wills. His throne extends over the heavens and the earth, and their preservation tires Him not. And He is the Most High, the Most Great.",
        "expected_cluster": "abrahamic_monotheism",
    },

    # ─── Hinduism ───
    {
        "tradition": "Hinduism",
        "source": "Bhagavad Gita 2:22-24",
        "text": "As a person puts on new garments, giving up old ones, the soul similarly accepts new material bodies, giving up the old and useless ones. The soul can never be cut to pieces by any weapon, nor burned by fire, nor moistened by water, nor withered by the wind. This individual soul is unbreakable and insoluble, and can be neither burned nor dried. It is everlasting, present everywhere, unchangeable, immovable and eternally the same.",
        "expected_cluster": "dharmic",
    },
    {
        "tradition": "Hinduism",
        "source": "Bhagavad Gita 11:32-33 (Vishvarupa)",
        "text": "I am time, the great destroyer of the worlds, and I have come here to destroy all people. With the exception of you, all the soldiers here on both sides will be slain. Therefore get up. Prepare to fight and win glory. Conquer your enemies and enjoy a flourishing kingdom. They are already put to death by My arrangement, and you can be but an instrument in the fight.",
        "expected_cluster": "dharmic",
    },
    {
        "tradition": "Hinduism",
        "source": "Isha Upanishad 1",
        "text": "The Lord is enshrined in the hearts of all. The Lord is the supreme Reality. Rejoice in him through renunciation. Covet nothing. All belongs to the Lord. Thus working may you live a hundred years. Thus alone will you work in real freedom.",
        "expected_cluster": "dharmic",
    },

    # ─── Buddhism ───
    {
        "tradition": "Buddhism",
        "source": "Dhammapada 1-5",
        "text": "All that we are is the result of what we have thought: it is founded on our thoughts, it is made up of our thoughts. If a man speaks or acts with an evil thought, pain follows him, as the wheel follows the foot of the ox that draws the carriage. If a man speaks or acts with a pure thought, happiness follows him, like a shadow that never leaves him.",
        "expected_cluster": "dharmic",
    },
    {
        "tradition": "Buddhism",
        "source": "Heart Sutra",
        "text": "Form is emptiness, emptiness is form. Emptiness is not separate from form, form is not separate from emptiness. Whatever is form is emptiness, whatever is emptiness is form. The same is true for feelings, perceptions, mental formations, and consciousness. All dharmas are marked with emptiness. They are neither produced nor destroyed, neither defiled nor immaculate, neither increasing nor decreasing.",
        "expected_cluster": "dharmic",
    },
    {
        "tradition": "Buddhism",
        "source": "Metta Sutta (Loving-Kindness)",
        "text": "May all beings be happy. May all beings be safe. May all beings be healthy. May all beings live with ease. Whatever living beings there may be, whether they are weak or strong, omitting none, the great or the mighty, medium, short or small, the seen and the unseen, those living near and far away, those born and to-be-born — may all beings be happy.",
        "expected_cluster": "dharmic",
    },

    # ─── Norse/Germanic ───
    {
        "tradition": "Norse",
        "source": "Hávamál (Sayings of the High One) 138-139",
        "text": "I know that I hung on a wind-rocked tree, nine whole nights, with a spear wounded, and to Odin offered, myself to myself; on that tree, of which no one knows from what root it springs. Neither food nor drink was given me. I peered downward, I took up the runes, shrieking I took them, and forthwith back I fell.",
        "expected_cluster": "warrior_polytheism",
    },
    {
        "tradition": "Norse",
        "source": "Völuspá (Prophecy of the Seeress) 56-58",
        "text": "The sun turns black, earth sinks in the sea, the hot stars down from heaven are whirled. Fierce grows the steam and the life-feeding flame, till fire leaps high about heaven itself. She sees arise a second time, earth from the ocean, beautifully green. The cataracts fall, the eagle flies over, hunting fish along the mountain.",
        "expected_cluster": "warrior_polytheism",
    },

    # ─── Greek ───
    {
        "tradition": "Greek",
        "source": "Homeric Hymn to Zeus",
        "text": "I will sing of Zeus, chiefest among the gods and greatest, all-seeing, the lord of all, the fulfiller who whispers words of wisdom to Themis as she sits leaning towards him. Be gracious, all-seeing Son of Kronos, most excellent and great!",
        "expected_cluster": "warrior_polytheism",
    },
    {
        "tradition": "Greek",
        "source": "Orphic Hymn to Gaia",
        "text": "O Goddess, Earth, of Gods and men the source, endued with fertile, all-destroying force; all-parent, bounding, whose prolific powers produce a store of beauteous fruits and flowers. All-various maid, the eternal world's strong base, immortal, blessed, crowned with every grace; from whose wide womb as from an endless root, fruits many-formed, mature, and grateful shoot.",
        "expected_cluster": "warrior_polytheism",
    },

    # ─── Daoism ───
    {
        "tradition": "Daoism",
        "source": "Tao Te Ching, Chapter 1",
        "text": "The Tao that can be told is not the eternal Tao. The name that can be named is not the eternal name. The nameless is the beginning of heaven and earth. The named is the mother of ten thousand things. Ever desireless, one can see the mystery. Ever desiring, one can see the manifestations. These two spring from the same source but differ in name; this appears as darkness. Darkness within darkness. The gate to all mystery.",
        "expected_cluster": "eastern_nontheistic",
    },
    {
        "tradition": "Daoism",
        "source": "Tao Te Ching, Chapter 76",
        "text": "A man is born gentle and weak. At his death he is hard and stiff. Green plants are tender and filled with sap. At their death they are withered and dry. Therefore the stiff and unbending is the disciple of death. The gentle and yielding is the disciple of life. Thus an army without flexibility never wins a battle. A tree that is unbending is easily broken. The hard and strong will fall. The soft and weak will overcome.",
        "expected_cluster": "eastern_nontheistic",
    },

    # ─── Indigenous / Animist ───
    {
        "tradition": "Lakota",
        "source": "Black Elk Speaks",
        "text": "The first peace, which is the most important, is that which comes within the souls of people when they realize their relationship, their oneness with the universe and all its powers, and when they realize at the center of the universe dwells the Great Spirit, and that its center is really everywhere, it is within each of us.",
        "expected_cluster": "animist_indigenous",
    },
    {
        "tradition": "Aboriginal Australian",
        "source": "Dreamtime narrative",
        "text": "In the Dreamtime, the ancestor spirits came up out of the earth and down from the sky to walk on the land. They created everything: the animals, the plants, the rocks, the rivers. They shaped the land and made the sacred places. When they were finished, they went back into the earth, into the sky, into the water. But they are still here. They are in everything. The land is alive with their presence.",
        "expected_cluster": "animist_indigenous",
    },

    # ─── Zoroastrianism ───
    {
        "tradition": "Zoroastrianism",
        "source": "Yasna 30:3-5 (Gathas)",
        "text": "Now the two primal Spirits, who revealed themselves in vision as Twins, are the Better and the Bad, in thought and word and action. And between these two the wise ones chose aright; the foolish did not so. And when these two Spirits came together in the beginning, they created Life and Not-Life, and that at the last Worst Existence shall be to the followers of the Lie, but the Best Existence to him that follows Right.",
        "expected_cluster": "abrahamic_monotheism",
    },
]


# ─── LLM Scoring ──────────────────────────────────────────────────────

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
        "temperature": 0.0,  # deterministic scoring
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
    """Score a religious passage on all 12 theological axes using frontier LLM."""
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
        # Validate
        vec = {}
        for axis in AXES:
            val = float(scores.get(axis, 0.0))
            vec[axis] = max(0.0, min(1.0, val))
        return vec
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  Parse error: {e}")
        print(f"  Raw: {response[:200]}")
        return {axis: 0.0 for axis in AXES}


# ─── Braille Encoding ─────────────────────────────────────────────────

def normalize(vec: dict) -> dict:
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1
    return {a: vec[a] / norm for a in AXES}


def vec_to_array(vec: dict) -> np.ndarray:
    return np.array([vec[a] for a in AXES])


def cosine_sim_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def encode_braille(vec: dict) -> np.ndarray:
    """Encode normalized 12D vector as 72 bits."""
    bits = []
    for axis in AXES:
        v = vec[axis]
        # Polarity: positive active
        bits.append(1 if v > 0.15 else 0)
        # Polarity: strong
        bits.append(1 if v > 0.3 else 0)
        # Polarity: dominant
        bits.append(1 if v > 0.45 else 0)
        # Intensity high
        bits.append(1 if v > 0.25 else 0)
        # Intensity low
        bits.append(1 if v > 0.1 else 0)
        # Rigidity
        bits.append(1 if v > 0.35 else 0)
    return np.array(bits, dtype=np.float32)


def decode_braille(bits: np.ndarray) -> np.ndarray:
    """Decode 72 bits back to approximate 12D vector."""
    vec = []
    for i in range(12):
        b = bits[i*6:(i+1)*6]
        # Reconstruct from bit pattern
        value = (b[0] * 0.15 + b[1] * 0.15 + b[2] * 0.15 +
                 b[3] * 0.15 + b[4] * 0.1 + b[5] * 0.15)
        vec.append(value)
    arr = np.array(vec)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr


def braille_unicode(bits: np.ndarray) -> str:
    chars = []
    offsets = [1, 2, 4, 8, 16, 32]
    for i in range(12):
        cell_bits = bits[i*6:(i+1)*6]
        code = 0x2800
        for j, b in enumerate(cell_bits):
            if int(b):
                code += offsets[j]
        chars.append(chr(code))
    return "".join(chars)


# ─── Baselines ─────────────────────────────────────────────────────────

def random_binary_projection(vec: np.ndarray, proj_matrix: np.ndarray) -> np.ndarray:
    """Random binary projection: project then threshold."""
    projected = proj_matrix @ vec
    return (projected > 0).astype(np.float32)


def pca_quantize(vec: np.ndarray, pca_components: np.ndarray,
                 thresholds: np.ndarray) -> np.ndarray:
    """PCA + uniform quantization to 72 bits."""
    projected = pca_components @ vec
    bits = (projected > thresholds).astype(np.float32)
    return bits


# ─── Main Experiment ──────────────────────────────────────────────────

def main():
    if not API_KEY:
        print("ERROR: No OPENROUTER_API_KEY found")
        return

    print("=" * 60)
    print("REAL EMBEDDINGS EXPERIMENT")
    print(f"Scoring {len(CORPUS)} religious passages on 12 theological axes")
    print(f"Scorer: {SCORER_MODEL}")
    print("=" * 60)

    # ─── Step 1: Score all passages ───────────────────────────────

    print("\n--- Step 1: Scoring passages ---\n")
    embeddings = []

    for i, passage in enumerate(CORPUS):
        print(f"  [{i+1}/{len(CORPUS)}] {passage['tradition']}: {passage['source']}")
        vec = score_passage(passage)
        vec_norm = normalize(vec)
        embeddings.append({
            "tradition": passage["tradition"],
            "source": passage["source"],
            "expected_cluster": passage["expected_cluster"],
            "raw_scores": vec,
            "normalized": vec_norm,
        })
        time.sleep(1.5)  # rate limiting

    print(f"\n  Scored {len(embeddings)} passages.")

    # ─── Step 2: Compute pairwise similarities (continuous) ──────

    print("\n--- Step 2: Pairwise similarities (continuous) ---\n")
    N = len(embeddings)
    continuous_vecs = np.array([vec_to_array(e["normalized"]) for e in embeddings])

    sim_continuous = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            sim_continuous[i, j] = cosine_sim_np(continuous_vecs[i], continuous_vecs[j])

    # ─── Step 3: Braille compression ─────────────────────────────

    print("--- Step 3: Braille compression ---\n")
    braille_bits = []
    braille_decoded = []
    for e in embeddings:
        bits = encode_braille(e["normalized"])
        decoded = decode_braille(bits)
        braille_bits.append(bits)
        braille_decoded.append(decoded)
        sig = braille_unicode(bits)
        cos = cosine_sim_np(continuous_vecs[embeddings.index(e)], decoded)
        print(f"  {e['tradition']:20s} | {e['source'][:30]:30s} | {sig} | cos={cos:.4f}")

    braille_bits = np.array(braille_bits)
    braille_decoded = np.array(braille_decoded)

    sim_braille = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            sim_braille[i, j] = cosine_sim_np(braille_decoded[i], braille_decoded[j])

    # ─── Step 4: Baselines ───────────────────────────────────────

    print("\n--- Step 4: Baselines ---\n")

    # Random binary projection (72 bits)
    np.random.seed(42)
    random_proj = np.random.randn(72, 12) / math.sqrt(12)
    random_bits = np.array([random_binary_projection(v, random_proj) for v in continuous_vecs])
    # Decode random: just use pseudoinverse
    random_decoded = np.array([np.linalg.pinv(random_proj) @ b for b in random_bits])
    for i in range(N):
        n = np.linalg.norm(random_decoded[i])
        if n > 0:
            random_decoded[i] /= n

    sim_random = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            sim_random[i, j] = cosine_sim_np(random_decoded[i], random_decoded[j])

    # PCA + quantize (72 bits)
    # Use SVD of the data as PCA
    mean_vec = continuous_vecs.mean(axis=0)
    centered = continuous_vecs - mean_vec
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # Repeat components to get 72 dimensions (6 thresholds per component)
    pca_bits_list = []
    for vec in continuous_vecs:
        projected = Vt @ (vec - mean_vec)  # 12 PCA components
        bits = []
        for comp_val in projected:
            # 6 bits per component at different thresholds
            for t in [-0.3, -0.1, 0.0, 0.1, 0.3, 0.5]:
                bits.append(1.0 if comp_val > t else 0.0)
        pca_bits_list.append(np.array(bits[:72]))
    pca_bits = np.array(pca_bits_list)

    # Decode PCA bits: reconstruct from thresholds
    pca_decoded = []
    for bits in pca_bits:
        comp_vals = []
        for i in range(12):
            b = bits[i*6:(i+1)*6]
            val = (b.sum() - 3) * 0.2  # rough reconstruction
            comp_vals.append(val)
        reconstructed = np.array(comp_vals) @ Vt + mean_vec
        n = np.linalg.norm(reconstructed)
        if n > 0:
            reconstructed /= n
        pca_decoded.append(reconstructed)
    pca_decoded = np.array(pca_decoded)

    sim_pca = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            sim_pca[i, j] = cosine_sim_np(pca_decoded[i], pca_decoded[j])

    # ─── Step 5: Evaluate ─────────────────────────────────────────

    print("\n--- Step 5: Evaluation ---\n")

    # Metric 1: Reconstruction cosine similarity
    recon_braille = [cosine_sim_np(continuous_vecs[i], braille_decoded[i]) for i in range(N)]
    recon_random = [cosine_sim_np(continuous_vecs[i], random_decoded[i]) for i in range(N)]
    recon_pca = [cosine_sim_np(continuous_vecs[i], pca_decoded[i]) for i in range(N)]

    print(f"  Reconstruction cosine similarity:")
    print(f"    Braille lattice: {np.mean(recon_braille):.4f} ± {np.std(recon_braille):.4f}")
    print(f"    Random binary:   {np.mean(recon_random):.4f} ± {np.std(recon_random):.4f}")
    print(f"    PCA + quantize:  {np.mean(recon_pca):.4f} ± {np.std(recon_pca):.4f}")

    # Metric 2: Pairwise similarity rank correlation (Spearman)
    from scipy.stats import spearmanr

    # Extract upper triangle of similarity matrices
    triu_idx = np.triu_indices(N, k=1)
    cont_pairs = sim_continuous[triu_idx]
    braille_pairs = sim_braille[triu_idx]
    random_pairs = sim_random[triu_idx]
    pca_pairs = sim_pca[triu_idx]

    rho_braille, p_braille = spearmanr(cont_pairs, braille_pairs)
    rho_random, p_random = spearmanr(cont_pairs, random_pairs)
    rho_pca, p_pca = spearmanr(cont_pairs, pca_pairs)

    print(f"\n  Pairwise similarity rank correlation (Spearman ρ):")
    print(f"    Braille lattice: ρ={rho_braille:.4f} (p={p_braille:.2e})")
    print(f"    Random binary:   ρ={rho_random:.4f} (p={p_random:.2e})")
    print(f"    PCA + quantize:  ρ={rho_pca:.4f} (p={p_pca:.2e})")

    # Metric 3: Cluster preservation
    # Do within-cluster similarities remain higher than between-cluster?
    clusters = {}
    for i, e in enumerate(embeddings):
        c = e["expected_cluster"]
        if c not in clusters:
            clusters[c] = []
        clusters[c].append(i)

    def cluster_separation(sim_matrix):
        within = []
        between = []
        for c, indices in clusters.items():
            for i, j in combinations(indices, 2):
                within.append(sim_matrix[i, j])
            for c2, indices2 in clusters.items():
                if c2 <= c:
                    continue
                for i in indices:
                    for j in indices2:
                        between.append(sim_matrix[i, j])
        return np.mean(within), np.mean(between), np.mean(within) - np.mean(between)

    w_cont, b_cont, sep_cont = cluster_separation(sim_continuous)
    w_braille, b_braille, sep_braille = cluster_separation(sim_braille)
    w_random, b_random, sep_random = cluster_separation(sim_random)
    w_pca, b_pca, sep_pca = cluster_separation(sim_pca)

    print(f"\n  Cluster separation (within - between):")
    print(f"    Continuous:      within={w_cont:.4f}, between={b_cont:.4f}, gap={sep_cont:.4f}")
    print(f"    Braille lattice: within={w_braille:.4f}, between={b_braille:.4f}, gap={sep_braille:.4f}")
    print(f"    Random binary:   within={w_random:.4f}, between={b_random:.4f}, gap={sep_random:.4f}")
    print(f"    PCA + quantize:  within={w_pca:.4f}, between={b_pca:.4f}, gap={sep_pca:.4f}")

    # Metric 4: Specific theological predictions
    print(f"\n  Theological predictions (continuous vs braille):")
    tradition_centroids = {}
    tradition_centroids_braille = {}
    for e, cv, bv in zip(embeddings, continuous_vecs, braille_decoded):
        t = e["tradition"]
        if t not in tradition_centroids:
            tradition_centroids[t] = []
            tradition_centroids_braille[t] = []
        tradition_centroids[t].append(cv)
        tradition_centroids_braille[t].append(bv)

    for t in tradition_centroids:
        tradition_centroids[t] = np.mean(tradition_centroids[t], axis=0)
        tradition_centroids_braille[t] = np.mean(tradition_centroids_braille[t], axis=0)

    traditions = sorted(tradition_centroids.keys())
    print(f"\n  Tradition similarity matrix (continuous):")
    print(f"  {'':20s}", end="")
    for t in traditions:
        print(f" {t[:8]:>8s}", end="")
    print()
    for t1 in traditions:
        print(f"  {t1:20s}", end="")
        for t2 in traditions:
            s = cosine_sim_np(tradition_centroids[t1], tradition_centroids[t2])
            print(f" {s:8.4f}", end="")
        print()

    print(f"\n  Tradition similarity matrix (braille-compressed):")
    print(f"  {'':20s}", end="")
    for t in traditions:
        print(f" {t[:8]:>8s}", end="")
    print()
    for t1 in traditions:
        print(f"  {t1:20s}", end="")
        for t2 in traditions:
            s = cosine_sim_np(tradition_centroids_braille[t1], tradition_centroids_braille[t2])
            print(f" {s:8.4f}", end="")
        print()

    # Key predictions to test
    predictions = [
        ("Judaism and Christianity should be close",
         "Judaism", "Christianity", ">", 0.85),
        ("Judaism and Islam should be close",
         "Judaism", "Islam", ">", 0.85),
        ("Christianity and Islam should be close",
         "Christianity", "Islam", ">", 0.85),
        ("Hinduism and Buddhism should be close",
         "Hinduism", "Buddhism", ">", 0.8),
        ("Judaism and Hinduism should be farther",
         "Judaism", "Hinduism", "<", 0.85),
        ("Norse and Greek should be close",
         "Norse", "Greek", ">", 0.8),
        ("Daoism and Buddhism should be somewhat close",
         "Daoism", "Buddhism", ">", 0.7),
    ]

    print(f"\n  Theological predictions:")
    for desc, t1, t2, op, threshold in predictions:
        if t1 in tradition_centroids and t2 in tradition_centroids:
            sim_c = cosine_sim_np(tradition_centroids[t1], tradition_centroids[t2])
            sim_b = cosine_sim_np(tradition_centroids_braille[t1], tradition_centroids_braille[t2])
            passed_c = (sim_c > threshold) if op == ">" else (sim_c < threshold)
            passed_b = (sim_b > threshold) if op == ">" else (sim_b < threshold)
            print(f"    {desc}")
            print(f"      Continuous: {sim_c:.4f} {'✓' if passed_c else '✗'}")
            print(f"      Braille:    {sim_b:.4f} {'✓' if passed_b else '✗'}")

    # ─── Save Results ─────────────────────────────────────────────

    output = {
        "experiment": "real_embeddings_braille_bottleneck",
        "scorer_model": SCORER_MODEL,
        "num_passages": len(CORPUS),
        "num_traditions": len(tradition_centroids),
        "embeddings": [
            {
                "tradition": e["tradition"],
                "source": e["source"],
                "expected_cluster": e["expected_cluster"],
                "raw_scores": e["raw_scores"],
                "braille_signature": braille_unicode(braille_bits[i]),
                "reconstruction_cosine_sim": float(recon_braille[i]),
            }
            for i, e in enumerate(embeddings)
        ],
        "metrics": {
            "reconstruction_cosine_sim": {
                "braille": {"mean": float(np.mean(recon_braille)), "std": float(np.std(recon_braille))},
                "random_binary": {"mean": float(np.mean(recon_random)), "std": float(np.std(recon_random))},
                "pca_quantize": {"mean": float(np.mean(recon_pca)), "std": float(np.std(recon_pca))},
            },
            "spearman_rank_correlation": {
                "braille": {"rho": float(rho_braille), "p": float(p_braille)},
                "random_binary": {"rho": float(rho_random), "p": float(p_random)},
                "pca_quantize": {"rho": float(rho_pca), "p": float(p_pca)},
            },
            "cluster_separation": {
                "continuous": {"within": float(w_cont), "between": float(b_cont), "gap": float(sep_cont)},
                "braille": {"within": float(w_braille), "between": float(b_braille), "gap": float(sep_braille)},
                "random_binary": {"within": float(w_random), "between": float(b_random), "gap": float(sep_random)},
                "pca_quantize": {"within": float(w_pca), "between": float(b_pca), "gap": float(sep_pca)},
            },
        },
    }

    out_path = Path(__file__).parent / "real_embeddings_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    # ─── Markdown Report ──────────────────────────────────────────

    lines = [
        "# Real Embeddings Experiment — Braille Bottleneck on Religious Texts",
        "",
        "## Design",
        "",
        f"**Scorer:** `{SCORER_MODEL}` scores {len(CORPUS)} canonical religious passages on 12 theological axes.",
        f"**Traditions:** {', '.join(sorted(tradition_centroids.keys()))}",
        f"**Bottleneck:** 72-bit braille lattice (6 bits × 12 axes)",
        f"**Baselines:** Random binary projection (72 bits), PCA + uniform quantization (72 bits)",
        "",
        "---",
        "",
        "## Reconstruction Quality",
        "",
        "| Method | Mean Cosine Sim | Std |",
        "|--------|----------------|-----|",
        f"| **Braille lattice** | **{np.mean(recon_braille):.4f}** | {np.std(recon_braille):.4f} |",
        f"| Random binary | {np.mean(recon_random):.4f} | {np.std(recon_random):.4f} |",
        f"| PCA + quantize | {np.mean(recon_pca):.4f} | {np.std(recon_pca):.4f} |",
        "",
        "## Pairwise Similarity Preservation (Spearman ρ)",
        "",
        "| Method | ρ | p-value |",
        "|--------|---|---------|",
        f"| **Braille lattice** | **{rho_braille:.4f}** | {p_braille:.2e} |",
        f"| Random binary | {rho_random:.4f} | {p_random:.2e} |",
        f"| PCA + quantize | {rho_pca:.4f} | {p_pca:.2e} |",
        "",
        "## Cluster Separation (within − between)",
        "",
        "| Method | Within | Between | Gap |",
        "|--------|--------|---------|-----|",
        f"| Continuous | {w_cont:.4f} | {b_cont:.4f} | {sep_cont:.4f} |",
        f"| **Braille lattice** | **{w_braille:.4f}** | **{b_braille:.4f}** | **{sep_braille:.4f}** |",
        f"| Random binary | {w_random:.4f} | {b_random:.4f} | {sep_random:.4f} |",
        f"| PCA + quantize | {w_pca:.4f} | {b_pca:.4f} | {sep_pca:.4f} |",
        "",
        "---",
        "",
        "*Generated by the Real Embeddings Pipeline — Gods as Centroids*",
    ]

    report_path = Path(__file__).parent / "real_embeddings_results.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Report saved to {report_path}")


if __name__ == "__main__":
    main()
