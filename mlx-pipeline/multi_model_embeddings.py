#!/usr/bin/env python3
"""
Multi-Model Theological Embeddings — Inter-LLM Variance Analysis

Score the same 24 religious passages with multiple frontier LLMs.
Measure:
1. Inter-model agreement: do LLMs agree on theological structure?
2. Per-axis variance: which theological axes are contested vs consensual?
3. Cluster stability: do all models produce the same tradition clusters?
4. Braille consensus: does the 72-bit bottleneck preserve the multi-model consensus?
5. Braiding: majority-vote across model embeddings → does the braid outperform individuals?

This is semantic braiding with REAL heterogeneous encoders (frontier LLMs).
"""

import json
import math
import os
import time
import numpy as np
import httpx
from pathlib import Path
from itertools import combinations
from scipy.stats import spearmanr

# ─── Config ───────────────────────────────────────────────────────────

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not API_KEY:
    env_path = Path(__file__).parent.parent / "web" / ".env.local"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                API_KEY = line.split("=", 1)[1].strip()

API_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = [
    "anthropic/claude-sonnet-4",
    "openai/gpt-4o",
    "google/gemini-2.0-flash-001",
    "meta-llama/llama-3.3-70b-instruct",
]

AXES = [
    "authority", "transcendence", "care", "justice", "wisdom", "power",
    "fertility", "war", "death", "creation", "nature", "order"
]

# Same corpus as real_embeddings.py
CORPUS = [
    {"tradition": "Judaism", "source": "Torah (Exodus 20:1-6)", "expected_cluster": "abrahamic",
     "text": "I am the LORD your God, who brought you out of Egypt, out of the land of slavery. You shall have no other gods before me. You shall not make for yourself an image in the form of anything in heaven above or on the earth beneath or in the waters below. You shall not bow down to them or worship them; for I, the LORD your God, am a jealous God, punishing the children for the sin of the parents to the third and fourth generation of those who hate me, but showing love to a thousand generations of those who love me and keep my commandments."},
    {"tradition": "Judaism", "source": "Psalms 23", "expected_cluster": "abrahamic",
     "text": "The LORD is my shepherd, I lack nothing. He makes me lie down in green pastures, he leads me beside quiet waters, he refreshes my soul. He guides me along the right paths for his name's sake. Even though I walk through the darkest valley, I will fear no evil, for you are with me; your rod and your staff, they comfort me."},
    {"tradition": "Judaism", "source": "Deuteronomy 6:4-9 (Shema)", "expected_cluster": "abrahamic",
     "text": "Hear, O Israel: The LORD our God, the LORD is one. Love the LORD your God with all your heart and with all your soul and with all your strength. These commandments that I give you today are to be on your hearts. Impress them on your children. Talk about them when you sit at home and when you walk along the road, when you lie down and when you get up."},
    {"tradition": "Christianity", "source": "Gospel of John 1:1-5", "expected_cluster": "abrahamic",
     "text": "In the beginning was the Word, and the Word was with God, and the Word was God. He was with God in the beginning. Through him all things were made; without him nothing was made that has been made. In him was life, and that life was the light of all mankind. The light shines in the darkness, and the darkness has not overcome it."},
    {"tradition": "Christianity", "source": "Sermon on the Mount (Matthew 5:3-12)", "expected_cluster": "abrahamic",
     "text": "Blessed are the poor in spirit, for theirs is the kingdom of heaven. Blessed are those who mourn, for they will be comforted. Blessed are the meek, for they will inherit the earth. Blessed are those who hunger and thirst for righteousness, for they will be filled. Blessed are the merciful, for they will be shown mercy. Blessed are the pure in heart, for they will see God. Blessed are the peacemakers, for they will be called children of God."},
    {"tradition": "Christianity", "source": "Romans 8:38-39", "expected_cluster": "abrahamic",
     "text": "For I am convinced that neither death nor life, neither angels nor demons, neither the present nor the future, nor any powers, neither height nor depth, nor anything else in all creation, will be able to separate us from the love of God that is in Christ Jesus our Lord."},
    {"tradition": "Islam", "source": "Quran, Al-Fatiha (1:1-7)", "expected_cluster": "abrahamic",
     "text": "In the name of God, the Most Gracious, the Most Merciful. Praise be to God, Lord of all the worlds. The Most Gracious, the Most Merciful. Master of the Day of Judgment. You alone we worship, and You alone we ask for help. Guide us on the Straight Path, the path of those who have received Your grace; not the path of those who have brought down wrath upon themselves, nor of those who have gone astray."},
    {"tradition": "Islam", "source": "Quran, Al-Ikhlas (112:1-4)", "expected_cluster": "abrahamic",
     "text": "Say: He is God, the One. God, the Eternal, the Absolute. He begets not, nor was He begotten. And there is none comparable to Him."},
    {"tradition": "Islam", "source": "Quran, Al-Baqarah 2:255 (Ayat al-Kursi)", "expected_cluster": "abrahamic",
     "text": "God! There is no deity except Him, the Ever-Living, the Sustainer of existence. Neither drowsiness overtakes Him nor sleep. To Him belongs whatever is in the heavens and whatever is on the earth. Who is it that can intercede with Him except by His permission? He knows what is before them and what will be after them, and they encompass not a thing of His knowledge except for what He wills. His throne extends over the heavens and the earth, and their preservation tires Him not. And He is the Most High, the Most Great."},
    {"tradition": "Hinduism", "source": "Bhagavad Gita 2:22-24", "expected_cluster": "dharmic",
     "text": "As a person puts on new garments, giving up old ones, the soul similarly accepts new material bodies, giving up the old and useless ones. The soul can never be cut to pieces by any weapon, nor burned by fire, nor moistened by water, nor withered by the wind. This individual soul is unbreakable and insoluble, and can be neither burned nor dried. It is everlasting, present everywhere, unchangeable, immovable and eternally the same."},
    {"tradition": "Hinduism", "source": "Bhagavad Gita 11:32-33 (Vishvarupa)", "expected_cluster": "dharmic",
     "text": "I am time, the great destroyer of the worlds, and I have come here to destroy all people. With the exception of you, all the soldiers here on both sides will be slain. Therefore get up. Prepare to fight and win glory. Conquer your enemies and enjoy a flourishing kingdom. They are already put to death by My arrangement, and you can be but an instrument in the fight."},
    {"tradition": "Hinduism", "source": "Isha Upanishad 1", "expected_cluster": "dharmic",
     "text": "The Lord is enshrined in the hearts of all. The Lord is the supreme Reality. Rejoice in him through renunciation. Covet nothing. All belongs to the Lord. Thus working may you live a hundred years. Thus alone will you work in real freedom."},
    {"tradition": "Buddhism", "source": "Dhammapada 1-5", "expected_cluster": "dharmic",
     "text": "All that we are is the result of what we have thought: it is founded on our thoughts, it is made up of our thoughts. If a man speaks or acts with an evil thought, pain follows him, as the wheel follows the foot of the ox that draws the carriage. If a man speaks or acts with a pure thought, happiness follows him, like a shadow that never leaves him."},
    {"tradition": "Buddhism", "source": "Heart Sutra", "expected_cluster": "dharmic",
     "text": "Form is emptiness, emptiness is form. Emptiness is not separate from form, form is not separate from emptiness. Whatever is form is emptiness, whatever is emptiness is form. The same is true for feelings, perceptions, mental formations, and consciousness. All dharmas are marked with emptiness. They are neither produced nor destroyed, neither defiled nor immaculate, neither increasing nor decreasing."},
    {"tradition": "Buddhism", "source": "Metta Sutta (Loving-Kindness)", "expected_cluster": "dharmic",
     "text": "May all beings be happy. May all beings be safe. May all beings be healthy. May all beings live with ease. Whatever living beings there may be, whether they are weak or strong, omitting none, the great or the mighty, medium, short or small, the seen and the unseen, those living near and far away, those born and to-be-born — may all beings be happy."},
    {"tradition": "Norse", "source": "Hávamál 138-139", "expected_cluster": "warrior_poly",
     "text": "I know that I hung on a wind-rocked tree, nine whole nights, with a spear wounded, and to Odin offered, myself to myself; on that tree, of which no one knows from what root it springs. Neither food nor drink was given me. I peered downward, I took up the runes, shrieking I took them, and forthwith back I fell."},
    {"tradition": "Norse", "source": "Völuspá 56-58", "expected_cluster": "warrior_poly",
     "text": "The sun turns black, earth sinks in the sea, the hot stars down from heaven are whirled. Fierce grows the steam and the life-feeding flame, till fire leaps high about heaven itself. She sees arise a second time, earth from the ocean, beautifully green. The cataracts fall, the eagle flies over, hunting fish along the mountain."},
    {"tradition": "Greek", "source": "Homeric Hymn to Zeus", "expected_cluster": "warrior_poly",
     "text": "I will sing of Zeus, chiefest among the gods and greatest, all-seeing, the lord of all, the fulfiller who whispers words of wisdom to Themis as she sits leaning towards him. Be gracious, all-seeing Son of Kronos, most excellent and great!"},
    {"tradition": "Greek", "source": "Orphic Hymn to Gaia", "expected_cluster": "warrior_poly",
     "text": "O Goddess, Earth, of Gods and men the source, endued with fertile, all-destroying force; all-parent, bounding, whose prolific powers produce a store of beauteous fruits and flowers. All-various maid, the eternal world's strong base, immortal, blessed, crowned with every grace; from whose wide womb as from an endless root, fruits many-formed, mature, and grateful shoot."},
    {"tradition": "Daoism", "source": "Tao Te Ching, Chapter 1", "expected_cluster": "eastern_non",
     "text": "The Tao that can be told is not the eternal Tao. The name that can be named is not the eternal name. The nameless is the beginning of heaven and earth. The named is the mother of ten thousand things. Ever desireless, one can see the mystery. Ever desiring, one can see the manifestations. These two spring from the same source but differ in name; this appears as darkness. Darkness within darkness. The gate to all mystery."},
    {"tradition": "Daoism", "source": "Tao Te Ching, Chapter 76", "expected_cluster": "eastern_non",
     "text": "A man is born gentle and weak. At his death he is hard and stiff. Green plants are tender and filled with sap. At their death they are withered and dry. Therefore the stiff and unbending is the disciple of death. The gentle and yielding is the disciple of life. Thus an army without flexibility never wins a battle. A tree that is unbending is easily broken. The hard and strong will fall. The soft and weak will overcome."},
    {"tradition": "Lakota", "source": "Black Elk Speaks", "expected_cluster": "animist",
     "text": "The first peace, which is the most important, is that which comes within the souls of people when they realize their relationship, their oneness with the universe and all its powers, and when they realize at the center of the universe dwells the Great Spirit, and that its center is really everywhere, it is within each of us."},
    {"tradition": "Aboriginal", "source": "Dreamtime narrative", "expected_cluster": "animist",
     "text": "In the Dreamtime, the ancestor spirits came up out of the earth and down from the sky to walk on the land. They created everything: the animals, the plants, the rocks, the rivers. They shaped the land and made the sacred places. When they were finished, they went back into the earth, into the sky, into the water. But they are still here. They are in everything. The land is alive with their presence."},
]


# ─── LLM Calls ────────────────────────────────────────────────────────

def call_llm(model: str, messages: list, max_tokens: int = 400) -> str:
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
            print(f"    API error ({model}, attempt {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(3 ** (attempt + 1))
    return "[ERROR]"


SCORE_PROMPT = """You are a computational theology engine. Score the following religious text on exactly 12 theological axes. Each score must be a float between 0.0 and 1.0.

TEXT: "{text}"
SOURCE: {source} ({tradition})

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
{{"authority": 0.8, "transcendence": 0.6, "care": 0.3, "justice": 0.5, "wisdom": 0.4, "power": 0.7, "fertility": 0.2, "war": 0.1, "death": 0.3, "creation": 0.6, "nature": 0.2, "order": 0.5}}"""


def score_passage(model: str, passage: dict) -> dict:
    prompt = SCORE_PROMPT.format(
        text=passage["text"], source=passage["source"], tradition=passage["tradition"]
    )
    response = call_llm(model, [{"role": "user", "content": prompt}], max_tokens=300)
    try:
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        scores = json.loads(response)
        return {axis: max(0.0, min(1.0, float(scores.get(axis, 0.0)))) for axis in AXES}
    except (json.JSONDecodeError, ValueError) as e:
        print(f"    Parse error ({model}): {e}")
        return {axis: 0.0 for axis in AXES}


# ─── Braille Encoding ─────────────────────────────────────────────────

def normalize(vec: dict) -> dict:
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1
    return {a: vec[a] / norm for a in AXES}

def vec_to_array(vec: dict) -> np.ndarray:
    return np.array([vec[a] for a in AXES])

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def encode_braille(vec: dict) -> np.ndarray:
    bits = []
    for axis in AXES:
        v = vec[axis]
        bits.extend([
            1 if v > 0.15 else 0,
            1 if v > 0.3 else 0,
            1 if v > 0.45 else 0,
            1 if v > 0.25 else 0,
            1 if v > 0.1 else 0,
            1 if v > 0.35 else 0,
        ])
    return np.array(bits, dtype=np.float32)

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


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    if not API_KEY:
        print("ERROR: No OPENROUTER_API_KEY found")
        return

    print("=" * 70)
    print("MULTI-MODEL THEOLOGICAL EMBEDDINGS")
    print(f"Models: {', '.join(m.split('/')[-1] for m in MODELS)}")
    print(f"Passages: {len(CORPUS)}")
    print("=" * 70)

    # ─── Step 1: Score all passages with all models ───────────────

    # model_name -> passage_idx -> {axis: score}
    all_scores = {m: [] for m in MODELS}

    for pi, passage in enumerate(CORPUS):
        print(f"\n  [{pi+1}/{len(CORPUS)}] {passage['tradition']}: {passage['source'][:40]}")
        for model in MODELS:
            short = model.split("/")[-1]
            print(f"    Scoring with {short}...", end=" ", flush=True)
            scores = score_passage(model, passage)
            all_scores[model].append(normalize(scores))
            print("done")
            time.sleep(1.0)

    N = len(CORPUS)
    M = len(MODELS)

    # Convert to arrays: (models, passages, 12)
    score_arrays = {}
    for model in MODELS:
        score_arrays[model] = np.array([vec_to_array(s) for s in all_scores[model]])

    # ─── Step 2: Inter-model agreement ────────────────────────────

    print("\n" + "=" * 70)
    print("INTER-MODEL AGREEMENT")
    print("=" * 70)

    # Pairwise model correlation (across all passages × axes)
    print("\n  Pairwise model correlation (Pearson, all scores flattened):")
    model_pairs_corr = {}
    for mi, mj in combinations(range(M), 2):
        m1, m2 = MODELS[mi], MODELS[mj]
        flat1 = score_arrays[m1].flatten()
        flat2 = score_arrays[m2].flatten()
        corr = np.corrcoef(flat1, flat2)[0, 1]
        model_pairs_corr[(m1, m2)] = corr
        print(f"    {m1.split('/')[-1]:30s} × {m2.split('/')[-1]:30s}: r={corr:.4f}")

    mean_corr = np.mean(list(model_pairs_corr.values()))
    print(f"\n  Mean inter-model correlation: {mean_corr:.4f}")

    # Per-axis variance across models
    print("\n  Per-axis variance (mean across passages):")
    print(f"  {'Axis':15s} {'Mean':>8s} {'Std':>8s} {'CV':>8s}  Interpretation")
    axis_stats = {}
    for ai, axis in enumerate(AXES):
        # For each passage, get the scores from all models on this axis
        axis_scores = np.array([[score_arrays[m][pi, ai] for m in MODELS] for pi in range(N)])
        mean_val = axis_scores.mean()
        mean_std = axis_scores.std(axis=1).mean()  # mean std across passages
        cv = mean_std / (mean_val + 1e-8)
        axis_stats[axis] = {"mean": float(mean_val), "std": float(mean_std), "cv": float(cv)}
        interp = "CONSENSUAL" if cv < 0.15 else ("MODERATE" if cv < 0.3 else "CONTESTED")
        print(f"  {axis:15s} {mean_val:8.4f} {mean_std:8.4f} {cv:8.4f}  {interp}")

    # ─── Step 3: Pairwise similarity preservation ─────────────────

    print("\n" + "=" * 70)
    print("SIMILARITY STRUCTURE PRESERVATION")
    print("=" * 70)

    # For each model, compute pairwise similarity matrix
    sim_matrices = {}
    for model in MODELS:
        sim = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                sim[i, j] = cosine_sim(score_arrays[model][i], score_arrays[model][j])
        sim_matrices[model] = sim

    # Consensus embedding: mean across models
    consensus_vecs = np.mean([score_arrays[m] for m in MODELS], axis=0)
    # Normalize each
    for i in range(N):
        n = np.linalg.norm(consensus_vecs[i])
        if n > 0:
            consensus_vecs[i] /= n

    sim_consensus = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            sim_consensus[i, j] = cosine_sim(consensus_vecs[i], consensus_vecs[j])

    # Rank correlation between each model's similarity matrix and consensus
    triu = np.triu_indices(N, k=1)
    consensus_pairs = sim_consensus[triu]

    print("\n  Spearman ρ (each model's similarity structure vs consensus):")
    for model in MODELS:
        model_pairs = sim_matrices[model][triu]
        rho, p = spearmanr(consensus_pairs, model_pairs)
        print(f"    {model.split('/')[-1]:30s}: ρ={rho:.4f} (p={p:.2e})")

    # ─── Step 4: Braille compression of each model ────────────────

    print("\n" + "=" * 70)
    print("BRAILLE COMPRESSION PER MODEL")
    print("=" * 70)

    braille_per_model = {}
    for model in MODELS:
        bits_list = []
        for pi in range(N):
            vec_dict = all_scores[model][pi]
            bits = encode_braille(vec_dict)
            bits_list.append(bits)
        braille_per_model[model] = np.array(bits_list)

    # Braille consensus: majority vote across models at each bit position
    braille_consensus = np.zeros((N, 72))
    for pi in range(N):
        all_bits = np.array([braille_per_model[m][pi] for m in MODELS])
        braille_consensus[pi] = (all_bits.mean(axis=0) > 0.5).astype(np.float32)

    # Bit agreement rate between models
    print("\n  Bit agreement rate (pairwise, averaged over passages):")
    bit_agreements = []
    for mi, mj in combinations(range(M), 2):
        m1, m2 = MODELS[mi], MODELS[mj]
        agree = np.mean([
            np.mean(braille_per_model[m1][pi] == braille_per_model[m2][pi])
            for pi in range(N)
        ])
        bit_agreements.append(agree)
        print(f"    {m1.split('/')[-1]:30s} × {m2.split('/')[-1]:30s}: {agree:.4f}")
    mean_bit_agree = np.mean(bit_agreements)
    print(f"\n  Mean bit agreement: {mean_bit_agree:.4f}")

    # Braille signatures for consensus
    print("\n  Consensus braille signatures:")
    for pi in range(N):
        sig = braille_unicode(braille_consensus[pi])
        print(f"    {CORPUS[pi]['tradition']:15s} {CORPUS[pi]['source'][:35]:35s} {sig}")

    # ─── Step 5: Does braille consensus preserve cluster structure? ─

    print("\n" + "=" * 70)
    print("CLUSTER PRESERVATION THROUGH BRAILLE CONSENSUS")
    print("=" * 70)

    # Hamming similarity for braille
    def hamming_sim(a, b):
        return 1.0 - np.mean(np.abs(a - b))

    clusters = {}
    for pi, p in enumerate(CORPUS):
        c = p["expected_cluster"]
        if c not in clusters:
            clusters[c] = []
        clusters[c].append(pi)

    def cluster_separation_hamming(bits_matrix):
        within, between = [], []
        for c, indices in clusters.items():
            for i, j in combinations(indices, 2):
                within.append(hamming_sim(bits_matrix[i], bits_matrix[j]))
            for c2, indices2 in clusters.items():
                if c2 <= c:
                    continue
                for i in indices:
                    for j in indices2:
                        between.append(hamming_sim(bits_matrix[i], bits_matrix[j]))
        return np.mean(within), np.mean(between), np.mean(within) - np.mean(between)

    print("\n  Cluster separation (Hamming similarity):")
    print(f"  {'Source':30s} {'Within':>8s} {'Between':>8s} {'Gap':>8s}")
    for model in MODELS:
        w, b, g = cluster_separation_hamming(braille_per_model[model])
        print(f"  {model.split('/')[-1]:30s} {w:8.4f} {b:8.4f} {g:8.4f}")
    w_con, b_con, g_con = cluster_separation_hamming(braille_consensus)
    print(f"  {'CONSENSUS (majority vote)':30s} {w_con:8.4f} {b_con:8.4f} {g_con:8.4f}")

    # Also measure with cosine on continuous consensus
    def cluster_separation_cosine(vecs):
        within, between = [], []
        for c, indices in clusters.items():
            for i, j in combinations(indices, 2):
                within.append(cosine_sim(vecs[i], vecs[j]))
            for c2, indices2 in clusters.items():
                if c2 <= c:
                    continue
                for i in indices:
                    for j in indices2:
                        between.append(cosine_sim(vecs[i], vecs[j]))
        return np.mean(within), np.mean(between), np.mean(within) - np.mean(between)

    print(f"\n  Cluster separation (cosine, continuous):")
    print(f"  {'Source':30s} {'Within':>8s} {'Between':>8s} {'Gap':>8s}")
    for model in MODELS:
        w, b, g = cluster_separation_cosine(score_arrays[model])
        print(f"  {model.split('/')[-1]:30s} {w:8.4f} {b:8.4f} {g:8.4f}")
    w_con, b_con, g_con = cluster_separation_cosine(consensus_vecs)
    print(f"  {'CONSENSUS (mean)':30s} {w_con:8.4f} {b_con:8.4f} {g_con:8.4f}")

    # ─── Step 6: Tradition similarity (consensus) ─────────────────

    print("\n" + "=" * 70)
    print("TRADITION SIMILARITY (CONSENSUS EMBEDDINGS)")
    print("=" * 70)

    tradition_centroids = {}
    for pi, p in enumerate(CORPUS):
        t = p["tradition"]
        if t not in tradition_centroids:
            tradition_centroids[t] = []
        tradition_centroids[t].append(consensus_vecs[pi])
    for t in tradition_centroids:
        tradition_centroids[t] = np.mean(tradition_centroids[t], axis=0)

    traditions = sorted(tradition_centroids.keys())
    print(f"\n  {'':15s}", end="")
    for t in traditions:
        print(f" {t[:8]:>8s}", end="")
    print()
    for t1 in traditions:
        print(f"  {t1:15s}", end="")
        for t2 in traditions:
            s = cosine_sim(tradition_centroids[t1], tradition_centroids[t2])
            print(f" {s:8.4f}", end="")
        print()

    # ─── Save Results ─────────────────────────────────────────────

    output = {
        "experiment": "multi_model_theological_embeddings",
        "models": MODELS,
        "num_passages": N,
        "inter_model_correlation": {
            f"{m1.split('/')[-1]}_x_{m2.split('/')[-1]}": float(v)
            for (m1, m2), v in model_pairs_corr.items()
        },
        "mean_inter_model_correlation": float(mean_corr),
        "per_axis_stats": axis_stats,
        "mean_bit_agreement": float(mean_bit_agree),
        "embeddings": [
            {
                "tradition": CORPUS[pi]["tradition"],
                "source": CORPUS[pi]["source"],
                "expected_cluster": CORPUS[pi]["expected_cluster"],
                "scores_per_model": {
                    m.split("/")[-1]: {a: float(all_scores[m][pi][a]) for a in AXES}
                    for m in MODELS
                },
                "consensus_braille": braille_unicode(braille_consensus[pi]),
            }
            for pi in range(N)
        ],
    }

    out_path = Path(__file__).parent / "multi_model_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    # ─── Markdown Report ──────────────────────────────────────────

    contested = sorted(axis_stats.items(), key=lambda x: -x[1]["cv"])
    consensual = sorted(axis_stats.items(), key=lambda x: x[1]["cv"])

    lines = [
        "# Multi-Model Theological Embeddings — Inter-LLM Variance",
        "",
        "## Design",
        "",
        f"**Models:** {', '.join(f'`{m}`' for m in MODELS)}",
        f"**Passages:** {N} canonical texts from {len(set(p['tradition'] for p in CORPUS))} traditions",
        f"**Task:** Score each passage on 12 theological axes (temperature=0)",
        "",
        "---",
        "",
        "## Inter-Model Agreement",
        "",
        f"**Mean Pearson correlation (all scores):** {mean_corr:.4f}",
        "",
        "### Most Consensual Axes (models agree)",
        "",
    ]
    for axis, stats in consensual[:4]:
        lines.append(f"- **{axis}**: CV={stats['cv']:.4f}")
    lines.extend([
        "",
        "### Most Contested Axes (models disagree)",
        "",
    ])
    for axis, stats in contested[:4]:
        lines.append(f"- **{axis}**: CV={stats['cv']:.4f}")

    lines.extend([
        "",
        "## Braille Consensus",
        "",
        f"**Mean bit agreement between models:** {mean_bit_agree:.4f}",
        "",
        "---",
        "",
        "*Generated by the Multi-Model Theological Embeddings Pipeline — Gods as Centroids*",
    ])

    report_path = Path(__file__).parent / "multi_model_results.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Report saved to {report_path}")


if __name__ == "__main__":
    main()
