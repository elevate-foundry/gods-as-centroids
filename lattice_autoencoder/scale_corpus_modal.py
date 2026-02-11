#!/usr/bin/env python3
"""
Scale Corpus via Web Scraping + Modal LLM Scoring
===================================================

Two-stage pipeline:
  Stage 1: Scrape sacred texts from public domain sources (sacred-texts.com, etc.)
  Stage 2: Score each passage with 4 LLMs in parallel on Modal

Target: ~2000 passages across 37+ traditions (up from 126)

Usage:
  # Stage 1: Scrape and chunk passages (local, fast)
  python lattice_autoencoder/scale_corpus_modal.py --stage scrape

  # Stage 2: Score passages with LLMs on Modal (parallel GPU)
  modal run lattice_autoencoder/scale_corpus_modal.py

  # Both stages
  python lattice_autoencoder/scale_corpus_modal.py --stage scrape
  modal run lattice_autoencoder/scale_corpus_modal.py
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════
# TRADITION → SOURCE URLS
# ═══════════════════════════════════════════════════════════════════════
# Public domain texts from sacred-texts.com and other open sources.
# Each entry: (tradition, source_name, url, css_selector_or_tag)

# ═══════════════════════════════════════════════════════════════════════
# VERIFIED SOURCES — Gutenberg plain text + APIs (all return 200)
# ═══════════════════════════════════════════════════════════════════════
# Format: (tradition, source_name, url, type)
# type: "gutenberg" = plain text, "quran_api" = JSON API, "kjv_book" = KJV book name

# KJV Bible books → tradition mapping (Gutenberg ID 10)
# We'll extract specific books from the single KJV file
KJV_BOOKS = {
    "Judaism": [
        "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
        "Psalms", "Proverbs", "Ecclesiastes", "Isaiah", "Jeremiah",
        "Ezekiel", "Daniel", "Job", "Song of Solomon",
    ],
    "Christianity": [
        "Matthew", "Mark", "Luke", "John", "Acts",
        "Romans", "Corinthians", "Galatians", "Ephesians",
        "Philippians", "Hebrews", "Revelation",
    ],
}

# Quran chapters to fetch (via quran.com API)
QURAN_CHAPTERS = list(range(1, 115))  # All 114 surahs

# Gutenberg texts (verified working)
GUTENBERG_SOURCES = [
    # (tradition, source_name, gutenberg_url)
    ("Hinduism", "Bhagavad Gita (Arnold)", "https://www.gutenberg.org/cache/epub/2388/pg2388.txt"),
    ("Hinduism", "The Upanishads", "https://www.gutenberg.org/cache/epub/3283/pg3283.txt"),
    ("Buddhism", "Dhammapada", "https://www.gutenberg.org/files/2017/2017-0.txt"),
    ("Daoism", "Tao Te Ching", "https://www.gutenberg.org/cache/epub/216/pg216.txt"),
    ("Confucianism", "Analects of Confucius", "https://www.gutenberg.org/cache/epub/3330/pg3330.txt"),
    ("Greek", "Iliad (Pope)", "https://www.gutenberg.org/cache/epub/6130/pg6130.txt"),
    ("Greek", "Odyssey (Butler)", "https://www.gutenberg.org/cache/epub/1727/pg1727.txt"),
    ("Norse", "Poetic Edda", "https://www.gutenberg.org/files/14726/14726-0.txt"),
    ("Mesopotamian", "Epic of Gilgamesh", "https://www.gutenberg.org/files/11000/11000-0.txt"),
    ("Kemetic", "Egyptian Book of the Dead", "https://www.gutenberg.org/cache/epub/69566/pg69566.txt"),
    ("Shinto", "Kojiki (Chamberlain)", "https://www.gutenberg.org/cache/epub/4018/pg4018.txt"),
    ("Maya", "Popol Vuh", "https://www.gutenberg.org/cache/epub/56550/pg56550.txt"),
]

AXES = [
    "authority", "transcendence", "care", "justice", "wisdom", "power",
    "fertility", "war", "death", "creation", "nature", "order"
]


# ═══════════════════════════════════════════════════════════════════════
# STAGE 1: Scrape and chunk passages
# ═══════════════════════════════════════════════════════════════════════

def scrape_and_chunk(min_words=40, max_words=200, target_per_tradition=50):
    """Scrape sacred texts from Gutenberg + Quran API and chunk into passages."""
    import httpx
    from collections import defaultdict

    headers = {"User-Agent": "Mozilla/5.0 (research; gods-as-centroids)"}

    def fetch_gutenberg(url: str) -> str:
        """Fetch a Gutenberg plain text file, strip header/footer."""
        resp = httpx.get(url, headers=headers, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        text = resp.text
        # Strip Project Gutenberg header and footer
        start_markers = ["*** START OF THE PROJECT GUTENBERG", "*** START OF THIS PROJECT GUTENBERG"]
        end_markers = ["*** END OF THE PROJECT GUTENBERG", "*** END OF THIS PROJECT GUTENBERG"]
        for m in start_markers:
            idx = text.find(m)
            if idx >= 0:
                text = text[text.index("\n", idx) + 1:]
                break
        for m in end_markers:
            idx = text.find(m)
            if idx >= 0:
                text = text[:idx]
                break
        return text

    def chunk_text(text: str, min_w=40, max_w=200) -> list:
        """Split plain text into passage-sized chunks using paragraph breaks."""
        # Split on double newlines (paragraph boundaries)
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        buffer = ""
        for p in paragraphs:
            p = re.sub(r'\s+', ' ', p).strip()
            if not p or len(p) < 20:
                continue
            words = p.split()
            if len(words) < 5:
                continue
            if len(buffer.split()) + len(words) <= max_w:
                buffer = (buffer + " " + p).strip()
            else:
                if len(buffer.split()) >= min_w:
                    chunks.append(buffer)
                buffer = p
            if len(buffer.split()) >= max_w:
                chunks.append(buffer)
                buffer = ""
        if buffer and len(buffer.split()) >= min_w:
            chunks.append(buffer)
        return chunks

    def extract_kjv_book(kjv_text: str, book_name: str) -> str:
        """Extract a single book from the KJV Bible text."""
        # KJV Gutenberg format uses headers like:
        #   "The First Book of Moses:  Called Genesis"
        #   "The Gospel According to Saint Matthew"
        #   "The Book of Psalms", "The Proverbs", "Ecclesiastes"
        #   "The Revelation of Saint John the Divine"
        # There's a TOC at the top, then actual text starts with the same headers.
        # We need to skip the TOC and find the actual book content.
        lines = kjv_text.split("\n")

        # Build patterns for this book
        book_pat = re.compile(re.escape(book_name), re.IGNORECASE)
        # Pattern for ANY book header (to detect where next book starts)
        header_pat = re.compile(
            r'^(?:The (?:Old|New) Testament|'
            r'The (?:First |Second |Third |Fourth |Fifth )?'
            r'(?:Book|Epistle|Gospel|General|Lamentations|Song|Proverbs|Ecclesiastes|Revelation|Acts)'
            r'|Genesis|Exodus|Leviticus|Numbers|Deuteronomy|Joshua|Judges|Ruth'
            r'|Ezra|Nehemiah|Esther|Job|Psalms|Proverbs|Ecclesiastes'
            r'|Hosea|Joel|Amos|Obadiah|Jonah|Micah|Nahum|Habakkuk'
            r'|Zephaniah|Haggai|Zechariah|Malachi)',
            re.IGNORECASE
        )
        # Verse pattern: "1:1 In the beginning..."
        verse_pat = re.compile(r'^\d+:\d+')

        # Find the SECOND occurrence of the book name (first is TOC, second is actual text)
        occurrences = []
        for i, line in enumerate(lines):
            if book_pat.search(line.strip()):
                occurrences.append(i)

        if len(occurrences) < 2:
            # Fallback: use first occurrence
            start = occurrences[0] + 1 if occurrences else 0
        else:
            start = occurrences[1] + 1

        # Collect lines until next book header
        book_lines = []
        found_verse = False
        for i in range(start, len(lines)):
            line = lines[i].strip()
            if not line:
                book_lines.append("")
                continue
            # Once we've found verses, a new book header means stop
            if found_verse and header_pat.match(line) and not verse_pat.match(line):
                break
            if verse_pat.match(line):
                found_verse = True
            book_lines.append(lines[i])

        return "\n".join(book_lines)

    def fetch_quran_chapter(chapter: int) -> list:
        """Fetch a Quran chapter from quran.com API."""
        url = f"https://api.quran.com/api/v4/verses/by_chapter/{chapter}?language=en&words=false&translations=20&per_page=300"
        resp = httpx.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        verses = []
        for v in data.get("verses", []):
            for t in v.get("translations", []):
                text = re.sub(r'<[^>]+>', '', t.get("text", ""))
                text = text.strip()
                if text and len(text) > 20:
                    verses.append(text)
        return verses

    all_passages = []
    tradition_counts = defaultdict(int)

    # ─── 1. KJV Bible (Judaism + Christianity) ───────────────────
    print("\n  Fetching KJV Bible from Gutenberg...")
    try:
        kjv_text = fetch_gutenberg("https://www.gutenberg.org/files/10/10-0.txt")
        print(f"    KJV: {len(kjv_text)} chars")

        for tradition, books in KJV_BOOKS.items():
            for book_name in books:
                if tradition_counts[tradition] >= target_per_tradition:
                    break
                book_text = extract_kjv_book(kjv_text, book_name)
                if len(book_text) < 100:
                    print(f"    {book_name}: not found, skipping")
                    continue
                chunks = chunk_text(book_text, min_w=min_words, max_w=max_words)
                added = 0
                for chunk in chunks:
                    if tradition_counts[tradition] >= target_per_tradition:
                        break
                    all_passages.append({
                        "tradition": tradition,
                        "source": f"KJV - {book_name}",
                        "text": chunk,
                    })
                    tradition_counts[tradition] += 1
                    added += 1
                if added > 0:
                    print(f"    {book_name}: {added} passages ({tradition})")
    except Exception as e:
        print(f"    KJV error: {e}")

    # ─── 2. Quran (Islam + Sufism) ────────────────────────────────
    print("\n  Fetching Quran from quran.com API...")
    quran_verses = []
    for ch in QURAN_CHAPTERS:
        if tradition_counts["Islam"] >= target_per_tradition and tradition_counts.get("Sufism", 0) >= target_per_tradition:
            break
        try:
            verses = fetch_quran_chapter(ch)
            quran_verses.extend(verses)
            if ch % 20 == 0:
                print(f"    Surah {ch}: {len(verses)} verses (total: {len(quran_verses)})")
            time.sleep(0.2)
        except Exception as e:
            print(f"    Surah {ch} error: {e}")

    # Chunk Quran verses into passages
    quran_text = "\n\n".join(quran_verses)
    quran_chunks = chunk_text(quran_text, min_w=min_words, max_w=max_words)
    print(f"    Quran: {len(quran_chunks)} passages from {len(quran_verses)} verses")

    for chunk in quran_chunks:
        # Split between Islam and Sufism (Sufism draws from same text)
        if tradition_counts["Islam"] < target_per_tradition:
            all_passages.append({"tradition": "Islam", "source": "Quran", "text": chunk})
            tradition_counts["Islam"] += 1
        elif tradition_counts.get("Sufism", 0) < target_per_tradition:
            all_passages.append({"tradition": "Sufism", "source": "Quran (Sufi reading)", "text": chunk})
            tradition_counts["Sufism"] = tradition_counts.get("Sufism", 0) + 1

    # ─── 3. Gutenberg texts ──────────────────────────────────────
    for tradition, source_name, url in GUTENBERG_SOURCES:
        if tradition_counts[tradition] >= target_per_tradition:
            continue
        print(f"\n  Fetching {source_name}...")
        try:
            text = fetch_gutenberg(url)
            chunks = chunk_text(text, min_w=min_words, max_w=max_words)
            added = 0
            for chunk in chunks:
                if tradition_counts[tradition] >= target_per_tradition:
                    break
                all_passages.append({
                    "tradition": tradition,
                    "source": source_name,
                    "text": chunk,
                })
                tradition_counts[tradition] += 1
                added += 1
            print(f"    {source_name}: {added} passages")
            time.sleep(0.5)
        except Exception as e:
            print(f"    Error: {e}")

    # ─── 4. Traditions without online sources: keep existing corpus passages ─
    # These traditions don't have easily scrapeable public domain texts:
    # Akan, Candomble, Vodou, Rastafari, Cao Dai, Druze, Muism, Tengrism,
    # Wicca, Lakota, Navajo, Hawaiian, Maori, Inca, Nahua, Aboriginal Australian,
    # Secular Humanism, Samaritanism, Bahai, Sikhism, Jainism, Zoroastrianism, Yoruba
    # We'll supplement these from the existing 126-passage corpus
    existing_path = Path(__file__).parent.parent / "mlx-pipeline" / "multi_scorer_consensus.json"
    if existing_path.exists():
        with open(existing_path) as f:
            existing = json.load(f)
        # Add existing passages for traditions we couldn't scrape
        for e in existing["embeddings"]:
            trad = e["tradition"]
            if tradition_counts[trad] < 5:  # tradition has very few scraped passages
                all_passages.append({
                    "tradition": trad,
                    "source": e["source"],
                    "text": f"[Existing corpus] {e['source']} ({trad})",
                    "existing_scores": e.get("normalized"),
                })
                tradition_counts[trad] += 1

    print(f"\n{'='*60}")
    print(f"Total passages scraped: {len(all_passages)}")
    print(f"Traditions covered: {len(tradition_counts)}")
    for t, c in sorted(tradition_counts.items()):
        print(f"  {t}: {c}")

    # Save
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "scraped_passages.json"
    with open(out_path, "w") as f:
        json.dump({
            "description": "Scraped sacred text passages from Gutenberg + Quran API",
            "n_passages": len(all_passages),
            "n_traditions": len(tradition_counts),
            "tradition_counts": dict(tradition_counts),
            "passages": all_passages,
        }, f, indent=2)

    print(f"\nSaved to {out_path}")
    return all_passages


# ═══════════════════════════════════════════════════════════════════════
# STAGE 2: Score with LLMs on Modal
# ═══════════════════════════════════════════════════════════════════════

import modal

app = modal.App("corpus-scorer")

scorer_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("httpx")
)


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
    try:
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
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
    except (json.JSONDecodeError, ValueError):
        return None


@app.function(
    image=scorer_image,
    timeout=300,
    retries=2,
    secrets=[modal.Secret.from_name("openrouter-secret")],
)
def score_passage(passage: dict, model_id: str, model_slug: str) -> dict:
    """Score a single passage with one LLM."""
    import httpx
    import json
    import time

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    api_url = "https://openrouter.ai/api/v1/chat/completions"

    prompt = make_prompt(passage)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/elevate-foundry/gods-as-centroids",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 800,
        "temperature": 0.0,
    }

    for attempt in range(3):
        try:
            resp = httpx.post(api_url, json=payload, headers=headers, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            scores = parse_scores(content)
            if scores:
                return {
                    "tradition": passage["tradition"],
                    "source": passage["source"],
                    "text": passage["text"],
                    "model": model_slug,
                    "raw_scores": scores,
                }
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))

    return None


@app.function(
    image=scorer_image,
    timeout=7200,
    secrets=[modal.Secret.from_name("openrouter-secret")],
)
def score_batch(passages: list, model_id: str, model_slug: str) -> list:
    """Score a batch of passages with one model (sequential to respect rate limits)."""
    import httpx
    import time

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    api_url = "https://openrouter.ai/api/v1/chat/completions"

    if not api_key:
        print(f"    {model_slug}: ERROR - OPENROUTER_API_KEY not set!")
        print(f"    Available env vars: {[k for k in os.environ if 'KEY' in k or 'SECRET' in k or 'TOKEN' in k or 'ROUTER' in k.upper()]}")
        return []

    print(f"    {model_slug}: API key found ({api_key[:8]}...), starting scoring")

    results = []
    errors = 0
    parse_fails = 0
    for i, passage in enumerate(passages):
        # Skip existing corpus placeholders
        if "existing_scores" in passage:
            continue

        prompt = make_prompt(passage)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/elevate-foundry/gods-as-centroids",
        }
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 800,
            "temperature": 0.0,
        }

        scores = None
        last_error = None
        last_content = None
        for attempt in range(3):
            try:
                resp = httpx.post(api_url, json=payload, headers=headers, timeout=90)
                if resp.status_code != 200:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    if i < 3:
                        print(f"    {model_slug}: HTTP {resp.status_code} on passage {i}: {resp.text[:200]}")
                    if attempt < 2:
                        time.sleep(2 ** (attempt + 1))
                    continue
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                last_content = content
                scores = parse_scores(content)
                if scores:
                    break
                elif i < 3:
                    print(f"    {model_slug}: parse_scores returned None for: {content[:150]}")
            except Exception as e:
                last_error = str(e)
                if i < 3:
                    print(f"    {model_slug}: Exception on passage {i}: {e}")
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))

        if scores:
            results.append({
                "tradition": passage["tradition"],
                "source": passage["source"],
                "text": passage["text"],
                "model": model_slug,
                "raw_scores": scores,
            })
        else:
            if last_error and errors < 3:
                print(f"    {model_slug}: API error on passage {i}: {last_error[:100]}")
            elif last_content and parse_fails < 3:
                print(f"    {model_slug}: Parse fail on passage {i}: {last_content[:100]}")
            if last_error:
                errors += 1
            else:
                parse_fails += 1

        if (i + 1) % 50 == 0:
            print(f"    {model_slug}: {i+1}/{len(passages)} scored ({len(results)} ok, {errors} errors, {parse_fails} parse fails)")

        # Rate limit: ~1 req/sec
        time.sleep(0.3)

    print(f"    {model_slug}: {len(results)}/{len(passages)} scored successfully")
    return results


@app.local_entrypoint()
def main():
    """Score all scraped passages with 4 LLMs in parallel."""
    # Load scraped passages
    data_path = Path("lattice_autoencoder/data/scraped_passages.json")
    if not data_path.exists():
        print("No scraped passages found. Run with --stage scrape first.")
        return

    with open(data_path) as f:
        data = json.load(f)

    passages = data["passages"]
    print(f"Loaded {len(passages)} passages from {data['n_traditions']} traditions")

    # Models to score with (same as existing pipeline)
    models = [
        ("anthropic/claude-sonnet-4", "claude"),
        ("openai/gpt-4o", "gpt4o"),
        ("google/gemini-2.0-flash-001", "gemini_flash"),
        ("meta-llama/llama-3.3-70b-instruct", "llama70b"),
    ]

    # Debug mode: test with small batch first
    debug = os.environ.get("DEBUG_SCORING", "")
    if debug:
        passages = [p for p in passages if "existing_scores" not in p][:5]
        models = models[:1]
        print(f"DEBUG MODE: {len(passages)} passages, {len(models)} model(s)")

    # Launch models in parallel, each scoring all passages
    print(f"\nScoring with {len(models)} models in parallel on Modal...")
    t0 = time.time()

    futures = []
    for model_id, slug in models:
        futures.append(score_batch.spawn(passages, model_id, slug))

    # Collect results
    all_model_results = {}
    for (model_id, slug), future in zip(models, futures):
        results = future.get()
        all_model_results[slug] = results
        print(f"  {slug}: {len(results)} passages scored")

    elapsed = time.time() - t0
    print(f"\nAll models done in {elapsed:.0f}s")

    # Build consensus
    print("\nBuilding consensus...")
    consensus_embeddings = []

    # Index results by (tradition, source, text) for each model
    model_scores = {}
    for slug, results in all_model_results.items():
        for r in results:
            key = (r["tradition"], r["source"], r["text"][:100])
            if key not in model_scores:
                model_scores[key] = {"tradition": r["tradition"], "source": r["source"],
                                      "text": r["text"], "scores": {}}
            model_scores[key]["scores"][slug] = r["raw_scores"]

    for key, entry in model_scores.items():
        if len(entry["scores"]) < 2:
            continue  # Need at least 2 models

        # Compute mean across models
        mean_scores = {}
        for axis in AXES:
            vals = [entry["scores"][m].get(axis, 0.0) for m in entry["scores"]]
            mean_scores[axis] = sum(vals) / len(vals)

        # Normalize
        norm = math.sqrt(sum(v * v for v in mean_scores.values())) or 1.0
        normalized = {a: v / norm for a, v in mean_scores.items()}

        # Compute scorer variance
        variance = {}
        for axis in AXES:
            vals = [entry["scores"][m].get(axis, 0.0) for m in entry["scores"]]
            mean = sum(vals) / len(vals)
            variance[axis] = sum((v - mean) ** 2 for v in vals) / len(vals)

        consensus_embeddings.append({
            "tradition": entry["tradition"],
            "source": entry["source"],
            "text": entry["text"],
            "raw_scores": mean_scores,
            "normalized": normalized,
            "scorer_variance": variance,
            "n_scorers": len(entry["scores"]),
        })

    print(f"Consensus passages: {len(consensus_embeddings)}")

    # Count per tradition
    from collections import Counter
    trad_counts = Counter(e["tradition"] for e in consensus_embeddings)
    for t, c in sorted(trad_counts.items()):
        print(f"  {t}: {c}")

    # Save
    out_dir = Path("lattice_autoencoder/data")
    out_dir.mkdir(exist_ok=True)

    output = {
        "description": "Scaled multi-LLM consensus corpus (scraped sacred texts)",
        "models": {slug: model_id for model_id, slug in models},
        "n_passages": len(consensus_embeddings),
        "n_traditions": len(trad_counts),
        "embeddings": consensus_embeddings,
    }

    out_path = out_dir / "scaled_consensus.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {out_path}")

    # Also merge with existing corpus
    existing_path = Path("mlx-pipeline/multi_scorer_consensus.json")
    if existing_path.exists():
        with open(existing_path) as f:
            existing = json.load(f)

        merged = existing["embeddings"] + consensus_embeddings
        merged_output = {
            "description": "Merged corpus: original 126 + scaled scraped passages",
            "n_passages": len(merged),
            "n_traditions": len(set(e["tradition"] for e in merged)),
            "embeddings": merged,
        }

        merged_path = out_dir / "merged_consensus.json"
        with open(merged_path, "w") as f:
            json.dump(merged_output, f, indent=2)

        print(f"Merged corpus: {len(merged)} passages → {merged_path}")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="scrape", choices=["scrape", "score", "both"])
    parser.add_argument("--target-per-tradition", type=int, default=50)
    parser.add_argument("--min-words", type=int, default=40)
    parser.add_argument("--max-words", type=int, default=200)
    args = parser.parse_args()

    if args.stage in ("scrape", "both"):
        print("STAGE 1: Scraping sacred texts...")
        scrape_and_chunk(
            min_words=args.min_words,
            max_words=args.max_words,
            target_per_tradition=args.target_per_tradition,
        )

    if args.stage == "score":
        print("Run 'modal run lattice_autoencoder/scale_corpus_modal.py' for Stage 2")
