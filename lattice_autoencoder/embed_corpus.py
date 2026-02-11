"""
Generate text embeddings for corpus passages.

Uses a simple but effective approach: TF-IDF-like bag-of-words vectors
from the passage texts, then PCA to a fixed dimension. This avoids
heavy dependencies while producing meaningful text representations
that are genuinely different from the LLM-scored theological vectors.

Supports both the original 126-passage corpus and the scaled corpus.
Usage:
  python embed_corpus.py              # original 126 passages
  python embed_corpus.py --scaled      # scaled corpus (scraped + original)

Output: lattice_autoencoder/data/text_embeddings.json
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

# Add parent to path for corpus_parts import
sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-pipeline"))
from corpus_parts import EXPANDED_CORPUS


def tokenize(text: str) -> list:
    """Simple whitespace + punctuation tokenizer."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return [w for w in text.split() if len(w) > 2]


def build_vocab(corpus: list, min_df: int = 2, max_df_frac: float = 0.8) -> dict:
    """Build vocabulary from corpus, filtering by document frequency."""
    doc_freq = Counter()
    for passage in corpus:
        tokens = set(tokenize(passage["text"]))
        for t in tokens:
            doc_freq[t] += 1

    n_docs = len(corpus)
    vocab = {}
    idx = 0
    for word, df in sorted(doc_freq.items()):
        if df >= min_df and df / n_docs <= max_df_frac:
            vocab[word] = idx
            idx += 1
    return vocab


def tfidf_vector(text: str, vocab: dict, idf: dict) -> list:
    """Compute TF-IDF vector for a text."""
    tokens = tokenize(text)
    tf = Counter(tokens)
    vec = [0.0] * len(vocab)
    for word, count in tf.items():
        if word in vocab:
            vec[vocab[word]] = (1 + math.log(count)) * idf.get(word, 1.0)
    # L2 normalize
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def pca_reduce(vectors: list, target_dim: int = 128) -> list:
    """Simple PCA via SVD (no numpy dependency beyond what torch provides)."""
    import torch

    X = torch.tensor(vectors, dtype=torch.float32)
    # Center
    mean = X.mean(dim=0)
    X_centered = X - mean

    # SVD
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)

    # Project to top-k components
    k = min(target_dim, X.shape[1], X.shape[0])
    X_reduced = X_centered @ Vh[:k].T

    # L2 normalize each row
    norms = X_reduced.norm(dim=1, keepdim=True).clamp(min=1e-8)
    X_reduced = X_reduced / norms

    return X_reduced.tolist(), mean.tolist(), Vh[:k].tolist()


def load_scaled_corpus():
    """Load the scaled corpus: scraped passages (with text) + original corpus."""
    data_dir = Path(__file__).parent / "data"
    scraped_path = data_dir / "scraped_passages.json"
    consensus_path = data_dir / "merged_consensus.json"

    if not scraped_path.exists() or not consensus_path.exists():
        raise FileNotFoundError("Scaled corpus not found. Run scale_corpus_modal.py first.")

    # Load scraped passages (these have actual text)
    with open(scraped_path) as f:
        scraped = json.load(f)

    # Load merged consensus (to know which passages have scores)
    with open(consensus_path) as f:
        consensus = json.load(f)

    # Build a set of (tradition, source) pairs that are in the consensus
    consensus_keys = set()
    for e in consensus["embeddings"]:
        consensus_keys.add((e["tradition"], e["source"]))

    # Collect passages with real text that are in the consensus
    corpus = []
    # First: scraped passages with real text
    for p in scraped["passages"]:
        if "existing_scores" in p:
            continue  # skip placeholders
        if (p["tradition"], p["source"]) in consensus_keys:
            corpus.append({"tradition": p["tradition"], "source": p["source"], "text": p["text"]})

    # Second: original corpus passages (for traditions not scraped)
    scraped_keys = set((p["tradition"], p["source"]) for p in corpus)
    for passage in EXPANDED_CORPUS:
        key = (passage["tradition"], passage["source"])
        if key not in scraped_keys and key in consensus_keys:
            corpus.append(passage)

    return corpus


def main():
    scaled = "--scaled" in sys.argv

    if scaled:
        corpus = load_scaled_corpus()
        print(f"Scaled corpus: {len(corpus)} passages")
    else:
        corpus = EXPANDED_CORPUS
        print(f"Original corpus: {len(corpus)} passages")

    # Build vocabulary
    vocab = build_vocab(corpus, min_df=2, max_df_frac=0.7)
    print(f"Vocabulary: {len(vocab)} terms")

    # Compute IDF
    n_docs = len(corpus)
    doc_freq = Counter()
    for passage in corpus:
        tokens = set(tokenize(passage["text"]))
        for t in tokens:
            if t in vocab:
                doc_freq[t] += 1
    idf = {w: math.log(n_docs / (df + 1)) for w, df in doc_freq.items()}

    # Generate TF-IDF vectors
    tfidf_vecs = []
    metadata = []
    for passage in corpus:
        vec = tfidf_vector(passage["text"], vocab, idf)
        tfidf_vecs.append(vec)
        metadata.append({
            "tradition": passage["tradition"],
            "source": passage["source"],
        })

    print(f"TF-IDF vectors: {len(tfidf_vecs)} × {len(tfidf_vecs[0])}")

    # PCA reduce
    target_dim = min(128, len(corpus) - 1)  # can't have more dims than samples
    reduced, pca_mean, pca_components = pca_reduce(tfidf_vecs, target_dim=target_dim)
    print(f"PCA reduced: {len(reduced)} × {len(reduced[0])}")

    # Save
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)

    desc = f"Text embeddings for {len(corpus)} corpus passages (TF-IDF + PCA-{len(reduced[0])})"
    output = {
        "description": desc,
        "embed_dim": len(reduced[0]),
        "n_passages": len(reduced),
        "vocab_size": len(vocab),
        "embeddings": [
            {
                "tradition": metadata[i]["tradition"],
                "source": metadata[i]["source"],
                "vector": reduced[i],
            }
            for i in range(len(reduced))
        ],
        "pca_mean": pca_mean,
        "pca_components": pca_components,
    }

    out_path = out_dir / "text_embeddings.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {out_path}")

    # Quick sanity check: are same-tradition passages closer?
    import torch
    X = torch.tensor(reduced, dtype=torch.float32)
    sims = X @ X.T

    intra_sims = []
    inter_sims = []
    for i in range(len(reduced)):
        for j in range(i + 1, len(reduced)):
            sim = sims[i, j].item()
            if metadata[i]["tradition"] == metadata[j]["tradition"]:
                intra_sims.append(sim)
            else:
                inter_sims.append(sim)

    if intra_sims and inter_sims:
        print(f"\nSanity check:")
        print(f"  Intra-tradition cosine: {sum(intra_sims)/len(intra_sims):.4f}")
        print(f"  Inter-tradition cosine: {sum(inter_sims)/len(inter_sims):.4f}")
        print(f"  Separation: {sum(intra_sims)/len(intra_sims) - sum(inter_sims)/len(inter_sims):.4f}")


if __name__ == "__main__":
    main()
