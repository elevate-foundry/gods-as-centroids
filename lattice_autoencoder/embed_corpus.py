"""
Generate text embeddings for the 126 corpus passages.

Uses a simple but effective approach: TF-IDF-like bag-of-words vectors
from the passage texts, then PCA to a fixed dimension. This avoids
heavy dependencies while producing meaningful text representations
that are genuinely different from the LLM-scored theological vectors.

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


def main():
    print(f"Corpus: {len(EXPANDED_CORPUS)} passages")

    # Build vocabulary
    vocab = build_vocab(EXPANDED_CORPUS, min_df=2, max_df_frac=0.7)
    print(f"Vocabulary: {len(vocab)} terms")

    # Compute IDF
    n_docs = len(EXPANDED_CORPUS)
    doc_freq = Counter()
    for passage in EXPANDED_CORPUS:
        tokens = set(tokenize(passage["text"]))
        for t in tokens:
            if t in vocab:
                doc_freq[t] += 1
    idf = {w: math.log(n_docs / (df + 1)) for w, df in doc_freq.items()}

    # Generate TF-IDF vectors
    tfidf_vecs = []
    metadata = []
    for passage in EXPANDED_CORPUS:
        vec = tfidf_vector(passage["text"], vocab, idf)
        tfidf_vecs.append(vec)
        metadata.append({
            "tradition": passage["tradition"],
            "source": passage["source"],
        })

    print(f"TF-IDF vectors: {len(tfidf_vecs)} × {len(tfidf_vecs[0])}")

    # PCA reduce to 128 dimensions
    reduced, pca_mean, pca_components = pca_reduce(tfidf_vecs, target_dim=128)
    print(f"PCA reduced: {len(reduced)} × {len(reduced[0])}")

    # Save
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)

    output = {
        "description": "Text embeddings for 126 corpus passages (TF-IDF + PCA-128)",
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

    print(f"\nSanity check:")
    print(f"  Intra-tradition cosine: {sum(intra_sims)/len(intra_sims):.4f}")
    print(f"  Inter-tradition cosine: {sum(inter_sims)/len(inter_sims):.4f}")
    print(f"  Separation: {sum(intra_sims)/len(intra_sims) - sum(inter_sims)/len(inter_sims):.4f}")


if __name__ == "__main__":
    main()
