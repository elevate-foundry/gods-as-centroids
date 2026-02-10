# Braille Bottleneck Experiment — Quantitative Results

## Claim

> Emergent theological structures—defined as population-level centroid attractors in a designed belief space—remain stable when forced through a fixed, 72-bit discrete bottleneck. This suggests that such structures are not artifacts of high-dimensional continuous embeddings, but persist under severe semantic compression.

**Architecture:** `Encoder(ℝ¹²) → BrailleQuantizer({0,1}⁷²) → Decoder(ℝ¹²)`
**Bottleneck:** 72 bits (6 per axis × 12 axes)
**Encoding:** 2 polarity + 2 intensity + 2 rigidity per axis

---

## Experiment A: Centroid Preservation (72-bit)

Does the centroid survive passage through the braille lattice?

| Deity | Cosine Sim | L2 Displacement | Active Bits |
|-------|-----------|-----------------|-------------|
| Zeus | 0.9955 | 0.0958 | 38/72 |
| Yahweh | 0.9922 | 0.1252 | 36/72 |
| Vishnu | 0.9979 | 0.0719 | 38/72 |
| Odin | 0.9962 | 0.0900 | 38/72 |
| Isis | 0.9949 | 0.1008 | 40/72 |
| Mars | 0.9948 | 0.1036 | 36/72 |
| Ra | 0.9951 | 0.0991 | 39/72 |
| Kali | 0.9970 | 0.0831 | 39/72 |
| Thor | 0.9986 | 0.0617 | 39/72 |
| Athena | 0.9969 | 0.0803 | 41/72 |
| Gaia | 0.9962 | 0.0906 | 43/72 |
| Anubis | 0.9967 | 0.0846 | 36/72 |
| Apollo | 0.9911 | 0.1330 | 38/72 |
| Shiva | 0.9926 | 0.1221 | 37/72 |
| Freya | 0.9947 | 0.1044 | 41/72 |
| Ares | 0.9967 | 0.0846 | 35/72 |

**Mean cosine similarity:** 0.9954
**Mean L2 displacement:** 0.0957
**Mean inter-deity distance:** 0.5539
**Displacement ratio:** 0.1727 (displacement / inter-deity distance — lower means centroids are preserved)

---

## Experiment B: Task Invariance (72-bit)

**Classification accuracy:** 0.8406
**Similarity rank correlation (Spearman):** 0.9713 (p = 1.99e-75)

---

## Experiment C: Capacity Stress Test

Phase transition in representational capacity:

| Total Bits | Bits/Axis | Classification | Cosine Sim | Displacement Ratio | Rank Correlation |
|-----------|-----------|---------------|-----------|-------------------|-----------------|
| 12 | 1 | 0.8375 | 0.9995 | 0.0809 | 0.9959 |
| 24 | 2 | 0.8450 | 0.9986 | 0.0988 | 0.9865 |
| 36 | 3 | 0.8481 | 0.9971 | 0.1402 | 0.9838 |
| 48 | 4 | 0.8400 | 0.9959 | 0.1602 | 0.9737 |
| 72 | 6 | 0.8419 | 0.9952 | 0.1786 | 0.9709 |
| 96 | 8 | 0.8400 | 0.9947 | 0.1841 | 0.9749 |

---

## Channel Invariance

### abstract_conceptual
- **Restricted axes:** transcendence, creation
- **Mean Hamming distance:** 11.9/72
- **Braille-identical centroids:** 1/16
- **Mean reconstructed cosine sim:** 0.9500
- **Same classification:** 9/16

### social_political
- **Restricted axes:** authority, justice, power, order
- **Mean Hamming distance:** 17.0/72
- **Braille-identical centroids:** 0/16
- **Mean reconstructed cosine sim:** 0.9248
- **Same classification:** 6/16

### visual_embodied
- **Restricted axes:** wisdom, power, fertility, war
- **Mean Hamming distance:** 18.8/72
- **Braille-identical centroids:** 0/16
- **Mean reconstructed cosine sim:** 0.9008
- **Same classification:** 7/16

---

## Experiment D: Semantic Braiding

Multiple encoder variants → shared braille bottleneck → bitwise majority consensus.

**Models braided:** 5
**Mean bit agreement between models:** 0.4920
**Mean braided cosine similarity:** 0.9574
**Mean individual cosine similarity:** 0.9946
**Braid improvement over individual:** -0.0371
**Braid correct classification rate:** 0.7500

| Deity | Braided Cos | Mean Indiv Cos | Improvement | Braid Correct | Models Agree |
|-------|-----------|---------------|------------|--------------|-------------|
| Zeus | 0.9542 | 0.9948 | -0.0405 | ✓ | ✓ |
| Yahweh | 0.9663 | 0.9951 | -0.0288 | ✓ | ✓ |
| Vishnu | 0.9827 | 0.9926 | -0.0099 | ✗ | ✓ |
| Odin | 0.9708 | 0.9961 | -0.0253 | ✓ | ✓ |
| Isis | 0.9620 | 0.9907 | -0.0287 | ✗ | ✓ |
| Mars | 0.8996 | 0.9963 | -0.0967 | ✓ | ✓ |
| Ra | 0.9840 | 0.9940 | -0.0100 | ✓ | ✓ |
| Kali | 0.9518 | 0.9925 | -0.0407 | ✓ | ✓ |
| Thor | 0.9391 | 0.9943 | -0.0552 | ✓ | ✓ |
| Athena | 0.9737 | 0.9958 | -0.0221 | ✓ | ✓ |
| Gaia | 0.9603 | 0.9972 | -0.0369 | ✓ | ✓ |
| Anubis | 0.9477 | 0.9953 | -0.0475 | ✗ | ✓ |
| Apollo | 0.9748 | 0.9958 | -0.0210 | ✗ | ✓ |
| Shiva | 0.9875 | 0.9923 | -0.0048 | ✓ | ✓ |
| Freya | 0.9620 | 0.9941 | -0.0322 | ✓ | ✓ |
| Ares | 0.9022 | 0.9960 | -0.0938 | ✓ | ✗ |

---

## Experiment F: Regime B — Co-Trained Semantic Braiding

K encoders trained **jointly** with alignment loss $\mathcal{L}_{\text{align}} = \sum_{i<j} |z^{(i)} - z^{(j)}|_1$.

**Regime:** B (shared decoder + alignment warmup)
**Models co-trained:** 5
**Alignment weight (α):** 0.01
**Mean bit agreement between models:** 0.8625
**Mean braided cosine similarity:** 0.9964
**Mean individual cosine similarity:** 0.9953
**Braid improvement over individual:** +0.0011
**Braid correct classification rate:** 1.0000

| Deity | Braided Cos | Mean Indiv Cos | Improvement | Braid Correct | Models Agree |
|-------|-----------|---------------|------------|--------------|-------------|
| Zeus | 0.9952 | 0.9952 | +0.0001 | ✓ | ✓ |
| Yahweh | 0.9956 | 0.9951 | +0.0006 | ✓ | ✓ |
| Vishnu | 0.9961 | 0.9930 | +0.0031 | ✓ | ✓ |
| Odin | 0.9984 | 0.9960 | +0.0024 | ✓ | ✓ |
| Isis | 0.9981 | 0.9953 | +0.0028 | ✓ | ✓ |
| Mars | 0.9967 | 0.9969 | -0.0002 | ✓ | ✓ |
| Ra | 0.9988 | 0.9944 | +0.0044 | ✓ | ✗ |
| Kali | 0.9958 | 0.9969 | -0.0010 | ✓ | ✓ |
| Thor | 0.9992 | 0.9945 | +0.0046 | ✓ | ✓ |
| Athena | 0.9935 | 0.9947 | -0.0012 | ✓ | ✓ |
| Gaia | 0.9958 | 0.9962 | -0.0004 | ✓ | ✓ |
| Anubis | 0.9965 | 0.9966 | -0.0001 | ✓ | ✓ |
| Apollo | 0.9952 | 0.9943 | +0.0009 | ✓ | ✓ |
| Shiva | 0.9945 | 0.9948 | -0.0004 | ✓ | ✓ |
| Freya | 0.9944 | 0.9942 | +0.0002 | ✓ | ✓ |
| Ares | 0.9985 | 0.9970 | +0.0015 | ✓ | ✓ |

### Regime A vs B Comparison

| Metric | Regime A (post-hoc) | Regime B (co-trained) |
|--------|--------------------|-----------------------|
| Bit agreement | 0.4920 | 0.8625 |
| Braided cosine sim | 0.9574 | 0.9964 |
| Braid correct rate | 0.7500 | 1.0000 |
| Braid improvement | -0.0371 | +0.0011 |

---

*Generated by the Braille Bottleneck + Semantic Braiding Pipeline — Gods as Centroids*