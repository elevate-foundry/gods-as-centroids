# Multi-Modal Lattice Autoencoder (MMLA)

## Architecture

A multi-modal autoencoder where the shared bottleneck is the **8-dot braille lattice** — a structured discrete codebook where every bit has a named semantic role.

```
                    ┌─────────────┐
  Text ──→ TextEnc ─┤             ├─→ TextDec ──→ Text'
 Image ──→ ImgEnc ──┤  96-bit     ├─→ ImgDec ──→ Image'
 Audio ──→ AudEnc ──┤  Braille    ├─→ AudDec ──→ Audio'
                    │  Lattice    │
                    │  Bottleneck │
                    └─────────────┘
                          │
                    Hamming-mean
                    centroid for
                    multi-modal
                    consensus
```

### Key Insight

Unlike VQ-VAE (learned codebook, uninterpretable codes) or standard VAE (continuous latent, no discrete structure), the braille lattice provides:

1. **Structured codebook**: Every bit has a named semantic role (polarity, intensity, rigidity, salience, momentum)
2. **Built-in distance metric**: Hamming distance is semantically meaningful
3. **Composable operators**: Fusion, fission, perturbation work directly on the latent codes
4. **Multi-model consensus**: Hamming-mean across encoders = majority-vote semantic agreement

## Components

### Phase 1: Semantic Domain (theology, validated)
- `lattice_autoencoder/model.py` — Core model architecture
- `lattice_autoencoder/lattice.py` — Differentiable lattice layer (straight-through estimator)
- `lattice_autoencoder/train.py` — Training loop
- `lattice_autoencoder/data.py` — Data loading (consensus corpus + augmentation)

### Phase 2: Multi-modal (text + image)
- Add image encoder/decoder
- Cross-modal alignment loss
- Paired dataset (image-caption)

### Phase 3: Operator layer
- Fusion/fission/perturbation as differentiable operations
- Operator prediction head

## Training

```bash
# Phase 1: Train on theological domain (126 passages, 37 traditions)
python -m lattice_autoencoder.train --domain theology --epochs 200

# Phase 1b: Train on political/personality domains
python -m lattice_autoencoder.train --domain political --epochs 200

# Phase 2: Multi-modal (requires paired data)
python -m lattice_autoencoder.train --multimodal --epochs 500
```

## Requirements

- PyTorch >= 2.0
- numpy
- (Phase 2: torchvision, torchaudio)

## Theoretical Foundation

See paper §5.4–5.5 (Braille Lattice Corollary, 8-Dot Extension) and §8.4 (4-LLM Lattice Consensus).
