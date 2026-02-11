# Gods as Centroids

**A Generative Vector-Space Model of Religious Evolution**

[![Paper](https://img.shields.io/badge/Paper-GitHub%20Pages-blue)](https://elevate-foundry.github.io/gods-as-centroids/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

We present a generative agent-based model in which deities emerge as mathematical centroids of belief-vector clusters. Each agent carries a belief vector in a 12-dimensional theological space. Through local communication, prestige-weighted social influence, and stochastic mutation, agents self-organize into clusters whose centroids correspond to emergent "godforms." A single coercion parameter drives a first-order phase transition from polytheism to monotheism with **hysteresis** — reducing coercion does not restore polytheism.

**Author:** Ryan Barrett (2026)

---

## Key Results

| Claim | Result | Script |
|-------|--------|--------|
| Hysteresis (§4.4) | Δγ ≈ 0.55, γ_c⁺ ≈ 0.85, γ_c⁻ ≈ 0.30 | `sim/hysteresis_modal.py` |
| Lattice snap dynamics (§4.5) | 9× lower threshold with Hamming centroids | `sim/lattice_hysteresis_modal.py` |
| Prophet escape (§7.3) | Sweet spot at prophet~0.005, μ~0.10–0.15 | `sim/prophet_escape_modal.py` |
| Multi-LLM agreement (§8.2) | Krippendorff's α = 0.903 | `mlx-pipeline/multi_scorer.py` |
| Braille bottleneck (§5.4) | 0.995 cosine through 72-bit lattice | `sim/recursive_compression.py` |
| Cross-domain (Extension) | Political 88%, Personality 90% accuracy | `mlx-pipeline/non_theological_demo.py` |
| MMLA Phase 2 (§5.4) | 100% classification, 87.1% bit agreement | `lattice_autoencoder/train_modal.py` |

---

## Repository Structure

```
gods-as-centroids/
├── paper/                      # Paper source (markdown + LaTeX)
│   ├── gods_as_centroids_v2.md # Source of truth
│   ├── md2html.py              # Markdown → GitHub Pages HTML
│   └── md2latex.py             # Markdown → LaTeX
├── sim/                        # Simulation engine
│   ├── swarm_kernel.py         # Core ABM (agents, clustering, operators)
│   ├── hysteresis_modal.py     # §4.4 hysteresis sweep (Modal)
│   ├── lattice_hysteresis_modal.py  # §4.5 Hamming vs arithmetic
│   ├── prophet_escape_modal.py # §7.3 prophet escape grid
│   ├── finite_size_modal.py    # §10 finite-size scaling
│   ├── corpus_calibration.py   # §8.3 parameter derivation
│   ├── recursive_compression.py # §5.4–5.5 braille lattice + operators
│   ├── configs/                # Preset configurations
│   ├── experiments/            # Standalone reproduction scripts
│   └── runs/                   # Saved simulation outputs
├── lattice_autoencoder/        # Multi-Modal Lattice Autoencoder
│   ├── model.py                # Phase 1/2/3 architectures
│   ├── lattice.py              # Differentiable braille lattice (STE)
│   ├── data.py                 # Dataset loaders
│   ├── train.py                # Phase 1 training
│   ├── train_multimodal.py     # Phase 2+3 local training
│   ├── train_modal.py          # GPU sweep on Modal (T4)
│   ├── embed_corpus.py         # TF-IDF+PCA text embeddings
│   ├── scale_corpus_modal.py   # 826-passage corpus scraper + scorer
│   └── data/                   # Corpus data files
├── mlx-pipeline/               # MLX/LLM scoring pipeline
│   ├── corpus_parts/           # 126-passage expanded corpus (37 traditions)
│   ├── multi_scorer.py         # 4-LLM inter-scorer agreement
│   ├── non_theological_demo.py # Cross-domain generalizability
│   ├── frontier_judge.py       # LLM-as-judge validation
│   └── real_embeddings.py      # Real passage embeddings
├── web/                        # Interactive web simulation (Next.js)
│   ├── app/lib/simulation.ts   # TypeScript SwarmKernel
│   ├── app/lib/braille-lattice.ts
│   ├── app/lib/historical-data.ts
│   └── app/components/         # React UI components
├── cpp/                        # C++17 port (performance)
├── rust/                       # Rust port (performance)
├── go/                         # Go port (performance)
├── docs/                       # GitHub Pages site
│   ├── template.html           # HTML template (immutable)
│   └── index.html              # Generated (never edit directly)
└── tests/                      # Automated test suite
    └── test_paper_claims.py    # Pytest verification of all paper claims
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ (for web app)
- [Modal](https://modal.com) account (for GPU experiments, optional)

### Install

```bash
# Clone
git clone https://github.com/elevate-foundry/gods-as-centroids.git
cd gods-as-centroids

# Python dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Web app (optional)
cd web && npm install && cd ..
```

### Run the Simulation

```bash
# Basic simulation (low coercion → polytheism)
python sim/run.py -c sim/configs/low_coercion.json -s 5000 -l polytheism_demo

# High coercion → monotheism
python sim/run.py -c sim/configs/high_coercion.json -s 5000 -l monotheism_demo
```

### Reproduce Paper Results

```bash
# Corollary 2: Ritual stabilization (§5.2)
python sim/experiments/corollary2_ritual.py

# Corollary 3: Prestige convergence (§5.3)
python sim/experiments/corollary3_prestige.py

# Historical backtesting (§6.2)
python sim/experiments/backtesting.py

# Full test suite (verifies all paper claims)
pytest tests/ -v
```

### GPU Experiments (Modal)

```bash
# Hysteresis sweep (§4.4) — 30 parallel runs
modal run sim/hysteresis_modal.py

# Lattice hysteresis (§4.5) — Hamming vs arithmetic
modal run sim/lattice_hysteresis_modal.py

# Prophet escape (§7.3) — 600-run grid
modal run sim/prophet_escape_modal.py

# MMLA training sweep (§5.4)
modal run lattice_autoencoder/train_modal.py
```

### Web App

```bash
cd web
npm run dev
# Open http://localhost:3000
```

### Regenerate Paper HTML

```bash
python paper/md2html.py
# View at docs/index.html or https://elevate-foundry.github.io/gods-as-centroids/
```

---

## Corpus

The model is calibrated against **826 canonical religious passages** from **37 traditions** spanning every inhabited continent, scored by 4 frontier LLMs (Claude Sonnet 4, GPT-4o, Gemini 2.0 Flash, Llama 3.3 70B) with Krippendorff's α = 0.903.

| Family | Traditions |
|--------|-----------|
| Abrahamic | Judaism, Christianity, Islam, Baha'i, Druze, Samaritanism, Zoroastrianism |
| Dharmic | Hinduism, Buddhism, Jainism, Sikhism |
| East Asian | Daoism, Confucianism, Shinto |
| African | Yoruba, Akan, Kemetic |
| African diaspora | Candomblé, Vodou, Rastafari |
| Indigenous | Lakota, Navajo, Aboriginal Australian, Maori, Hawaiian |
| Mesoamerican/Andean | Nahua, Maya, Inca |
| Other | Tengrism, Korean Muism, Cao Dai, Wicca, Sufism, Secular Humanism |

---

## Citation

```bibtex
@article{barrett2026gods,
  title={Gods as Centroids: A Generative Vector-Space Model of Religious Evolution},
  author={Barrett, Ryan},
  year={2026},
  url={https://elevate-foundry.github.io/gods-as-centroids/}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
