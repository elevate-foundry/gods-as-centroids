.PHONY: help install test sim-low sim-high corollary2 corollary3 backtest paper web clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Setup ───────────────────────────────────────────────────────────

install: ## Install Python dependencies
	pip install -r requirements.txt

install-web: ## Install web app dependencies
	cd web && npm install

# ── Simulation ──────────────────────────────────────────────────────

sim-low: ## Run low-coercion simulation (polytheism)
	python sim/run.py -c sim/configs/low_coercion.json -s 5000 -l polytheism_demo

sim-high: ## Run high-coercion simulation (monotheism)
	python sim/run.py -c sim/configs/high_coercion.json -s 5000 -l monotheism_demo

# ── Paper Reproduction ──────────────────────────────────────────────

corollary2: ## Reproduce Corollary 2: Ritual Stabilization (§5.2)
	python sim/experiments/corollary2_ritual.py

corollary3: ## Reproduce Corollary 3: Prestige Convergence (§5.3)
	python sim/experiments/corollary3_prestige.py

backtest: ## Reproduce Historical Backtesting (§6.2)
	python sim/experiments/backtesting.py

reproduce-all: corollary2 corollary3 backtest ## Run all reproduction scripts

# ── GPU Experiments (Modal) ─────────────────────────────────────────

hysteresis: ## Run hysteresis sweep on Modal (§4.4)
	modal run sim/hysteresis_modal.py

lattice-hysteresis: ## Run lattice hysteresis on Modal (§4.5)
	modal run sim/lattice_hysteresis_modal.py

prophet-escape: ## Run prophet escape grid on Modal (§7.3)
	modal run sim/prophet_escape_modal.py

mmla-train: ## Run MMLA training sweep on Modal
	modal run lattice_autoencoder/train_modal.py

# ── Testing ─────────────────────────────────────────────────────────

test: ## Run all tests
	pytest tests/ -v

test-fast: ## Run fast tests only (skip slow simulation tests)
	pytest tests/ -v -k "not Slow"

# ── Paper ───────────────────────────────────────────────────────────

paper: ## Regenerate GitHub Pages HTML from markdown
	python paper/md2html.py

latex: ## Generate LaTeX from markdown
	python paper/md2latex.py

# ── Web App ─────────────────────────────────────────────────────────

web: ## Start web app dev server
	cd web && npm run dev

web-build: ## Build web app for production
	cd web && npm run build

# ── Cleanup ─────────────────────────────────────────────────────────

clean: ## Remove generated artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache
