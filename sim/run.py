from __future__ import annotations

import argparse
import json
from pathlib import Path

from swarm_kernel import Config, SwarmKernel, Callbacks


class PrintCB(Callbacks):
    def on_step(self, t: int, k: SwarmKernel):
        if t % 500 == 0:
            snap = k.snapshot()
            m = snap["metrics"]
            print(
                f"t={snap['t']:6d} gen={snap['gen']} | "
                f"zipf={m['zipf_slope']:+.2f} heapsK={m['heaps_k']:.3f} "
                f"H={m['cond_entropy']:.2f} topo={m['topo_similarity']:+.2f} "
                f"churn={m['churn']:.2f} | top={snap['top_forms'][:5]}"
            )

    def on_generation(self, gen: int, k: SwarmKernel):
        print(f"-- generation {gen} --")


def load_config(path: Path | None) -> Config:
    if path is None:
        return Config()
    with path.open() as f:
        data = json.load(f)
    cfg = Config(**{k: v for k, v in data.items() if hasattr(Config, k)})
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None, help="Path to JSON config")
    ap.add_argument("--steps", type=int, default=5000, help="Number of steps to run")
    ap.add_argument("--outdir", type=Path, default=Path("sim/runs"), help="Base output directory for runs")
    ap.add_argument("--label", type=str, default=None, help="Optional run label for directory name")
    args = ap.parse_args()

    cfg = load_config(args.config)
    kern = SwarmKernel(cfg)
    # Prepare output directory
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"_{args.label}" if args.label else ""
    run_dir = args.outdir / f"{ts}{label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    snapshot_path = run_dir / "snapshot.json"

    class LogCB(PrintCB):
        def on_step(self, t: int, k: SwarmKernel):
            super().on_step(t, k)
            if t % 25 == 0:
                snap = k.snapshot()
                rec = {
                    "t": snap["t"],
                    "gen": snap["gen"],
                    **{f"m_{k}": v for k, v in snap["metrics"].items()},
                }
                with metrics_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
        def on_generation(self, gen: int, k: SwarmKernel):
            super().on_generation(gen, k)

    kern.run(steps=args.steps, callbacks=LogCB())
    with snapshot_path.open("w") as f:
        json.dump(kern.snapshot(), f, indent=2)
    print(json.dumps({"run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
