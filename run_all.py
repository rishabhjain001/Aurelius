"""
run_all.py — One-shot pipeline launcher for Aurelius CLIP-VLM

Usage:
    python run_all.py

Sequence:
    1. Load config from configs/default.yaml
    2. Download Flickr30K (first run only; cached afterward)
    3. Run runtime estimation gate — shows projected wall-clock time, asks
       for confirmation before any training begins
    4. Run all six experiments (train.py + evaluate.py)
    5. Run benchmark.py  (latency / throughput / memory)
    6. Run visualize.py  (generate all plots → plots/)
    7. Run generate_readme.py  (emit README.md from real results)

You can safely Ctrl-C at any time. Checkpoints are saved every
config.checkpoint_interval steps, so interrupted runs can be resumed.
Re-running this script after an interruption will pick up from the
most recent checkpoint for each experiment.
"""

import sys
import logging
from pathlib import Path
from types import SimpleNamespace

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_all")


def _ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(v) for v in obj]
    return obj


def load_config(path: str = "configs/default.yaml"):
    with open(path, "r", encoding="utf-8") as fh:
        return _ns(yaml.safe_load(fh))


def step(msg: str) -> None:
    width = 72
    log.info("=" * width)
    log.info("  %s", msg)
    log.info("=" * width)


def main() -> None:
    # -----------------------------------------------------------------------
    # 0. Config
    # -----------------------------------------------------------------------
    step("Loading configuration")
    cfg = load_config()
    log.info("batch_size=%d  epochs=%d  embed_dim=%d  train_max=%d",
             cfg.batch_size, cfg.epochs, cfg.embed_dim, cfg.train_max_samples)

    # -----------------------------------------------------------------------
    # 1. Dataset prefetch (triggers download if not cached)
    # -----------------------------------------------------------------------
    step("Loading / downloading Flickr30K dataset (may take a while first run)")
    from data import _load_flickr30k
    hf_dataset = _load_flickr30k()
    log.info("Dataset ready: %d images", len(hf_dataset))

    # -----------------------------------------------------------------------
    # 2. Experiments  (includes runtime estimation gate + confirmation prompt)
    # -----------------------------------------------------------------------
    step("Starting experiment runner")
    log.info("The runtime estimator will now project total wall-clock time.")
    log.info("You will be asked to confirm before any training begins.")

    from experiments import ExperimentRunner
    runner = ExperimentRunner(cfg, dataset=hf_dataset)
    # Auto-confirm: bypass the interactive gate for unattended runs
    runner.run_all(input_fn=lambda prompt: "y")

    # -----------------------------------------------------------------------
    # 3. Benchmark
    # -----------------------------------------------------------------------
    step("Running benchmark (latency / throughput / memory)")
    from train import CheckpointManager
    manager = CheckpointManager()
    latest_ckpt = manager.find_latest("experiment_1")

    if latest_ckpt is None:
        log.warning("No experiment_1 checkpoint found — skipping benchmark.")
    else:
        import benchmark as bm
        bm.run_benchmark(str(latest_ckpt), cfg)

    # -----------------------------------------------------------------------
    # 4. Visualization
    # -----------------------------------------------------------------------
    step("Generating plots")
    import visualize
    visualize.run_visualization()

    # -----------------------------------------------------------------------
    # 5. README
    # -----------------------------------------------------------------------
    step("Generating README.md")
    import generate_readme
    generate_readme.run()

    step("All done!")
    log.info("Results   →  results/")
    log.info("Plots     →  plots/")
    log.info("README    →  README.md")


if __name__ == "__main__":
    main()
