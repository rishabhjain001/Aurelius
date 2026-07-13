"""
run_exp3to6.py — Resume pipeline from Experiment 3 onwards.

Experiments 1 and 2 are already completed. This script runs 3, 4, 5, 6,
then benchmark → visualize → generate_readme.
"""

import logging
from pathlib import Path
from types import SimpleNamespace

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_exp3to6")


def _ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(v) for v in obj]
    return obj


def step(msg):
    log.info("=" * 72)
    log.info("  %s", msg)
    log.info("=" * 72)


def main():
    step("Loading config")
    with open("configs/default.yaml", "r", encoding="utf-8") as fh:
        cfg = _ns(yaml.safe_load(fh))

    step("Loading dataset from disk cache")
    from data import _load_flickr30k
    hf_dataset = _load_flickr30k()
    log.info("Dataset ready: %d images", len(hf_dataset))

    step("Running Experiments 3, 4, 5, 6")
    from experiments import ExperimentRunner
    runner = ExperimentRunner(cfg, dataset=hf_dataset)

    for run_fn, num in (
        (runner.run_experiment_3, 3),
        (runner.run_experiment_4, 4),
        (runner.run_experiment_5, 5),
        (runner.run_experiment_6, 6),
    ):
        step(f"Experiment {num}")
        try:
            run_fn()
            log.info("Experiment %d done.", num)
        except Exception as exc:
            log.error("Experiment %d raised: %s", num, exc)

    step("Benchmark")
    from train import CheckpointManager
    ckpt = CheckpointManager().find_latest("experiment_1")
    if ckpt:
        import benchmark as bm
        bm.run_benchmark(str(ckpt), cfg)
    else:
        log.warning("No experiment_1 checkpoint found — skipping benchmark.")

    step("Visualization")
    import visualize
    visualize.run_visualization()

    step("README generation")
    import generate_readme
    generate_readme.run()

    step("All done!")
    log.info("Results  → results/")
    log.info("Plots    → plots/")
    log.info("README   → README.md")


if __name__ == "__main__":
    main()
