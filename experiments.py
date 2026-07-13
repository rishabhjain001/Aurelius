"""
experiments.py — Experiment_Runner for Aurelius CLIP-VLM

Orchestrates all six experiments in strict numeric order with a runtime
estimation gate, reproducibility metadata, random-encoder baselines, and
graceful failure isolation.

Public interface:
    - RuntimeEstimator
    - ReproducibilityLogger
    - random_baseline_model
    - ExperimentRunner

Requirements: 3.1-3.6, 10.1-10.5, 11.1-11.7, 12.1-12.3, 13.1-13.4,
              14.1-14.5, 15.1-15.4, 16.1-16.5, 17.1-17.5,
              22.1-22.11, 23.1-23.6, 25.3
"""

import copy
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import psutil
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset

from evaluate import (
    retrieval_metrics_from_model,
    zero_shot_metrics,
    lock_prompt_set,
    PROMPT_SET,
)
from model import CLIPModel
from train import CheckpointManager, TrainingResult, run_training


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _to_plain(obj):
    """Recursive vars()-based conversion for JSON serialization."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {k: _to_plain(v) for k, v in vars(obj).items()}
    return str(obj)


def _dm_to_dict(m):
    """Convert DirectionalMetrics to plain dict, or None."""
    if m is None:
        return None
    return {
        "r1": float(m.r1),
        "r5": float(m.r5),
        "r10": float(m.r10),
        "mrr": float(m.mrr),
        "mean_rank": float(m.mean_rank),
        "median_rank": float(m.median_rank),
    }


def _retrieval_to_dict(result):
    """Convert RetrievalResult to plain dict, or None."""
    if result is None:
        return None
    if getattr(result, "skipped", False):
        return {
            "skipped": True,
            "skip_reason": result.skip_reason,
            "i2t": None,
            "t2i": None,
        }
    return {
        "skipped": False,
        "skip_reason": None,
        "i2t": _dm_to_dict(result.i2t),
        "t2i": _dm_to_dict(result.t2i),
    }


# ---------------------------------------------------------------------------
# Runtime estimator
# ---------------------------------------------------------------------------

class RuntimeEstimator:
    """Times one training step and one eval step, projects total wall-clock time."""

    def estimate(
        self,
        config,
        model_factory,
        sample_batch,
        steps_per_epoch: Optional[int] = None,
        n_val_batches: Optional[int] = None,
    ) -> float:
        images, captions = sample_batch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Timed training step: forward, backward, optimizer step.
        train_model = model_factory().to(device)
        train_model.train()
        optimizer = AdamW(train_model.parameters(), lr=1e-3)
        images_dev = images.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss, _ = train_model(images_dev, captions)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        per_step_train_s = time.perf_counter() - t0

        # Timed evaluation-only forward step.
        eval_model = model_factory().to(device)
        eval_model.eval()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            eval_model(images_dev, captions)
        if device.type == "cuda":
            torch.cuda.synchronize()
        per_step_eval_s = time.perf_counter() - t0

        if steps_per_epoch is None:
            steps_per_epoch = int(getattr(config, "steps_per_epoch", 100))
        if n_val_batches is None:
            n_val_batches = int(getattr(config, "n_val_batches", 50))

        epochs = int(config.epochs)
        full_train_run = float(epochs) * float(steps_per_epoch) * per_step_train_s
        full_eval_run = float(n_val_batches) * per_step_eval_s

        # Exp 1: 1 train + 1 eval
        # Exp 2: 1 eval only (reuses Exp 1 model)
        # Exp 3: 3 train
        # Exp 4: 2 train
        # Exp 5: 3 train
        # Exp 6: 2 train
        n_train_runs = 1 + 3 + 2 + 3 + 2
        n_eval_runs = 1 + 1
        return n_train_runs * full_train_run + n_eval_runs * full_eval_run

    def format_hms(self, seconds: float) -> str:
        """Format seconds as `Hh Mm Ss`."""
        seconds = int(max(0, round(seconds)))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h}h {m}m {s}s"

    def confirm(self, projected_seconds: float, input_fn=input) -> bool:
        """Print projection and require an affirmative developer confirmation."""
        print(f"Projected total wall-clock time: {self.format_hms(projected_seconds)}")
        try:
            response = input_fn("Proceed with all experiments? [y/N]: ")
        except (EOFError, KeyboardInterrupt):
            return False
        if not isinstance(response, str) or not response:
            return False
        return response[0] in ("y", "Y")


# ---------------------------------------------------------------------------
# Reproducibility logger
# ---------------------------------------------------------------------------

class ReproducibilityLogger:
    """Records reproducibility metadata to results/experimentN_metadata.json."""

    def snapshot(
        self,
        config,
        seeds: dict,
        resumed_from: Optional[str] = None,
        resume_step: Optional[int] = None,
        failure_reason: Optional[str] = None,
    ) -> dict:
        meta: dict = {}

        try:
            meta["config_snapshot"] = _to_plain(config)
        except Exception as exc:
            meta["config_snapshot"] = None
            meta["config_snapshot_unavailable_reason"] = str(exc)

        meta["seeds"] = {
            "torch": int(seeds.get("torch", 0)),
            "numpy": int(seeds.get("numpy", 0)),
            "random": int(seeds.get("random", 0)),
        }

        # Git commit hash — literal "unavailable" if git call fails (Req 22.4).
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode("utf-8", errors="replace").strip()
            meta["git_commit"] = commit if commit else "unavailable"
        except Exception as exc:
            meta["git_commit"] = "unavailable"
            meta["git_commit_unavailable_reason"] = str(exc)

        try:
            status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode("utf-8", errors="replace")
            meta["git_dirty"] = bool(status.strip())
        except Exception as exc:
            meta["git_dirty"] = None
            meta["git_dirty_unavailable_reason"] = str(exc)

        meta["python_version"] = sys.version
        meta["pytorch_version"] = torch.__version__
        meta["timestamp"] = datetime.now(timezone.utc).isoformat()

        try:
            cpu_count = psutil.cpu_count()
            total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_count = torch.cuda.device_count()
            else:
                gpu_name = None
                gpu_count = None
            meta["hardware"] = {
                "cpu_count": cpu_count,
                "total_ram_gb": float(total_ram_gb),
                "gpu_name": gpu_name,
                "gpu_count": gpu_count,
            }
        except Exception as exc:
            meta["hardware"] = None
            meta["hardware_unavailable_reason"] = str(exc)

        meta["resumed_from"] = resumed_from
        meta["resume_step"] = resume_step
        meta["failure_reason"] = failure_reason
        return meta

    def write(self, experiment_num: int, metadata: dict) -> str:
        """Serialize metadata to results/experimentN_metadata.json (Req 22.11)."""
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        path = results_dir / f"experiment{experiment_num}_metadata.json"
        try:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2, default=str)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to write metadata file '{path}': {exc}"
            ) from exc
        return str(path)


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

def random_baseline_model(config) -> CLIPModel:
    """Fresh, untrained CLIPModel matching the trained comparator's shape."""
    return CLIPModel(config)


# ---------------------------------------------------------------------------
# Validation dataset returning all captions per image
# ---------------------------------------------------------------------------

class _AllCaptionsDataset(Dataset):
    """Validation split view yielding (image_tensor, list_of_all_captions)."""

    def __init__(self, hf_dataset, indices, transform):
        self.hf = hf_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        item = self.hf[self.indices[idx]]
        if "captions" in item:
            captions = item["captions"]
        elif "caption" in item:
            captions = item["caption"]
        else:
            captions = [""]
        if not isinstance(captions, list):
            captions = [captions]
        image = self.transform(item["image"])
        return image, list(captions)


def _build_val_dataset(config, hf_dataset=None):
    """Build validation dataset (all captions) using the same split as data.py."""
    from data import _build_base_transform, _load_flickr30k, _split_indices

    if hf_dataset is None:
        hf_dataset = _load_flickr30k()
    _, val_idx = _split_indices(
        len(hf_dataset), config.val_split_fraction, config.val_split_seed
    )
    transform = _build_base_transform(
        config.image_size, config.image_mean, config.image_std
    )
    return _AllCaptionsDataset(hf_dataset, val_idx, transform), hf_dataset


# ---------------------------------------------------------------------------
# Analysis helpers (embedding separation, training stability)
# ---------------------------------------------------------------------------

def _embedding_separation(model, val_dataset) -> dict:
    """Mean/std of matched vs unmatched image-caption cosine similarities."""
    if len(val_dataset) == 0:
        return {
            "matched_mean": None,
            "matched_std": None,
            "unmatched_mean": None,
            "unmatched_std": None,
            "skip_reason": "empty validation split",
        }

    model.eval()
    matched_sims: list = []
    all_image_embeds: list = []
    all_text_embeds: list = []
    caption_to_image: list = []

    with torch.no_grad():
        for i in range(len(val_dataset)):
            image, captions = val_dataset[i]
            img_e = model.encode_image(image.unsqueeze(0))    # [1, D]
            txt_e = model.encode_text(captions)               # [K, D]
            all_image_embeds.append(img_e)
            all_text_embeds.append(txt_e)
            caption_to_image.extend([i] * len(captions))
            sims = (img_e @ txt_e.T).squeeze(0)               # [K]
            matched_sims.append(float(sims.mean().item()))

    image_embeds = torch.cat(all_image_embeds, dim=0)         # [N, D]
    text_embeds = torch.cat(all_text_embeds, dim=0)           # [M, D]
    full = (image_embeds @ text_embeds.T).cpu().numpy()       # [N, M]

    cap_to_img_arr = np.asarray(caption_to_image, dtype=np.int64)
    n_images = full.shape[0]
    row_idx = np.arange(n_images).reshape(-1, 1)
    mask_unmatched = cap_to_img_arr.reshape(1, -1) != row_idx  # [N, M]
    unmatched_vals = full[mask_unmatched]

    matched_arr = np.asarray(matched_sims, dtype=np.float64)
    if unmatched_vals.size == 0:
        return {
            "matched_mean": float(matched_arr.mean()),
            "matched_std": float(matched_arr.std()),
            "unmatched_mean": None,
            "unmatched_std": None,
            "skip_reason": "no unmatched pairs available",
        }
    return {
        "matched_mean": float(matched_arr.mean()),
        "matched_std": float(matched_arr.std()),
        "unmatched_mean": float(unmatched_vals.mean()),
        "unmatched_std": float(unmatched_vals.std()),
    }


def _training_stability(steps: list, window: int = 100) -> dict:
    """Variance of loss across sliding non-overlapping windows."""
    if not steps:
        return {
            "window_size": window,
            "windows": [],
            "mean_variance": None,
            "max_variance": None,
        }
    losses = np.asarray([s["loss"] for s in steps], dtype=np.float64)
    if len(losses) < window:
        var = float(losses.var()) if len(losses) > 1 else 0.0
        return {
            "window_size": window,
            "windows": [var],
            "mean_variance": var,
            "max_variance": var,
        }
    windows = [
        float(losses[i:i + window].var())
        for i in range(0, len(losses) - window + 1, window)
    ]
    return {
        "window_size": window,
        "windows": windows,
        "mean_variance": float(np.mean(windows)),
        "max_variance": float(np.max(windows)),
    }


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """Executes Experiments 1–6 in strict numeric order with failure isolation."""

    def __init__(self, config, dataset=None):
        self.config = config
        self.dataset = dataset
        self.logger = ReproducibilityLogger()
        self.estimator = RuntimeEstimator()

    def _seeds_for(self, experiment_num: int) -> dict:
        seed_cfg = getattr(self.config, "experiment_seeds", None)
        base = 0
        if seed_cfg is not None:
            key = f"experiment_{experiment_num}"
            if hasattr(seed_cfg, key):
                base = getattr(seed_cfg, key)
            elif isinstance(seed_cfg, dict) and key in seed_cfg:
                base = seed_cfg[key]
        base = int(base) if base is not None else 0
        return {"torch": base, "numpy": base, "random": base}

    def _apply_seeds(self, seeds: dict) -> None:
        random.seed(seeds["random"])
        np.random.seed(seeds["numpy"])
        torch.manual_seed(seeds["torch"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seeds["torch"])

    def write_result(self, experiment_num: int, result: dict) -> None:
        """Write results/experimentN.json; on failure, mark experiment failed."""
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        path = results_dir / f"experiment{experiment_num}.json"
        try:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, default=str)
        except Exception as exc:
            print(f"Failed to write result file '{path}': {exc}")
            # Reflect write failure in metadata (Req 11.7).
            try:
                fail_meta = self.logger.snapshot(
                    self.config,
                    self._seeds_for(experiment_num),
                    failure_reason=f"Result file write failed: {exc}",
                )
                fail_meta["experiment"] = experiment_num
                self.logger.write(experiment_num, fail_meta)
            except Exception as meta_exc:
                print(f"Failed to write metadata after write error: {meta_exc}")

    def _write_metadata(
        self,
        experiment_num: int,
        seeds: dict,
        failure_reason: Optional[str] = None,
        resumed_from: Optional[str] = None,
        resume_step: Optional[int] = None,
    ) -> None:
        meta = self.logger.snapshot(
            self.config,
            seeds,
            resumed_from=resumed_from,
            resume_step=resume_step,
            failure_reason=failure_reason,
        )
        meta["experiment"] = experiment_num
        self.logger.write(experiment_num, meta)

    def _reload_model(self, config, checkpoint_path: str) -> Optional[CLIPModel]:
        """Rebuild a CLIPModel and load state from a checkpoint path."""
        if checkpoint_path is None:
            return None
        try:
            model = CLIPModel(config)
            optimizer = AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            manager = CheckpointManager()
            manager.load(checkpoint_path, model, optimizer, config)
            model.eval()
            return model
        except Exception as exc:
            print(f"Failed to reload model from '{checkpoint_path}': {exc}")
            return None

    def _run_single(self, config, experiment_id: str):
        """Run one training pass; returns (training_result, reloaded_model_or_None)."""
        try:
            training_result = run_training(config, experiment_id)
        except Exception as exc:
            return (
                TrainingResult(
                    experiment_id=experiment_id,
                    status="failed",
                    failure_reason=str(exc),
                ),
                None,
            )
        model = self._reload_model(config, training_result.final_checkpoint_path)
        return training_result, model

    # -----------------------------------------------------------------
    # Individual experiments
    # -----------------------------------------------------------------

    def run_experiment_1(self):
        """Exp 1: exactly one training run + retrieval eval on validation split."""
        seeds = self._seeds_for(1)
        self._apply_seeds(seeds)

        result = {
            "experiment": 1,
            "status": "failed",
            "failure_reason": None,
            "steps": [],
            "retrieval": None,
        }
        model = None
        try:
            training_result, model = self._run_single(self.config, "experiment_1")
            result["steps"] = training_result.steps
            result["status"] = training_result.status
            result["failure_reason"] = training_result.failure_reason

            if model is not None and result["status"] in ("completed", "partial"):
                try:
                    val_ds, _ = _build_val_dataset(self.config, self.dataset)
                    retrieval = retrieval_metrics_from_model(model, val_ds)
                    result["retrieval"] = _retrieval_to_dict(retrieval)
                except Exception as exc:
                    result["retrieval"] = None
                    result["retrieval_unavailable_reason"] = str(exc)
        except Exception as exc:
            result["status"] = "failed"
            result["failure_reason"] = str(exc)
            model = None

        self._write_metadata(1, seeds, failure_reason=result["failure_reason"])
        self.write_result(1, result)
        if result["status"] in ("completed", "partial"):
            return model
        return None

    def run_experiment_2(self, exp1_model):
        """Exp 2: reuse Exp 1 model; compare trained vs random baseline retrieval."""
        seeds = self._seeds_for(2)
        self._apply_seeds(seeds)

        result = {
            "experiment": 2,
            "status": "failed",
            "failure_reason": None,
            "retrieval": {
                "trained": None,
                "random_baseline": None,
            },
        }

        if exp1_model is None:
            result["status"] = "failed"
            result["failure_reason"] = "Experiment 1 model unavailable"
            self._write_metadata(2, seeds, failure_reason=result["failure_reason"])
            self.write_result(2, result)
            return

        try:
            val_ds, _ = _build_val_dataset(self.config, self.dataset)
            try:
                trained_retrieval = retrieval_metrics_from_model(exp1_model, val_ds)
                result["retrieval"]["trained"] = _retrieval_to_dict(trained_retrieval)
            except Exception as exc:
                result["retrieval"]["trained"] = None
                result["retrieval"]["trained_unavailable_reason"] = str(exc)

            try:
                baseline = random_baseline_model(self.config)
                baseline.eval()
                baseline_retrieval = retrieval_metrics_from_model(baseline, val_ds)
                result["retrieval"]["random_baseline"] = _retrieval_to_dict(
                    baseline_retrieval
                )
            except Exception as exc:
                result["retrieval"]["random_baseline"] = None
                result["retrieval"]["random_baseline_unavailable_reason"] = str(exc)

            result["status"] = "completed"
        except Exception as exc:
            result["status"] = "failed"
            result["failure_reason"] = str(exc)

        self._write_metadata(2, seeds, failure_reason=result["failure_reason"])
        self.write_result(2, result)

    def _run_ablation(
        self,
        experiment_num: int,
        overrides: list,
        override_key: str,
        extra_analysis=None,
    ):
        """Shared driver for ablation experiments (3, 4, 5, 6)."""
        seeds = self._seeds_for(experiment_num)
        self._apply_seeds(seeds)

        result = {
            "experiment": experiment_num,
            "status": "completed",
            "failure_reason": None,
            "runs": [],
        }
        any_failure = False

        for value in overrides:
            run_config = copy.deepcopy(self.config)
            setattr(run_config, override_key, value)
            run_id = f"experiment_{experiment_num}_{override_key}_{value}"

            run_entry = {
                override_key: value,
                "run_id": run_id,
                "status": "failed",
                "failure_reason": None,
                "retrieval": None,
                "convergence": None,
            }

            try:
                training_result, model = self._run_single(run_config, run_id)
                run_entry["status"] = training_result.status
                run_entry["failure_reason"] = training_result.failure_reason
                run_entry["convergence"] = training_result.steps

                if model is not None and training_result.status in (
                    "completed",
                    "partial",
                ):
                    try:
                        val_ds, _ = _build_val_dataset(run_config, self.dataset)
                        retrieval = retrieval_metrics_from_model(model, val_ds)
                        run_entry["retrieval"] = _retrieval_to_dict(retrieval)
                        if extra_analysis is not None:
                            run_entry.update(
                                extra_analysis(model, val_ds, training_result)
                            )
                    except Exception as exc:
                        run_entry["retrieval"] = None
                        run_entry["retrieval_unavailable_reason"] = str(exc)
                else:
                    if run_entry["failure_reason"] is None:
                        run_entry["failure_reason"] = "training did not produce a usable model"
            except Exception as exc:
                run_entry["status"] = "failed"
                run_entry["failure_reason"] = str(exc)

            if run_entry["status"] not in ("completed",):
                any_failure = True
            result["runs"].append(run_entry)

        if any_failure and all(r["status"] == "failed" for r in result["runs"]):
            result["status"] = "failed"
            result["failure_reason"] = "all runs failed"
        elif any_failure:
            result["status"] = "partial"
            result["failure_reason"] = "one or more runs failed or were partial"

        self._write_metadata(
            experiment_num, seeds, failure_reason=result["failure_reason"]
        )
        self.write_result(experiment_num, result)

    def run_experiment_3(self):
        """Exp 3: three training runs, embed_dim ∈ {128, 256, 512}."""
        self._run_ablation(3, [128, 256, 512], "embed_dim")

    def run_experiment_4(self):
        """Exp 4: two training runs, projection_head ∈ {linear, mlp}."""
        def analysis(model, val_ds, _training_result):
            return {"embedding_separation": _embedding_separation(model, val_ds)}
        self._run_ablation(4, ["linear", "mlp"], "projection_head", extra_analysis=analysis)

    def run_experiment_5(self):
        """Exp 5: three training runs, temperature_init ∈ {0.05, 0.07, 0.10}."""
        self._run_ablation(5, [0.05, 0.07, 0.10], "temperature_init")

    def run_experiment_6(self):
        """Exp 6: two training runs, augmentation ∈ {weak, strong}."""
        def analysis(_model, _val_ds, training_result):
            return {"training_stability": _training_stability(training_result.steps)}
        self._run_ablation(6, ["weak", "strong"], "augmentation_mode", extra_analysis=analysis)

    # -----------------------------------------------------------------
    # Top-level orchestration
    # -----------------------------------------------------------------

    def _sample_batch_and_shape(self):
        """Build loaders once; return a sample batch, steps_per_epoch, n_val_batches."""
        from data import make_loaders

        train_loader, val_loader = make_loaders(self.config, self.dataset)
        try:
            sample = next(iter(train_loader))
        except StopIteration:
            raise RuntimeError("Training loader produced no batches; cannot estimate runtime.")
        steps_per_epoch = max(1, len(train_loader))
        n_val_batches = max(1, len(val_loader))
        return sample, steps_per_epoch, n_val_batches

    def run_all(self, input_fn=input) -> None:
        """Run all six experiments in strict order after runtime confirmation."""
        try:
            sample_batch, steps_per_epoch, n_val_batches = self._sample_batch_and_shape()
        except Exception as exc:
            print(f"Failed to prepare data loaders for runtime estimate: {exc}")
            return

        projected = self.estimator.estimate(
            self.config,
            lambda: CLIPModel(self.config),
            sample_batch,
            steps_per_epoch=steps_per_epoch,
            n_val_batches=n_val_batches,
        )
        if not self.estimator.confirm(projected, input_fn=input_fn):
            print("Aborted: developer did not confirm runtime estimate.")
            return

        # Freeze PROMPT_SET before any experiment begins (Req 9.1, 9.3).
        lock_prompt_set()

        exp1_model = None
        try:
            exp1_model = self.run_experiment_1()
        except Exception as exc:
            print(f"Experiment 1 raised unexpectedly: {exc}")
            exp1_model = None

        # If Exp 1 failed, mark Exp 2 failed but continue with Exp 3–6 (Req 23.6).
        if exp1_model is None:
            seeds_2 = self._seeds_for(2)
            reason = "Experiment 1 model unavailable"
            failed_result = {
                "experiment": 2,
                "status": "failed",
                "failure_reason": reason,
                "retrieval": {"trained": None, "random_baseline": None},
            }
            self._write_metadata(2, seeds_2, failure_reason=reason)
            self.write_result(2, failed_result)
        else:
            try:
                self.run_experiment_2(exp1_model)
            except Exception as exc:
                print(f"Experiment 2 raised unexpectedly: {exc}")

        for run_fn, num in (
            (self.run_experiment_3, 3),
            (self.run_experiment_4, 4),
            (self.run_experiment_5, 5),
            (self.run_experiment_6, 6),
        ):
            try:
                run_fn()
            except Exception as exc:
                print(f"Experiment {num} raised unexpectedly: {exc}")


if __name__ == "__main__":
    # Minimal CLI entry: expects `configs/default.yaml`.
    import yaml
    from types import SimpleNamespace

    def _ns(obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [_ns(v) for v in obj]
        return obj

    with open("configs/default.yaml", "r", encoding="utf-8") as fh:
        cfg_dict = yaml.safe_load(fh)
    cfg = _ns(cfg_dict)
    ExperimentRunner(cfg).run_all()
