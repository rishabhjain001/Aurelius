"""
train.py — Training pipeline for Aurelius CLIP-VLM

Implements:
    - CheckpointManager
    - TrainingResult
    - run_training(config, experiment_id)

Requirements: 1.1, 7.7, 12.1, 12.2, 12.3, 22.1, 23.1, 23.3,
              24.1, 24.2, 24.3, 24.4, 24.5, 24.6, 24.7, 24.8, 24.9, 25.3
"""

import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import psutil
import torch
from torch.optim import AdamW

from data import make_loaders
from model import CLIPModel


# Structural config fields whose values determine checkpoint compatibility.
# A mismatch here means the on-disk weights are shape-incompatible with the
# current run and must not be loaded (Req 24.5).
_STRUCTURAL_KEYS = ("embed_dim", "projection_head", "image_encoder", "text_encoder")


def _config_hash(config) -> str:
    """Hash of structural config fields used to detect checkpoint incompatibility."""
    def as_plain(obj):
        if hasattr(obj, "__dict__"):
            return {k: as_plain(v) for k, v in vars(obj).items()}
        if isinstance(obj, dict):
            return {k: as_plain(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [as_plain(v) for v in obj]
        return obj

    structural = {
        k: as_plain(getattr(config, k))
        for k in _STRUCTURAL_KEYS
        if hasattr(config, k)
    }
    payload = json.dumps(structural, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _accuracy_at_1(similarity: torch.Tensor) -> tuple[float, float]:
    """Return (image→text acc@1, text→image acc@1) from a similarity matrix."""
    n = similarity.shape[0]
    if n == 0:
        return 0.0, 0.0
    targets = torch.arange(n, device=similarity.device)
    i2t = (similarity.argmax(dim=1) == targets).float().mean().item()
    t2i = (similarity.argmax(dim=0) == targets).float().mean().item()
    return i2t, t2i


class CheckpointManager:
    """Save, locate, and restore training checkpoints for a given experiment."""

    def __init__(self, checkpoints_dir: str = "checkpoints") -> None:
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model, optimizer, step: int, experiment_id: str, config) -> str:
        """Serialize model, optimizer, RNG, and step to checkpoints/{id}_step{N}.pt."""
        checkpoint = {
            "experiment_id": experiment_id,
            "step": step,
            "config_hash": _config_hash(config),
            "embed_dim": config.embed_dim,
            "projection_head": config.projection_head,
            "image_encoder_state": model.image_encoder.state_dict(),
            "text_encoder_state": model.text_encoder.state_dict(),
            "temperature": model.loss_module.log_temperature.detach().exp().item(),
            "log_temperature_state": model.loss_module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "rng_torch": torch.get_rng_state(),
            "rng_numpy": np.random.get_state(),
            "rng_random": random.getstate(),
        }
        path = self.checkpoints_dir / f"{experiment_id}_step{step:08d}.pt"
        torch.save(checkpoint, path)
        return str(path)

    def find_latest(self, experiment_id: str) -> Optional[Path]:
        """Return path of most recent checkpoint for experiment_id, or None."""
        matches = sorted(self.checkpoints_dir.glob(f"{experiment_id}_step*.pt"))
        return matches[-1] if matches else None

    def load(self, path, model, optimizer, config) -> int:
        """Restore state from checkpoint at path. Returns the step index recorded in the file."""
        try:
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to load checkpoint '{path}': {exc}") from exc

        expected = _config_hash(config)
        if checkpoint.get("config_hash") != expected:
            mismatches = []
            saved_dim = checkpoint.get("embed_dim")
            if saved_dim != config.embed_dim:
                mismatches.append(
                    f"embed_dim: checkpoint={saved_dim}, config={config.embed_dim}"
                )
            saved_head = checkpoint.get("projection_head")
            if saved_head != config.projection_head:
                mismatches.append(
                    f"projection_head: checkpoint={saved_head}, "
                    f"config={config.projection_head}"
                )
            if not mismatches:
                mismatches.append(
                    f"config_hash: checkpoint={checkpoint.get('config_hash')}, "
                    f"config={expected}"
                )
            raise ValueError(
                "Checkpoint incompatible with current config: " + "; ".join(mismatches)
            )

        model.image_encoder.load_state_dict(checkpoint["image_encoder_state"])
        model.text_encoder.load_state_dict(checkpoint["text_encoder_state"])
        model.loss_module.load_state_dict(checkpoint["log_temperature_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        torch.set_rng_state(checkpoint["rng_torch"])
        np.random.set_state(checkpoint["rng_numpy"])
        random.setstate(checkpoint["rng_random"])
        return int(checkpoint["step"])


@dataclass
class TrainingResult:
    """Structured outcome of a single training run."""

    experiment_id: str
    steps: list = field(default_factory=list)
    status: str = "completed"
    failure_reason: Optional[str] = None
    final_step: int = 0
    final_checkpoint_path: Optional[str] = None



def run_training(
    config,
    experiment_id: str,
    train_loader=None,
    val_loader=None,
) -> TrainingResult:
    """Run one full training run for experiment_id, resuming from any existing checkpoint."""
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIPModel(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    manager = CheckpointManager()
    start_step = 0
    latest = manager.find_latest(experiment_id)
    if latest is not None:
        start_step = manager.load(latest, model, optimizer, config)
        print(f"Resumed from step {start_step}")

    if train_loader is None:
        train_loader, val_loader = make_loaders(config)

    steps_per_epoch = max(1, len(train_loader))
    max_steps = config.epochs * steps_per_epoch
    runtime_budget_s = float(config.runtime_budget_hours) * 3600.0
    memory_budget_b = float(config.memory_budget_gb) * (1024 ** 3)
    interval = int(config.checkpoint_interval)

    result_path = Path("results") / f"{experiment_id}.jsonl"
    step_records: list = []
    start_time = time.perf_counter()
    process = psutil.Process()

    step = start_step
    status = "completed"
    failure_reason: Optional[str] = None
    final_ckpt: Optional[str] = None
    last_saved = start_step

    stop = False
    for _ in range(config.epochs):
        if stop or step >= max_steps:
            break
        for images, captions in train_loader:
            if step >= max_steps:
                break

            images = images.to(device)

            optimizer.zero_grad()
            loss, similarity = model(images, captions)
            loss.backward()
            optimizer.step()

            step += 1
            i2t, t2i = _accuracy_at_1(similarity.detach())
            record = {
                "step": step,
                "loss": float(loss.detach().item()),
                "temperature": float(model.loss_module.log_temperature.detach().exp().item()),
                "i2t_acc1": i2t,
                "t2i_acc1": t2i,
            }
            step_records.append(record)
            with result_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

            if step % interval == 0:
                print(f"Saving checkpoint at step {step}")
                final_ckpt = manager.save(model, optimizer, step, experiment_id, config)
                last_saved = step

            elapsed = time.perf_counter() - start_time
            rss = process.memory_info().rss
            if elapsed > runtime_budget_s:
                status = "partial"
                failure_reason = (
                    f"Runtime budget exceeded: {elapsed:.1f}s > "
                    f"{runtime_budget_s:.1f}s"
                )
                print(f"Budget exceeded, stopping: {failure_reason}")
                stop = True
                break
            if rss > memory_budget_b:
                status = "partial"
                failure_reason = (
                    f"Memory budget exceeded: {rss / (1024 ** 3):.2f}GB > "
                    f"{config.memory_budget_gb}GB"
                )
                print(f"Budget exceeded, stopping: {failure_reason}")
                stop = True
                break

    # Final checkpoint reflects the last completed step so resume is possible (Req 24.9).
    if step > last_saved:
        final_ckpt = manager.save(model, optimizer, step, experiment_id, config)

    return TrainingResult(
        experiment_id=experiment_id,
        steps=step_records,
        status=status,
        failure_reason=failure_reason,
        final_step=step,
        final_checkpoint_path=final_ckpt,
    )
