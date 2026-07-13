"""
benchmark.py — Benchmark_Tool for Aurelius CLIP-VLM

Measures image encoding latency, text encoding latency, end-to-end retrieval
query latency, and zero-shot inference latency at batch size 1. Samples memory
via psutil throughout and writes results/benchmark.json.

Requirements: 18.1, 18.2, 18.3, 18.4, 18.5, 18.6, 18.7, 25.3, 28.7
"""

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from evaluate import PROMPT_SET
from model import CLIPModel
from train import CheckpointManager


class BenchmarkRunner:
    """Runs all four latency measurements and samples memory usage."""

    def __init__(self, model: CLIPModel, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.model.eval()
        self._ram_samples: list = []

    def _measure(self, fn: Callable, n_iters: int = 100) -> list:
        """Run fn n_iters times, return list of latencies in milliseconds."""
        latencies = []
        for _ in range(n_iters):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                fn()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)
        return latencies

    def _stats(self, latencies: list) -> dict:
        """Compute mean, median, p95, std for a list of latencies."""
        arr = np.asarray(latencies, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "std": float(arr.std()),
        }

    def _sample_ram(self, stop_event: threading.Event) -> None:
        """Background thread: sample RSS every 100ms until stop_event is set."""
        proc = psutil.Process()
        while not stop_event.is_set():
            self._ram_samples.append(proc.memory_info().rss)
            time.sleep(0.1)

    def run(self) -> dict:
        """Run all four measurements, collect memory, return result dict."""
        # Start memory sampling thread before any measurement begins.
        self._ram_samples = []
        stop_event = threading.Event()
        sampler = threading.Thread(target=self._sample_ram, args=(stop_event,), daemon=True)
        sampler.start()

        # --- Image encoding latency ---
        dummy_image = torch.zeros(1, 3, 224, 224, device=self.device)
        img_latencies = self._measure(lambda: self.model.encode_image(dummy_image))

        # --- Text encoding latency ---
        dummy_caption = ["a photo of a dog"]
        txt_latencies = self._measure(lambda: self.model.encode_text(dummy_caption))

        # --- End-to-end retrieval query latency ---
        # Precompute a cache of 100 image embeddings once, outside timed loop.
        with torch.no_grad():
            cache_images = torch.randn(100, 3, 224, 224, device=self.device)
            cached_embeds = self.model.encode_image(cache_images)  # [100, D]
            cached_embeds = F.normalize(cached_embeds, dim=-1)

        def retrieval_query():
            query_embed = self.model.encode_text(["a photo of a cat"])  # [1, D]
            sims = query_embed @ cached_embeds.T                        # [1, 100]
            sims.argmax(dim=1)

        retrieval_latencies = self._measure(retrieval_query)

        # --- Zero-shot inference latency ---
        # Precompute class embeddings from PROMPT_SET once, outside timed loop.
        class_labels = list(PROMPT_SET.keys())
        with torch.no_grad():
            class_vecs = []
            for cls in class_labels:
                prompt_embeds = self.model.encode_text(PROMPT_SET[cls])  # [P, D]
                mean = prompt_embeds.mean(dim=0, keepdim=True)           # [1, D]
                class_vecs.append(F.normalize(mean, dim=-1).squeeze(0))
            class_embeds = torch.stack(class_vecs, dim=0)                # [C, D]

        def zero_shot_query():
            img_embed = self.model.encode_image(dummy_image)   # [1, D]
            sims = img_embed @ class_embeds.T                  # [1, C]
            sims.argmax(dim=1)

        zero_shot_latencies = self._measure(zero_shot_query)

        # Stop memory sampling.
        stop_event.set()
        sampler.join(timeout=2.0)

        # Compute throughput: items/sec = 1000 / latency_ms for each sample.
        all_latencies = (
            img_latencies + txt_latencies + retrieval_latencies + zero_shot_latencies
        )
        throughput_samples = [1000.0 / lat for lat in all_latencies if lat > 0]

        # Memory stats.
        if self._ram_samples:
            ram_mb = np.asarray(self._ram_samples, dtype=np.float64) / (1024 ** 2)
            peak_ram_mb = float(ram_mb.max())
            avg_ram_mb = float(ram_mb.mean())
        else:
            peak_ram_mb = 0.0
            avg_ram_mb = 0.0

        return {
            "image_encode_latency_ms": self._stats(img_latencies),
            "text_encode_latency_ms": self._stats(txt_latencies),
            "retrieval_query_latency_ms": self._stats(retrieval_latencies),
            "zero_shot_latency_ms": self._stats(zero_shot_latencies),
            "throughput_items_per_sec": self._stats(throughput_samples),
            "peak_ram_mb": peak_ram_mb,
            "avg_ram_mb": avg_ram_mb,
            "iterations": 100,
        }


def run_benchmark(checkpoint_path: str, config) -> None:
    """Load model from checkpoint, run BenchmarkRunner, write results/benchmark.json."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIPModel(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=getattr(config, "learning_rate", 5e-4),
        weight_decay=getattr(config, "weight_decay", 0.1),
    )
    manager = CheckpointManager()
    manager.load(checkpoint_path, model, optimizer, config)
    model = model.to(device)
    model.eval()

    runner = BenchmarkRunner(model, device)
    results = runner.run()

    Path("results").mkdir(parents=True, exist_ok=True)
    output_path = Path("results") / "benchmark.json"
    try:
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
    except Exception as exc:
        print(f"Benchmark error: failed to write '{output_path}': {exc}")
        raise SystemExit(1) from exc

    print(f"Benchmark results written to {output_path}")


def _find_latest_checkpoint() -> Optional[Path]:
    """Return the most recently modified .pt file in checkpoints/, or None."""
    ckpt_dir = Path("checkpoints")
    if not ckpt_dir.exists():
        return None
    pts = sorted(ckpt_dir.glob("*.pt"))
    return pts[-1] if pts else None


if __name__ == "__main__":
    import yaml

    def _ns(obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [_ns(v) for v in obj]
        return obj

    with open("configs/default.yaml", "r", encoding="utf-8") as fh:
        cfg_dict = yaml.safe_load(fh)
    cfg = _ns(cfg_dict)

    latest = _find_latest_checkpoint()
    if latest is None:
        print(
            "Benchmark refused: no completed training run found. "
            "Run training first (e.g. python experiments.py)."
        )
        raise SystemExit(1)

    print(f"Using checkpoint: {latest}")
    run_benchmark(str(latest), cfg)
