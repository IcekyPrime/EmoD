"""Accelerate training base: unified checkpointing, resume, eval and logging."""

from __future__ import annotations

import itertools
import math
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from emod.config import as_rel, get, resolve_path
from emod.utils.checkpoint import build_payload, latest_checkpoint, load_checkpoint, save_checkpoint
from emod.utils.logging import AverageMeter, JsonlWriter, get_logger
from emod.utils.misc import count_parameters, ensure_dir, format_seconds, resolve_batch_plan, set_seed


class BaseTrainer(ABC):
    stage: int = 0

    def __init__(self, cfg: Dict[str, Any], dry_run: bool = False) -> None:
        self.cfg = cfg
        self.dry_run = dry_run
        self.logger = get_logger(f"emod.stage{self.stage}")

        self.world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
        self.num_emotions = len(get(cfg, "data.emotions"))
        self.plan = resolve_batch_plan(cfg, self.world_size, self.num_emotions)
        self.accum_steps = int(self.plan["grad_accum_steps"])

        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=bool(get(cfg, "train.find_unused_parameters", False))
        )
        self.accelerator = Accelerator(
            gradient_accumulation_steps=max(1, self.accum_steps),
            mixed_precision=str(get(cfg, "runtime.mixed_precision", "no")),
            kwargs_handlers=[ddp_kwargs],
        )

        seed = int(get(cfg, "project.seed", 1234)) + self.accelerator.process_index
        set_seed(
            seed,
            deterministic=bool(get(cfg, "runtime.deterministic", False)),
            tf32=bool(get(cfg, "runtime.tf32", True)),
        )

        self.out_dir = ensure_dir(resolve_path(get(cfg, "paths.checkpoint_dir")))
        self.log_dir = ensure_dir(resolve_path(get(cfg, "paths.log_dir")))
        self.writer = JsonlWriter(self.log_dir / f"stage{self.stage}.jsonl")

        self.epochs = int(get(cfg, "train.epochs", 1))
        self.max_steps = get(cfg, "train.max_steps", None)
        self.max_steps = int(self.max_steps) if self.max_steps else None
        self.log_every = int(get(cfg, "train.log_every", 50))
        self.eval_every = int(get(cfg, "train.eval_every_steps", 1000))
        self.save_every = int(get(cfg, "train.save_every_steps", 2000))
        self.keep_last = int(get(cfg, "train.keep_last", 3))

        self.epoch = 0
        self.global_step = 0
        self.best_metric = float("inf")
        self.artifacts: Dict[str, Any] = {}
        self._skip_batches = 0
        self._start_time = time.time()

        if dry_run:
            self.epochs = 1
            self.max_steps = 2
            self.eval_every = 1
            self.save_every = 10**9
            self.log_every = 1

    # -- to implement ------------------------------------------------------
    @abstractmethod
    def build_model(self) -> torch.nn.Module:
        ...

    @abstractmethod
    def build_loaders(self) -> Tuple[Any, Optional[Any]]:
        ...

    @abstractmethod
    def training_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        ...

    def evaluation(self) -> Dict[str, float]:
        return {}

    def on_train_end(self) -> None:
        return None

    def primary_metric(self, metrics: Dict[str, float]) -> float:
        return metrics.get("val_loss", 0.0)

    # -- setup -------------------------------------------------------------
    def build_optimizer(self, model: torch.nn.Module):
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
        groups = [
            {"params": decay, "weight_decay": float(get(self.cfg, "train.weight_decay", 0.01))},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(groups, lr=float(get(self.cfg, "train.lr", 1e-4)))

        steps_per_epoch = self._steps_per_epoch()
        total = self.max_steps or max(1, steps_per_epoch * self.epochs)
        warmup = max(1, int(total * float(get(self.cfg, "train.warmup_ratio", 0.05))))

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / float(warmup)
            progress = (step - warmup) / float(max(1, total - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return optimizer, scheduler

    def _steps_per_epoch(self) -> int:
        sampler = getattr(self, "train_sampler", None)
        if sampler is not None:
            return len(sampler)
        loader = getattr(self, "train_loader", None)
        if loader is not None:
            return max(1, len(loader))
        return 1

    def setup(self) -> None:
        self.model = self.build_model()
        self.train_loader, self.val_loader = self.build_loaders()
        self.optimizer, self.scheduler = self.build_optimizer(self.model)

        self._restore()
        self.accelerator.wait_for_everyone()
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler
        )
        self.logger.info(
            "stage=%s world=%s per_gpu=%s accum=%s effective=%s params=%s",
            self.stage,
            self.world_size,
            self.plan["per_gpu_batch_size"],
            self.plan["grad_accum_steps"],
            self.plan["effective_batch_size"],
            f"{count_parameters(self.accelerator.unwrap_model(self.model), True)/1e6:.1f}M",
        )

    # -- checkpoint --------------------------------------------------------
    def _resume_path(self) -> Optional[Path]:
        value = get(self.cfg, "train.resume", "auto")
        if value in ("none", None, False):
            return None
        if value == "auto":
            return latest_checkpoint(self.out_dir)
        return resolve_path(value)

    def _restore(self) -> None:
        path = self._resume_path()
        if path is None or not Path(path).exists():
            return
        payload = load_checkpoint(path)
        if int(payload.get("stage", -1)) != self.stage:
            raise RuntimeError(f"cannot resume stage {self.stage} from {payload.get('stage')}")

        self.model.load_state_dict(payload["model"], strict=True)
        raw_opt = getattr(self.optimizer, "optimizer", self.optimizer)
        raw_sched = getattr(self.scheduler, "scheduler", self.scheduler)
        if "optimizer" in payload:
            raw_opt.load_state_dict(payload["optimizer"])
        if "scheduler" in payload:
            raw_sched.load_state_dict(payload["scheduler"])

        self.epoch = int(payload.get("epoch", 0))
        self.global_step = int(payload.get("step", 0))
        self.best_metric = float(payload.get("metrics", {}).get("best", self.best_metric))
        self.artifacts.update(payload.get("artifacts", {}))

        steps_per_epoch = max(1, self._steps_per_epoch())
        offset = self.global_step - self.epoch * steps_per_epoch
        if offset >= steps_per_epoch:
            # Checkpoint taken at an epoch boundary: continue with the next epoch.
            self.epoch += 1
            self._skip_batches = 0
        else:
            self._skip_batches = max(0, offset)
        self.logger.info(
            "resumed %s at epoch=%s step=%s (skip %s batches)",
            as_rel(path),
            self.epoch,
            self.global_step,
            self._skip_batches,
        )

    def save(self, name: str, metrics: Optional[Dict[str, float]] = None) -> None:
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            return
        model = self.accelerator.unwrap_model(self.model)
        payload = build_payload(
            stage=self.stage,
            model=model,
            epoch=self.epoch,
            step=self.global_step,
            cfg=self.cfg,
            optimizer=getattr(self.optimizer, "optimizer", self.optimizer),
            scheduler=getattr(self.scheduler, "scheduler", self.scheduler),
            artifacts=self.artifacts,
            metrics={**(metrics or {}), "best": self.best_metric},
        )
        path = self.out_dir / name
        save_checkpoint(path, payload)
        self.logger.info("saved %s", as_rel(path))
        self._prune()

    def _prune(self) -> None:
        steps = sorted(self.out_dir.glob("step_*.pt"))
        for path in steps[: max(0, len(steps) - self.keep_last)]:
            path.unlink(missing_ok=True)

    # -- loops -------------------------------------------------------------
    def run(self) -> Dict[str, float]:
        self.setup()
        self.logger.info(
            "start training: epochs=%s max_steps=%s steps_per_epoch=%s",
            self.epochs,
            self.max_steps,
            self._steps_per_epoch(),
        )
        final_metrics: Dict[str, float] = {}
        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch
            if hasattr(self, "train_sampler") and hasattr(self.train_sampler, "set_epoch"):
                self.train_sampler.set_epoch(epoch)
            if hasattr(self.train_loader, "set_epoch"):
                self.train_loader.set_epoch(epoch)

            train_metrics = self.train_epoch()
            self.logger.info(
                "epoch %s done | %s | %s",
                epoch,
                " ".join(f"{k}={v:.4f}" for k, v in train_metrics.items()),
                format_seconds(time.time() - self._start_time),
            )

            if self.val_loader is not None:
                val_metrics = self.evaluation()
                final_metrics = val_metrics
                score = self.primary_metric(val_metrics)
                if self.accelerator.is_main_process and score < self.best_metric:
                    self.best_metric = score
                    self.save("best.pt", {**val_metrics, "best": score})
                self.logger.info(
                    "epoch %s eval | %s",
                    epoch,
                    " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()),
                )

            self.save("last.pt", final_metrics)
            if self.max_steps and self.global_step >= self.max_steps:
                break

        self.on_train_end()
        self.accelerator.wait_for_everyone()
        if self.dry_run:
            self.logger.info("dry run finished")
        return final_metrics

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        meters: Dict[str, AverageMeter] = {}
        loader = self.train_loader
        iterator = loader
        if self._skip_batches:
            iterator = itertools.islice(loader, self._skip_batches)
            self._skip_batches = 0

        for batch in iterator:
            batch = self.to_device(batch)
            with self.accelerator.accumulate(self.model):
                loss, metrics = self.training_step(batch)
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        float(get(self.cfg, "train.grad_clip", 1.0)),
                    )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            for key, value in metrics.items():
                meters.setdefault(key, AverageMeter()).update(value)
            meters.setdefault("loss", AverageMeter()).update(float(loss.detach()))

            if self.accelerator.sync_gradients:
                self.global_step += 1
                self._maybe_log(meters)
                if self.global_step % self.save_every == 0:
                    self.save(f"step_{self.global_step:08d}.pt")
                if self.val_loader is not None and self.global_step % self.eval_every == 0:
                    mid = self.evaluation()
                    if self.accelerator.is_main_process:
                        self.writer.write(
                            {"stage": self.stage, "epoch": self.epoch, "step": self.global_step,
                             **{f"eval_{k}": v for k, v in mid.items()}}
                        )
                        self.logger.info(
                            "step %s eval | %s",
                            self.global_step,
                            " ".join(f"{k}={v:.4f}" for k, v in mid.items()),
                        )
                if self.max_steps and self.global_step >= self.max_steps:
                    break

        return {k: m.avg for k, m in meters.items()}

    def _maybe_log(self, meters: Dict[str, AverageMeter]) -> None:
        if self.global_step % self.log_every != 0 or not self.accelerator.is_main_process:
            return
        record = {
            "stage": self.stage,
            "epoch": self.epoch,
            "step": self.global_step,
            "lr": self.scheduler.get_last_lr()[0] if self.scheduler else 0.0,
            **{k: round(m.avg, 6) for k, m in meters.items()},
        }
        self.writer.write(record)
        self.logger.info(
            "step %s | %s | lr=%.2e",
            self.global_step,
            " ".join(f"{k}={v:.4f}" for k, v in record.items() if isinstance(v, float)),
            record["lr"],
        )

    # -- helpers -----------------------------------------------------------
    def to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        device = self.accelerator.device
        out = {}
        for key, value in batch.items():
            out[key] = value.to(device) if torch.is_tensor(value) else value
        return out

    def reduce_metrics(self, sums: Dict[str, float], counts: Dict[str, float]) -> Dict[str, float]:
        keys = sorted(sums)
        packed = torch.tensor(
            [sums[k] for k in keys] + [counts[k] for k in keys],
            dtype=torch.float64,
            device=self.accelerator.device,
        )
        packed = self.accelerator.reduce(packed, reduction="sum")
        n = len(keys)
        return {
            k: float(packed[i] / packed[n + i].clamp_min(1e-6).item()) for i, k in enumerate(keys)
        }
