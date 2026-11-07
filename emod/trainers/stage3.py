"""Stage 3: T2E distillation from the frozen stage-2 encoder."""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from emod.config import as_rel, get, resolve_path
from emod.data.collate import collate_batch
from emod.data.dataset import EmotionSpeechDataset, manifest_path
from emod.models.factory import build_frontend, build_stage2_bundle, build_stage3_model
from emod.trainers.base import BaseTrainer
from emod.utils.audio import lengths_to_mask
from emod.utils.checkpoint import load_checkpoint


class Stage3Model(nn.Module):
    def __init__(self, frontend: nn.Module, teacher: nn.Module, t2e: nn.Module) -> None:
        super().__init__()
        self.frontend = frontend
        self.teacher = teacher
        self.t2e = t2e

    def train(self, mode: bool = True) -> "Stage3Model":
        super().train(mode)
        self.teacher.eval()
        return self

    @torch.no_grad()
    def teacher_frames(self, wav: torch.Tensor, wav_lengths: torch.Tensor):
        feats, lengths = self.frontend(wav, wav_lengths)
        mask = lengths_to_mask(lengths, feats.shape[1])
        v_emo, _ = self.teacher.emotion_encoder(feats, mask)
        return v_emo, mask


def phone_runs(labels: torch.Tensor, ignore_index: int) -> Tuple[List[int], List[int]]:
    """Frame-level phone ids -> (unique phones, run lengths)."""
    ids = labels[labels != ignore_index]
    if ids.numel() == 0:
        return [0], [1]
    change = torch.ones_like(ids, dtype=torch.bool)
    change[1:] = ids[1:] != ids[:-1]
    starts = change.nonzero(as_tuple=False).squeeze(1)
    ends = torch.cat([starts[1:], torch.tensor([ids.numel()], device=ids.device)])
    return ids[starts].tolist(), (ends - starts).tolist()


def pad_phones(
    sequences: List[Tuple[List[int], List[int]]], pad_id: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(p) for p, _ in sequences)
    phones = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    durations = torch.zeros(len(sequences), max_len, dtype=torch.long)
    lengths = torch.zeros(len(sequences), dtype=torch.long)
    for i, (ids, durs) in enumerate(sequences):
        phones[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        durations[i, : len(durs)] = torch.tensor(durs, dtype=torch.long)
        lengths[i] = len(ids)
    return phones, durations, lengths


class Stage3Trainer(BaseTrainer):
    stage = 3

    def build_model(self) -> nn.Module:
        stage2_path = resolve_path(get(self.cfg, "paths.stage2_checkpoint"))
        payload = load_checkpoint(stage2_path, expected_stage=2)
        artifacts = payload["artifacts"]
        if "directions" not in artifacts:
            raise RuntimeError(
                f"{as_rel(stage2_path)} has no calibrated directions; run stage 2 to completion"
            )

        teacher, _, artifacts = build_stage2_bundle(self.cfg, checkpoint=payload, device="cpu")
        for param in teacher.parameters():
            param.requires_grad = False
        teacher.eval()

        self.num_phones = int(artifacts["num_phones"])
        self.emotions = list(artifacts["emotions"])
        self.emotion_to_id = {e: i for i, e in enumerate(self.emotions)}
        self.direction_table = artifacts["directions"]["table"].float()
        self.scale_table = artifacts["directions"]["scale"].float()

        frontend = build_frontend(self.cfg)
        t2e = build_stage3_model(self.cfg, artifacts)
        self.stage2_path = stage2_path

        self.artifacts.update(
            {
                "stage2_checkpoint": as_rel(stage2_path),
                "num_phones": self.num_phones,
                "emotions": self.emotions,
                "directions": artifacts["directions"],
                "sample_rate": int(artifacts.get("sample_rate", 16000)),
                "hop_length": int(artifacts.get("hop_length", 320)),
            }
        )
        return Stage3Model(frontend, teacher, t2e)

    def build_loaders(self):
        train_dataset = EmotionSpeechDataset(
            self.cfg, manifest_path(self.cfg, "train"), training=True, require_phone_labels=True
        )
        collate = partial(
            collate_batch, ignore_index=int(get(self.cfg, "data.ignore_index", -100))
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=int(self.plan["per_gpu_batch_size"]),
            shuffle=True,
            drop_last=True,
            num_workers=int(get(self.cfg, "runtime.num_workers", 4)),
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate,
        )
        val_dataset = EmotionSpeechDataset(
            self.cfg,
            manifest_path(self.cfg, "val"),
            training=False,
            require_phone_labels=True,
            allow_unknown_speakers=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=int(self.plan["per_gpu_batch_size"]),
            shuffle=False,
            num_workers=int(get(self.cfg, "runtime.num_workers", 4)),
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate,
        )
        return self.train_loader, self.val_loader

    def _prepare_batch(self, batch: Dict[str, Any]):
        ignore = int(get(self.cfg, "data.ignore_index", -100))
        labels = batch["phone_ids"]
        lengths = batch["mel_lengths"]
        sequences = [
            phone_runs(labels[i, : int(lengths[i])], ignore) for i in range(labels.shape[0])
        ]
        phones, durations, phone_lengths = pad_phones(sequences)
        return phones, durations, phone_lengths

    def training_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        model: Stage3Model = self.model
        phones, durations, phone_lengths = self._prepare_batch(batch)
        phones = phones.to(self.accelerator.device)
        durations = durations.to(self.accelerator.device)
        phone_lengths = phone_lengths.to(self.accelerator.device)

        table = self.direction_table.to(phones.device)
        scale = self.scale_table.to(phones.device)

        out = model.t2e(
            phones,
            phone_lengths,
            batch["emotion_id"],
            table,
            scale,
            durations=durations,
        )
        target, target_mask = model.teacher_frames(batch["wav"], batch["wav_lengths"])
        target = F.interpolate(
            target.transpose(1, 2),
            size=out["v_emo"].shape[1],
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

        frame_mask = lengths_to_mask(out["frame_lengths"], out["v_emo"].shape[1])
        diff = (out["v_emo"] - target).abs().mean(dim=-1)
        weights = frame_mask.to(diff.dtype)
        l1 = (diff * weights).sum() / weights.sum().clamp_min(1.0)
        cos = 1.0 - F.cosine_similarity(out["v_emo"], target, dim=-1, eps=1e-8)
        cos = (cos * weights).sum() / weights.sum().clamp_min(1.0)

        logits = model.teacher.emotion_head(out["v_emo_utt"])
        ce = F.cross_entropy(logits, batch["emotion_id"])

        direction = table[batch["emotion_id"]]
        scale_e = scale[batch["emotion_id"]]
        alpha_teacher = (target * direction.unsqueeze(1)).sum(-1) / scale_e.unsqueeze(1).clamp_min(1e-6)
        # The student can only emit alpha in [0, alpha_max]; clamp the teacher to
        # the same range so an early, badly calibrated direction cannot dominate.
        alpha_teacher = alpha_teacher.clamp(0.0, model.t2e.alpha_max * 1.5)
        alpha_student = out["alpha"]
        alpha_match = (alpha_student.mean(dim=1) - alpha_teacher.mean(dim=1)).abs().mean()
        alpha_match = alpha_match + (
            alpha_student.std(dim=1) - alpha_teacher.std(dim=1)
        ).abs().mean()

        residual_norm = out["residual"].norm(dim=-1).mean() / (
            float(get(self.cfg, "model.t2e.residual_ratio", 0.2)) * scale_e.mean().clamp_min(1e-6)
        )

        log_dur_pred = out["log_durations"]
        log_dur_gt = torch.log1p(durations.to(log_dur_pred.dtype))
        phone_mask = lengths_to_mask(phone_lengths, phones.shape[1]).to(log_dur_pred.dtype)
        dur_loss = ((log_dur_pred - log_dur_gt).abs() * phone_mask).sum() / phone_mask.sum().clamp_min(1.0)

        total = (
            float(get(self.cfg, "loss.distill_l1_weight", 1.0)) * l1
            + float(get(self.cfg, "loss.distill_cos_weight", 1.0)) * cos
            + float(get(self.cfg, "loss.emo_ce_weight", 0.5)) * ce
            + float(get(self.cfg, "loss.alpha_match_weight", 0.1)) * alpha_match
            + float(get(self.cfg, "loss.residual_weight", 0.05)) * residual_norm
            + float(get(self.cfg, "loss.duration_weight", 0.1)) * dur_loss
        )

        with torch.no_grad():
            accuracy = (logits.argmax(dim=-1) == batch["emotion_id"]).float().mean()
        return total, {
            "l1": float(l1.detach()),
            "cos": float(cos.detach()),
            "ce": float(ce.detach()),
            "alpha": float(alpha_match.detach()),
            "dur": float(dur_loss.detach()),
            "emo_acc": float(accuracy),
            "loss": float(total.detach()),
        }

    @torch.no_grad()
    def evaluation(self) -> Dict[str, float]:
        model: Stage3Model = self.accelerator.unwrap_model(self.model)
        model.eval()
        sums = {"l1": 0.0, "cos": 0.0, "emo_acc": 0.0}
        counts = {"l1": 0.0, "cos": 0.0, "emo_acc": 0.0}
        table = self.direction_table.to(self.accelerator.device)
        scale = self.scale_table.to(self.accelerator.device)

        for batch in self.val_loader:
            batch = self.to_device(batch)
            phones, durations, phone_lengths = self._prepare_batch(batch)
            phones = phones.to(self.accelerator.device)
            durations = durations.to(self.accelerator.device)
            phone_lengths = phone_lengths.to(self.accelerator.device)
            out = model.t2e(
                phones, phone_lengths, batch["emotion_id"], table, scale, durations=durations
            )
            target, _ = model.teacher_frames(batch["wav"], batch["wav_lengths"])
            target = F.interpolate(
                target.transpose(1, 2), size=out["v_emo"].shape[1], mode="linear", align_corners=False
            ).transpose(1, 2)
            mask = lengths_to_mask(out["frame_lengths"], out["v_emo"].shape[1])
            weights = mask.to(out["v_emo"].dtype)
            n = float(mask.sum())

            sums["l1"] += float(((out["v_emo"] - target).abs().mean(-1) * weights).sum())
            sums["cos"] += float(
                ((1 - F.cosine_similarity(out["v_emo"], target, dim=-1)) * weights).sum()
            )
            counts["l1"] += n
            counts["cos"] += n
            acc = (model.teacher.emotion_head(out["v_emo_utt"]).argmax(-1) == batch["emotion_id"])
            sums["emo_acc"] += float(acc.float().sum())
            counts["emo_acc"] += float(acc.numel())

        metrics = self.reduce_metrics(sums, counts)
        metrics["val_loss"] = metrics.get("l1", 0.0)
        model.train()
        return metrics

    def primary_metric(self, metrics: Dict[str, float]) -> float:
        return metrics.get("val_loss", float("inf"))

    def on_train_end(self) -> None:
        metrics = self.evaluation() if self.val_loader is not None else {}
        if not self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            return
        self.save("final.pt", metrics)
        self.accelerator.wait_for_everyone()
