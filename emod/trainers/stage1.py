"""Stage 1: frame-level PPG head, then fit the DPPG tokenizer."""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from emod.config import get, load_phone_set
from emod.data.collate import collate_batch
from emod.data.dataset import EmotionSpeechDataset, manifest_path
from emod.losses import frame_ce
from emod.models.factory import build_frontend, build_ppg_head
from emod.models.dppg import DPPGTokenizer
from emod.trainers.base import BaseTrainer
from emod.utils.audio import lengths_to_mask


class Stage1Model(nn.Module):
    def __init__(self, frontend: nn.Module, ppg: nn.Module) -> None:
        super().__init__()
        self.frontend = frontend
        self.ppg = ppg

    def forward(self, wav: torch.Tensor, wav_lengths: torch.Tensor):
        feats, lengths = self.frontend(wav, wav_lengths)
        mask = lengths_to_mask(lengths, feats.shape[1])
        return self.ppg(feats, mask), feats, lengths, mask


class Stage1Trainer(BaseTrainer):
    stage = 1

    def build_model(self) -> nn.Module:
        self.phones = load_phone_set(get(self.cfg, "data.phone_set"))
        self.phone_to_id = {phone: i for i, phone in enumerate(self.phones)}
        self.num_phones = len(self.phones)

        frontend = build_frontend(self.cfg)
        ppg = build_ppg_head(self.cfg, int(frontend.output_dim), self.num_phones)
        self.artifacts.update(
            {
                "phones": self.phones,
                "num_phones": self.num_phones,
                "content_vocab": self.num_phones
                * int(get(self.cfg, "model.dppg.max_clusters", 8)),
            }
        )
        return Stage1Model(frontend, ppg)

    def _loader(self, split: str, training: bool) -> DataLoader:
        dataset = EmotionSpeechDataset(
            self.cfg,
            manifest_path(self.cfg, split),
            training=training,
            require_phone_labels=True,
            allow_unknown_speakers=not training,
        )
        if training:
            self.train_dataset = dataset
            self.train_sampler = None
        return DataLoader(
            dataset,
            batch_size=int(self.plan["per_gpu_batch_size"]),
            shuffle=training,
            drop_last=training,
            num_workers=int(get(self.cfg, "runtime.num_workers", 4)),
            pin_memory=torch.cuda.is_available(),
            collate_fn=partial(
                collate_batch, ignore_index=int(get(self.cfg, "data.ignore_index", -100))
            ),
        )

    def build_loaders(self):
        self.train_loader = self._loader("train", training=True)
        try:
            self.val_loader = self._loader("val", training=False)
        except FileNotFoundError:
            self.val_loader = None
        return self.train_loader, self.val_loader

    def training_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits, _, _, mask = self.model(batch["wav"], batch["wav_lengths"])
        labels = batch["phone_ids"][:, : logits.shape[1]]
        valid = mask & (labels != int(get(self.cfg, "data.ignore_index", -100)))
        loss = frame_ce(logits, labels.clamp_min(0), valid)
        with torch.no_grad():
            accuracy = (
                (logits.argmax(dim=-1) == labels)[valid].float().mean()
                if bool(valid.any())
                else logits.new_zeros(())
            )
        return loss, {"ppg_acc": float(accuracy), "phone_ce": float(loss.detach())}

    @torch.no_grad()
    def evaluation(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        model = self.accelerator.unwrap_model(self.model)
        model.eval()
        sums = {"phone_ce": 0.0, "ppg_acc": 0.0}
        counts = {"phone_ce": 0.0, "ppg_acc": 0.0}
        for batch in self.val_loader:
            batch = self.to_device(batch)
            logits, _, _, mask = model(batch["wav"], batch["wav_lengths"])
            labels = batch["phone_ids"][:, : logits.shape[1]]
            valid = mask & (labels != int(get(self.cfg, "data.ignore_index", -100)))
            loss = frame_ce(logits, labels.clamp_min(0), valid)
            acc = (logits.argmax(dim=-1) == labels)[valid].float().mean()
            n = float(valid.sum())
            sums["phone_ce"] += float(loss) * n
            sums["ppg_acc"] += float(acc) * n
            counts["phone_ce"] += n
            counts["ppg_acc"] += n
        metrics = self.reduce_metrics(sums, counts)
        metrics["val_loss"] = metrics.get("phone_ce", 0.0)
        model.train()
        return metrics

    def primary_metric(self, metrics: Dict[str, float]) -> float:
        return -metrics.get("ppg_acc", 0.0)

    def on_train_end(self) -> None:
        """Fit per-phoneme k-means once, on rank 0, and store it in the checkpoint."""
        metrics = self.evaluation() if self.val_loader is not None else {}
        if not self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            return

        model = self.accelerator.unwrap_model(self.model)
        model.eval()
        max_frames = int(get(self.cfg, "model.dppg.max_frames", 200000))
        loader = DataLoader(
            self.train_dataset,
            batch_size=max(1, int(self.plan["per_gpu_batch_size"])),
            shuffle=False,
            num_workers=int(get(self.cfg, "runtime.num_workers", 4)),
            collate_fn=partial(
                collate_batch, ignore_index=int(get(self.cfg, "data.ignore_index", -100))
            ),
        )

        probs_all, phones_all, collected = [], [], 0
        for batch in loader:
            batch = self.to_device(batch)
            logits, _, _, mask = model(batch["wav"], batch["wav_lengths"])
            probs = torch.softmax(logits, dim=-1)[mask]
            phones = batch["phone_ids"][:, : logits.shape[1]][mask]
            valid = phones != int(get(self.cfg, "data.ignore_index", -100))
            probs_all.append(probs[valid].cpu())
            phones_all.append(phones[valid].cpu())
            collected += int(valid.sum())
            if collected >= max_frames:
                break

        tokenizer = DPPGTokenizer(
            num_phones=self.num_phones,
            max_clusters=int(get(self.cfg, "model.dppg.max_clusters", 8)),
            min_clusters=int(get(self.cfg, "model.dppg.min_clusters", 2)),
        )
        report = tokenizer.fit(
            torch.cat(probs_all),
            torch.cat(phones_all),
            max_frames=max_frames,
            samples_per_phone=int(get(self.cfg, "model.dppg.samples_per_phone", 2000)),
        )
        self.artifacts["dppg"] = tokenizer.state()
        self.artifacts["dppg_clusters"] = report
        self.save("final.pt", metrics)
        self.accelerator.wait_for_everyone()
