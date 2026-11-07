"""Stage 2: additive VAE in the joint timbre-emotion space."""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from emod.calibration import estimate_directions, estimate_neutral_vectors
from emod.config import as_rel, get, resolve_path
from emod.data.collate import collate_batch
from emod.data.dataset import EmotionSpeechDataset, manifest_path
from emod.data.manifest import read_manifest
from emod.data.sampler import StratifiedEmotionSampler
from emod.losses import (
    cross_speaker_delta_loss,
    direction_compactness,
    frame_ce,
    kl_free_bits,
    neutral_reference,
    orthogonality_loss,
    reconstruction_loss,
    scheduled_weight,
    swap_probe_loss,
)
from emod.models.blocks import grad_reverse, masked_mean
from emod.models.dppg import DPPGTokenizer
from emod.models.factory import build_additive_vae, build_frontend, build_ppg_head
from emod.trainers.base import BaseTrainer
from emod.utils.audio import lengths_to_mask
from emod.utils.checkpoint import load_checkpoint
from emod.utils.geometry import l2_normalize


class Stage2Model(nn.Module):
    def __init__(self, frontend: nn.Module, ppg: nn.Module, vae: nn.Module) -> None:
        super().__init__()
        self.frontend = frontend
        self.ppg = ppg
        self.vae = vae

    def train(self, mode: bool = True) -> "Stage2Model":
        super().train(mode)
        self.ppg.eval()  # frozen content branch
        return self

    def encode_inputs(self, wav, wav_lengths, with_content: bool = True):
        feats, lengths = self.frontend(wav, wav_lengths)
        mask = lengths_to_mask(lengths, feats.shape[1])
        if not with_content:
            return feats, mask, None
        logits = self.ppg(feats, mask)
        return feats, mask, torch.softmax(logits, dim=-1)


class Stage2Trainer(BaseTrainer):
    stage = 2

    def build_model(self) -> nn.Module:
        stage1_path = resolve_path(get(self.cfg, "paths.stage1_checkpoint"))
        payload = load_checkpoint(stage1_path, expected_stage=1)
        artifacts = payload["artifacts"]

        self.num_phones = int(artifacts["num_phones"])
        self.phones = list(artifacts["phones"])
        self.tokenizer = DPPGTokenizer.from_state(artifacts["dppg"])
        self.content_vocab = self.num_phones * int(get(self.cfg, "model.dppg.max_clusters", 8))

        # The front-end must match stage 1, otherwise content ids are meaningless.
        src = payload["config"]["model"]["frontend"]
        dst = get(self.cfg, "model.frontend")
        for key in ("type", "name", "output_dim"):
            if str(src.get(key)) != str(dst.get(key)):
                raise RuntimeError(
                    f"frontend mismatch between stage1 ({src.get(key)}) and stage2 ({dst.get(key)})"
                )

        rows = read_manifest(manifest_path(self.cfg, "train"))
        self.speakers = sorted({r["speaker"] for r in rows})
        self.emotions = list(get(self.cfg, "data.emotions"))
        self.neutral_id = self.emotions.index("neutral")

        frontend = build_frontend(self.cfg)
        ppg = build_ppg_head(self.cfg, int(frontend.output_dim), self.num_phones)
        ppg_state = {
            k[len("ppg.") :]: v for k, v in payload["model"].items() if k.startswith("ppg.")
        }
        ppg.load_state_dict(ppg_state, strict=True)
        for param in ppg.parameters():
            param.requires_grad = False

        vae = build_additive_vae(
            self.cfg,
            num_speakers=len(self.speakers),
            num_emotions=len(self.emotions),
            num_phones=self.num_phones,
            content_vocab=self.content_vocab,
        )

        self.direction_table = torch.zeros(len(self.emotions), int(get(self.cfg, "model.latent_dim")))
        self._dir_running = torch.zeros_like(self.direction_table)
        self._dir_seen = torch.zeros(len(self.emotions), dtype=torch.bool)

        self.artifacts.update(
            {
                "speakers": self.speakers,
                "emotions": self.emotions,
                "num_phones": self.num_phones,
                "content_vocab": self.content_vocab,
                "dppg": self.tokenizer.state(),
                "stage1_checkpoint": as_rel(stage1_path),
                "sample_rate": int(get(self.cfg, "data.sample_rate", 16000)),
                "hop_length": int(get(self.cfg, "data.hop_length", 320)),
                "n_mels": int(get(self.cfg, "data.n_mels", 80)),
            }
        )
        return Stage2Model(frontend, ppg, vae)

    # -- data --------------------------------------------------------------
    def build_loaders(self):
        train_dataset = EmotionSpeechDataset(
            self.cfg, manifest_path(self.cfg, "train"), training=True, speaker_list=self.speakers
        )
        self.train_dataset = train_dataset
        self.train_sampler = StratifiedEmotionSampler(
            rows=train_dataset.rows,
            emotions=self.emotions,
            num_speakers=int(get(self.cfg, "sampler.num_speakers", 4)),
            per_cell=int(get(self.cfg, "sampler.per_cell", 2)),
            steps_per_epoch=int(get(self.cfg, "sampler.steps_per_epoch", 400)),
            seed=int(get(self.cfg, "project.seed", 1234)) + self.accelerator.process_index,
        )
        collate = partial(
            collate_batch, ignore_index=int(get(self.cfg, "data.ignore_index", -100))
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_sampler=self.train_sampler,
            num_workers=int(get(self.cfg, "runtime.num_workers", 4)),
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate,
        )
        val_dataset = EmotionSpeechDataset(
            self.cfg,
            manifest_path(self.cfg, "val"),
            training=False,
            speaker_list=self.speakers,
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

    # -- training ----------------------------------------------------------
    def _align(self, feats, mask, mel):
        frames = min(feats.shape[1], mel.shape[1])
        return feats[:, :frames], mask[:, :frames], mel[:, :frames], frames

    def training_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        model: Stage2Model = self.model
        feats, mask, probs = model.encode_inputs(batch["wav"], batch["wav_lengths"])
        feats, mask, mel, frames = self._align(feats, mask, batch["mel"])

        phone = probs[:, :frames].argmax(dim=-1)
        with torch.no_grad():
            content = self.tokenizer.encode(probs[:, :frames], phone).clamp(
                0, self.content_vocab - 1
            )

        out = model.vae(feats, mask, content)
        speaker_id = batch["speaker_id"]
        emotion_id = batch["emotion_id"]

        rec = reconstruction_loss(out["mel_hat"], mel, mask)
        beta = float(get(self.cfg, "loss.kl_weight")) * min(
            1.0, self.global_step / max(1, int(get(self.cfg, "loss.kl_warmup_steps", 10000)))
        )
        kl = kl_free_bits(
            out["v_emo_mu"],
            out["v_emo_logvar"],
            float(get(self.cfg, "loss.kl_free_bits", 0.05)),
        )

        total = rec["total"] + beta * kl

        attr_start = int(get(self.cfg, "loss.attr_start_step", 0))
        metrics: Dict[str, float] = {
            "rec": float(rec["rec"].detach()),
            "kl": float(kl.detach()),
        }

        if self.global_step >= attr_start:
            v_spk = out["v_spk"]
            v_emo = out["v_emo"]
            v_emo_utt = out["v_emo_utt"]

            spk_loss = model.vae.speaker_head(v_spk, speaker_id)
            emo_utt_logits = model.vae.emotion_head(v_emo_utt)
            emo_loss = F.cross_entropy(emo_utt_logits, emotion_id)
            frame_logits = model.vae.emotion_head(v_emo)
            emo_frame = frame_ce(
                frame_logits,
                emotion_id.unsqueeze(1).expand(-1, frame_logits.shape[1]),
                mask,
            )

            v_neu = neutral_reference(v_emo, mask, speaker_id, emotion_id, self.neutral_id)
            delta = v_emo - v_neu
            cross = cross_speaker_delta_loss(
                delta, mask, speaker_id, emotion_id, batch["text_id"], self.neutral_id
            )
            orth = orthogonality_loss(v_spk, v_emo_utt)

            if self.direction_table.device != v_emo.device:
                self.direction_table = self.direction_table.to(v_emo.device)
                self._dir_running = self._dir_running.to(v_emo.device)
                self._dir_seen = self._dir_seen.to(v_emo.device)
            dir_loss = torch.zeros((), device=v_emo.device)
            if bool(self._dir_seen.any()):
                dir_loss = direction_compactness(
                    delta, mask, self.direction_table, emotion_id, self.neutral_id
                )

            total = (
                total
                + float(get(self.cfg, "loss.spk_weight")) * spk_loss
                + float(get(self.cfg, "loss.emo_weight")) * (emo_loss + emo_frame)
                + float(get(self.cfg, "loss.cross_weight")) * cross
                + float(get(self.cfg, "loss.orth_weight")) * orth
                + float(get(self.cfg, "loss.dir_weight")) * dir_loss
            )

            adv_w = scheduled_weight(
                float(get(self.cfg, "loss.adv_weight")),
                self.global_step,
                int(get(self.cfg, "loss.adv_start_step", 10000)),
                int(get(self.cfg, "loss.adv_ramp_steps", 5000)),
            )
            adv = torch.zeros((), device=total.device)
            if adv_w > 0:
                adv = F.cross_entropy(model.vae.adv_head(grad_reverse(v_emo_utt)), speaker_id)
                total = total + adv_w * adv

            swap = torch.zeros((), device=total.device)
            swap_w = float(get(self.cfg, "loss.swap_weight", 0.0))
            if swap_w > 0:
                swap = swap_probe_loss(
                    model.vae, v_spk, v_emo, content, mask, speaker_id, emotion_id
                )
                total = total + swap_w * swap

            with torch.no_grad():
                accuracy = (emo_utt_logits.argmax(-1) == emotion_id).float().mean()
            metrics.update(
                {
                    "spk": float(spk_loss.detach()),
                    "emo": float((emo_loss + emo_frame).detach()),
                    "cross": float(cross.detach()),
                    "dir": float(dir_loss.detach()),
                    "adv": float(adv.detach()),
                    "swap": float(swap.detach()),
                    "emo_acc": float(accuracy),
                }
            )
            self._update_directions(delta.detach(), mask, emotion_id)

        metrics["loss"] = float(total.detach())
        return total, metrics

    @torch.no_grad()
    def _update_directions(self, delta, mask, emotion_id) -> None:
        pooled = masked_mean(delta, mask)
        for emotion in torch.unique(emotion_id).tolist():
            if emotion == self.neutral_id:
                continue
            rows = emotion_id == emotion
            mean = pooled[rows].mean(dim=0)
            if not bool(self._dir_seen[emotion]):
                self._dir_running[emotion] = mean
                self._dir_seen[emotion] = True
            else:
                self._dir_running[emotion] = 0.99 * self._dir_running[emotion] + 0.01 * mean
            self.direction_table[emotion] = l2_normalize(
                self._dir_running[emotion], dim=-1
            )

    @torch.no_grad()
    def evaluation(self) -> Dict[str, float]:
        model: Stage2Model = self.accelerator.unwrap_model(self.model)
        model.eval()
        sums = {"rec": 0.0, "emo_acc": 0.0}
        counts = {"rec": 0.0, "emo_acc": 0.0}
        for batch in self.val_loader:
            batch = self.to_device(batch)
            feats, mask, probs = model.encode_inputs(
                batch["wav"], batch["wav_lengths"], with_content=True
            )
            feats, mask, mel, frames = self._align(feats, mask, batch["mel"])
            phone = probs[:, :frames].argmax(dim=-1)
            content = self.tokenizer.encode(probs[:, :frames], phone).clamp(
                0, self.content_vocab - 1
            )
            out = model.vae(feats, mask, content_ids=content, sample=False)
            rec = reconstruction_loss(out["mel_hat"], mel, mask)["rec"]
            acc = (
                out["v_emo_utt"].argmax(dim=-1) == batch["emotion_id"]
            ).float().mean()
            n = float(mask.sum())
            sums["rec"] += float(rec) * n
            counts["rec"] += n
            sums["emo_acc"] += float(acc) * batch["wav"].shape[0]
            counts["emo_acc"] += batch["wav"].shape[0]
        metrics = self.reduce_metrics(sums, counts)
        metrics["val_loss"] = metrics.get("rec", 0.0)
        model.train()
        return metrics

    def primary_metric(self, metrics: Dict[str, float]) -> float:
        return metrics.get("val_loss", float("inf"))

    # -- calibration -------------------------------------------------------
    def on_train_end(self) -> None:
        metrics = self.evaluation() if self.val_loader is not None else {}
        if not self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            return

        model: Stage2Model = self.accelerator.unwrap_model(self.model)
        model.eval()
        loader = DataLoader(
            self.train_dataset,
            batch_size=int(self.plan["per_gpu_batch_size"]),
            shuffle=False,
            num_workers=int(get(self.cfg, "runtime.num_workers", 4)),
            collate_fn=partial(
                collate_batch, ignore_index=int(get(self.cfg, "data.ignore_index", -100))
            ),
        )

        def encode_fn(batch):
            feats, mask, _ = model.encode_inputs(
                batch["wav"], batch["wav_lengths"], with_content=False
            )
            v_spk, v_emo, _, _ = model.vae.encode(feats, mask, sample=False)
            return v_emo, mask, batch["speaker_id"], batch["emotion_id"]

        neutral = estimate_neutral_vectors(
            encode_fn, loader, self.neutral_id, self.accelerator.device
        )
        directions = estimate_directions(
            encode_fn,
            loader,
            neutral,
            len(self.emotions),
            self.neutral_id,
            self.accelerator.device,
        )

        self.artifacts["directions"] = {
            "table": directions["table"],
            "scale": directions["scale"],
            "counts": directions["counts"],
        }
        self.save("final.pt", metrics)
        self.accelerator.wait_for_everyone()
