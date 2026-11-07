"""DPPG: per-phoneme k-means discretization of PPG frames.

Token id = phone_id * max_clusters + cluster_id, so the vocabulary is a fixed
rectangle and stays stable across datasets.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch


def _kmeans(x: torch.Tensor, k: int, iters: int = 25, seed: int = 0) -> Tuple[torch.Tensor, float]:
    """Lloyd's algorithm with k-means++ init on [N, D] CPU tensors."""
    n = x.shape[0]
    k = max(1, min(k, n))
    gen = torch.Generator().manual_seed(seed)

    centers = [x[torch.randint(n, (1,), generator=gen).item()]]
    d2 = ((x - centers[0]) ** 2).sum(dim=1)
    for _ in range(1, k):
        if float(d2.sum()) <= 0:
            break
        idx = torch.multinomial(d2, 1, generator=gen).item()
        centers.append(x[idx])
        d2 = torch.minimum(d2, ((x - x[idx]) ** 2).sum(dim=1))
    centroids = torch.stack(centers)

    assign = torch.zeros(n, dtype=torch.long)
    for _ in range(iters):
        dist = torch.cdist(x, centroids)
        assign = dist.argmin(dim=1)
        for j in range(centroids.shape[0]):
            member = assign == j
            if member.any():
                centroids[j] = x[member].mean(dim=0)

    inertia = float(((x - centroids[assign]) ** 2).sum().item())
    return centroids, inertia


def _kneedle(ks: List[int], inertias: List[float]) -> int:
    """Pick the elbow by maximum distance to the endpoint chord."""
    if len(ks) <= 2:
        return ks[0]
    k_arr = torch.tensor(ks, dtype=torch.float32)
    i_arr = torch.tensor(inertias, dtype=torch.float32)
    k_norm = (k_arr - k_arr.min()) / (k_arr.max() - k_arr.min()).clamp_min(1e-6)
    i_norm = (i_arr - i_arr.min()) / (i_arr.max() - i_arr.min()).clamp_min(1e-6)
    line = k_norm  # chord from (0,0) to (1,1)
    distance = (i_norm - line).abs()
    return int(ks[int(distance.argmax().item())])


class DPPGTokenizer:
    def __init__(
        self,
        num_phones: int,
        max_clusters: int = 8,
        min_clusters: int = 2,
        min_frames: int = 32,
        seed: int = 0,
    ) -> None:
        self.num_phones = int(num_phones)
        self.max_clusters = int(max_clusters)
        self.min_clusters = int(min_clusters)
        self.min_frames = int(min_frames)
        self.seed = int(seed)
        self.centroids: List[Optional[torch.Tensor]] = [None] * self.num_phones

    @property
    def vocab_size(self) -> int:
        return self.num_phones * self.max_clusters

    def fit(
        self,
        ppg_probs: torch.Tensor,
        phone_ids: torch.Tensor,
        max_frames: int = 200_000,
        samples_per_phone: int = 2000,
    ) -> Dict[str, Any]:
        """Fit one k-means per phone class. Inputs are flattened frame tensors."""
        probs = ppg_probs.detach().float().cpu()
        phones = phone_ids.detach().long().cpu().reshape(-1)

        if probs.numel() == 0:
            raise ValueError("cannot fit DPPG on empty tensors")
        if max_frames and probs.shape[0] > max_frames:
            keep = torch.randperm(probs.shape[0])[:max_frames]
            probs, phones = probs[keep], phones[keep]

        report = {}
        for phone in range(self.num_phones):
            frames = probs[phones == phone]
            if frames.shape[0] < self.min_frames:
                continue
            if frames.shape[0] > samples_per_phone:
                keep = torch.randperm(frames.shape[0])[:samples_per_phone]
                frames = frames[keep]

            ks, inertias = [], []
            for k in range(self.min_clusters, self.max_clusters + 1):
                if k >= frames.shape[0]:
                    break
                _, inertia = _kmeans(frames, k, seed=self.seed + phone * 17 + k)
                ks.append(k)
                inertias.append(inertia)
            if not ks:
                continue
            best_k = _kneedle(ks, inertias) if len(ks) > 1 else ks[0]
            centroids, _ = _kmeans(frames, best_k, seed=self.seed + phone * 31)
            self.centroids[phone] = centroids
            report[str(phone)] = int(best_k)
        return report

    def encode(self, ppg_probs: torch.Tensor, phone_ids: torch.Tensor) -> torch.Tensor:
        """[B, T, P] + [B, T] -> DPPG ids [B, T]."""
        ids = phone_ids.clamp(0, self.num_phones - 1) * self.max_clusters
        device = ids.device
        for phone, centroids in enumerate(self.centroids):
            if centroids is None:
                continue
            member = phone_ids == phone
            if not bool(member.any()):
                continue
            frames = ppg_probs[member]
            dist = torch.cdist(frames, centroids.to(frames.device, frames.dtype))
            ids[member] = ids[member] + dist.argmin(dim=1).to(device)
        return ids

    # -- serialization ------------------------------------------------------
    def state(self) -> Dict[str, Any]:
        size = self.max_clusters
        num_clusters = []
        flat = []
        for centroids in self.centroids:
            if centroids is None:
                num_clusters.append(0)
                flat.append(torch.zeros(size, self.num_phones))
            else:
                num_clusters.append(centroids.shape[0])
                padded = torch.zeros(size, self.num_phones)
                padded[: centroids.shape[0]] = centroids
                flat.append(padded)
        return {
            "num_phones": self.num_phones,
            "max_clusters": self.max_clusters,
            "num_clusters": num_clusters,
            "centroids": flat,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "DPPGTokenizer":
        tokenizer = cls(
            num_phones=int(state["num_phones"]),
            max_clusters=int(state["max_clusters"]),
        )
        for i, (count, padded) in enumerate(zip(state["num_clusters"], state["centroids"])):
            if int(count) > 0:
                tokenizer.centroids[i] = padded[: int(count)].float()
        return tokenizer
