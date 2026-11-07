from pathlib import Path

import pytest

from emod.config import get, apply_overrides, load_config, load_phone_set, PROJECT_ROOT


def test_stage_config_inherits_and_merges():
    cfg = load_config("configs/stage2_vae.yaml")
    assert cfg["stage"] == 2
    assert cfg["model"]["latent_dim"] == 64
    assert cfg["data"]["n_mels"] == 80
    # GPU profile from the stage config is applied last.
    assert cfg["sampler"]["num_speakers"] == 4


def test_gpu_override_wins():
    cfg = load_config("configs/stage2_vae.yaml", extras=["configs/gpu/h200_8gpu.yaml"])
    assert cfg["sampler"]["num_speakers"] == 8
    assert cfg["runtime"]["mixed_precision"] == "bf16"


def test_cli_overrides_are_typed():
    cfg = apply_overrides({"train": {"lr": 1e-4}}, ["train.lr=2e-5", "train.epochs=3"])
    assert cfg["train"]["lr"] == pytest.approx(2e-5)
    assert cfg["train"]["epochs"] == 3


def test_phone_set_is_unique():
    phones = load_phone_set("configs/phones/en_zh_phones.yaml")
    assert len(phones) == len(set(phones))
    assert "SIL" in phones


def test_all_config_paths_are_relative():
    for path in (PROJECT_ROOT / "configs").rglob("*.yaml"):
        cfg = load_config(path.relative_to(PROJECT_ROOT))
        for value in _walk(cfg):
            if isinstance(value, str):
                assert not Path(value).is_absolute(), f"absolute path in {path}: {value}"
                assert ":" not in value or value.startswith("http"), f"drive-like path: {value}"


def _walk(node):
    if isinstance(node, dict):
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)
    else:
        yield node
