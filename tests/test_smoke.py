import pytest

torch = pytest.importorskip("torch")

from emod.models.dppg import DPPGTokenizer  # noqa: E402
from emod.models.factory import build_additive_vae, build_stage2_bundle  # noqa: E402
from emod.models.frontend import MockFrontend  # noqa: E402
from emod.models.ppg import PPGHead  # noqa: E402
from emod.models.t2e import T2EModel  # noqa: E402
from emod.utils.checkpoint import build_payload, load_checkpoint, save_checkpoint  # noqa: E402
from emod.utils.misc import resolve_batch_plan  # noqa: E402


def _cfg():
    return {
        "stage": 2,
        "model": {
            "frontend": {"type": "mock", "output_dim": 32, "adapter_dim": 0},
            "latent_dim": 8,
            "emotion_encoder": {"hidden": 32, "layers": 2, "heads": 4, "dropout": 0.0},
            "speaker_encoder": {"hidden": 32, "dropout": 0.0},
            "decoder": {"hidden": 32, "layers": 2, "heads": 4, "dropout": 0.0},
            "dppg": {"max_clusters": 4},
        },
        "data": {"n_mels": 80, "emotions": ["neutral", "angry", "happy", "sad", "surprise"]},
        "sampler": {"num_speakers": 4, "per_cell": 2},
        "train": {"effective_batch_size": 160, "per_gpu_batch_size": 8},
    }


def test_additive_vae_shapes_and_backward():
    cfg = _cfg()
    model = build_additive_vae(cfg, num_speakers=4, num_emotions=5, num_phones=6)
    feats = torch.randn(3, 16, 32)
    mask = torch.ones(3, 16, dtype=torch.bool)
    mask[2, 10:] = False
    content = torch.randint(0, 24, (3, 16))

    out = model(feats, mask, content)
    assert out["mel_hat"].shape == (3, 16, 80)
    assert torch.allclose(out["z"], out["v_spk"].unsqueeze(1) + out["v_emo"], atol=1e-5)

    out["mel_hat"].mean().backward()
    assert any(p.grad is not None for p in model.parameters())


def test_ppg_head_downstream_shape():
    frontend = MockFrontend(output_dim=32)
    ppg = PPGHead(32, num_phones=6, hidden=32, layers=1, heads=4)
    wav = torch.randn(2, 16000)
    feats, lengths = frontend(wav, torch.tensor([16000, 12000]))
    mask = torch.arange(feats.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    logits = ppg(feats, mask)
    assert logits.shape[0] == 2 and logits.shape[-1] == 6
    assert logits.shape[1] == feats.shape[1]


def test_t2e_output_lengths_follow_durations():
    model = T2EModel(num_phones=6, num_emotions=5, latent_dim=8, hidden=32, layers=2, heads=4)
    phones = torch.tensor([[2, 3], [4, 5]])
    lengths = torch.tensor([2, 2])
    durations = torch.tensor([[4, 6], [3, 3]])
    table = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
    scale = torch.ones(5)

    out = model(phones, lengths, torch.tensor([1, 2]), table, scale, durations=durations)
    assert out["v_emo"].shape == (2, 10, 8)
    assert out["frame_lengths"].tolist() == [10, 6]


def test_stage2_bundle_strips_prefix(tmp_path):
    cfg = _cfg()
    vae = build_additive_vae(cfg, num_speakers=4, num_emotions=5, num_phones=6)
    tokenizer = DPPGTokenizer(num_phones=6, max_clusters=4)
    artifacts = {
        "speakers": ["a", "b", "c", "d"],
        "emotions": cfg["data"]["emotions"],
        "num_phones": 6,
        "content_vocab": 24,
        "dppg": tokenizer.state(),
    }
    payload = {
        "format_version": 1,
        "stage": 2,
        "epoch": 0,
        "step": 0,
        "config": cfg,
        "model": {f"vae.{k}": v for k, v in vae.state_dict().items()},
        "artifacts": artifacts,
        "metrics": {},
    }
    restored, _, _ = build_stage2_bundle(None, payload)
    for key, value in vae.state_dict().items():
        assert torch.equal(restored.state_dict()[key], value)


def test_checkpoint_roundtrip(tmp_path):
    model = torch.nn.Linear(4, 2)
    payload = build_payload(
        stage=1, model=model, epoch=2, step=10, cfg={"a": 1}, metrics={"x": 0.5}
    )
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, payload)
    loaded = load_checkpoint(path, expected_stage=1)
    assert loaded["epoch"] == 2 and loaded["step"] == 10
    assert loaded["metrics"]["x"] == pytest.approx(0.5)


def test_batch_plan_absorbs_world_size():
    cfg = _cfg()
    one = resolve_batch_plan(cfg, world_size=1, num_emotions=5)
    four = resolve_batch_plan(cfg, world_size=4, num_emotions=5)
    assert one["per_gpu_batch_size"] == 40 and one["grad_accum_steps"] == 4
    assert four["grad_accum_steps"] == 1
    assert four["effective_batch_size"] == 160
