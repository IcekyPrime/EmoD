import pytest

torch = pytest.importorskip("torch")

from emod.models.dppg import DPPGTokenizer  # noqa: E402


def _toy_frames(num_phones=3, clusters=3, per_cluster=200, dim=3, seed=0):
    gen = torch.Generator().manual_seed(seed)
    probs, phones = [], []
    for phone in range(num_phones):
        for cluster in range(clusters):
            center = torch.zeros(dim)
            center[phone] = 3.0
            center[(phone + 1) % dim] = float(cluster)
            probs.append(center + 0.1 * torch.randn(per_cluster, dim, generator=gen))
            phones.append(torch.full((per_cluster,), phone, dtype=torch.long))
    return torch.cat(probs), torch.cat(phones)


def test_fit_encode_and_roundtrip():
    probs, phones = _toy_frames()
    tokenizer = DPPGTokenizer(num_phones=3, max_clusters=4, min_clusters=2)
    report = tokenizer.fit(probs, phones, max_frames=10_000, samples_per_phone=2_000)
    assert report, "k-means should find clusters"

    ids = tokenizer.encode(probs.unsqueeze(0), phones.unsqueeze(0))
    assert ids.shape == phones.unsqueeze(0).shape
    assert int(ids.max()) < tokenizer.vocab_size

    restored = DPPGTokenizer.from_state(tokenizer.state())
    ids2 = restored.encode(probs.unsqueeze(0), phones.unsqueeze(0))
    assert torch.equal(ids, ids2)
