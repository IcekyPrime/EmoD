import pytest

torch = pytest.importorskip("torch")

from emod.utils.geometry import (  # noqa: E402
    additivity_error,
    l2_normalize,
    project_intensity,
    weighted_direction,
)


def test_l2_normalize():
    x = torch.randn(4, 8)
    y = l2_normalize(x, dim=-1)
    assert torch.allclose(y.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_projection_recovers_lambda():
    direction = l2_normalize(torch.randn(8), dim=-1)
    scale = 2.5
    lam = 1.7
    delta = direction * (scale * lam)
    value = project_intensity(delta, direction, scale)
    assert value.item() == pytest.approx(lam, rel=1e-4)


def test_weighted_direction_normalized():
    table = l2_normalize(torch.randn(4, 8), dim=-1)
    vec = weighted_direction([0.8, 0.2, 0.0, 0.0], table)
    assert vec.norm().item() == pytest.approx(1.0, rel=1e-4)
    assert additivity_error(vec, vec, 1.0).item() == pytest.approx(0.0, abs=1e-6)
