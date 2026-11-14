import math

import pytest

from training_utils import PlateauScale, warmup_cosine_lr, perplexity_from_loss


def test_plateau_scale_reduces_learning_rate_after_patience():
    controller = PlateauScale(patience=2, factor=0.5, min_scale=0.125)
    assert controller.observe(1.0) == 1.0
    assert controller.observe(1.01) == 1.0
    assert controller.observe(1.02) == 0.5
    assert controller.observe(1.03) == 0.5
    assert controller.observe(1.04) == 0.25
    assert controller.observe(1.05) == 0.25
    assert controller.observe(1.06) == 0.125
    assert controller.observe(1.07) == 0.125


def test_plateau_scale_resets_patience_after_real_improvement():
    controller = PlateauScale(patience=2, factor=0.5, min_scale=0.125)
    controller.observe(1.0)
    controller.observe(1.01)
    assert controller.observe(0.9) == 1.0
    assert controller.observe(0.91) == 1.0
    assert controller.observe(0.92) == 0.5


def test_plateau_scale_state_round_trip():
    controller = PlateauScale(patience=2)
    controller.observe(1.0)
    controller.observe(1.1)
    restored = PlateauScale(patience=2)
    restored.load_state_dict(controller.state_dict())
    assert restored.best == pytest.approx(1.0)
    assert restored.bad_evals == 1
    assert restored.scale == pytest.approx(1.0)


def test_warmup_cosine_lr_stays_bounded_and_decays():
    values = [warmup_cosine_lr(step, base_lr=1.0, warmup=2, total_steps=10,
                               min_ratio=0.1) for step in range(10)]
    assert values[0] == pytest.approx(0.5)
    assert values[1] == pytest.approx(1.0)
    assert values[2] == pytest.approx(1.0)
    assert values[-1] >= 0.1
    assert values[-1] < values[2]
    assert all(math.isfinite(x) and 0 < x <= 1.0 for x in values)


def test_perplexity_has_mathematical_lower_bound_one():
    assert perplexity_from_loss(0.0) == pytest.approx(1.0)
    assert perplexity_from_loss(math.log(2.0)) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        perplexity_from_loss(-0.1)
