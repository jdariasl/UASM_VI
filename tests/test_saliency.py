import numpy as np
import pytest

from uasm_vi import uncertainty_aware_saliency


def _linear(w):
    return lambda x: (float(w @ x), w)


def test_relevant_features_get_higher_saliency():
    w = np.array([2.0, 0.0, 0.0, -0.0, 2.0, 0.0])
    x = np.ones(6)
    r = uncertainty_aware_saliency(_linear(w), x, n_steps=200, seed=0)
    assert r.saliency.shape == x.shape == r.uncertainty.shape
    assert r.saliency[[0, 4]].min() > r.saliency[[1, 2, 3, 5]].max()
    assert (r.uncertainty > 0).all()
    assert np.isfinite(r.elbo_history).all()


def test_deterministic_with_seed():
    f = _linear(np.array([1.0, -1.0]))
    a = uncertainty_aware_saliency(f, np.ones(2), n_steps=20, seed=1)
    b = uncertainty_aware_saliency(f, np.ones(2), n_steps=20, seed=1)
    assert np.allclose(a.saliency, b.saliency)


def test_invalid_args():
    with pytest.raises(ValueError):
        uncertainty_aware_saliency(_linear(np.ones(2)), np.ones(2), prior_std=0)
