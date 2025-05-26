import numpy as np
import pytest

from zener.optimize import transform_params
from zener.settings import settings


@pytest.fixture
def n_f_setter():
    original_n_f = settings.n_f

    def _set_n_f(value):
        settings.n_f = value

    yield _set_n_f

    settings.n_f = original_n_f


@pytest.mark.parametrize('n_f, test_params, expected', [
    # n_f=0: No Em_F parameters, nothing should change
    (0, [10.0, 0.2, 2.5], [10.0, 0.2, 2.5]),

    # n_f=1: One Em_F parameter at index 3 should be transformed
    (1, [10.0, 0.2, 2.5, 2.0, 30.0], [10.0, 0.2, 2.5, 100.0, 30.0]),

    # n_f=2: Two Em_F parameters at indices 3 and 4 should be transformed
    (2, [10.0, 0.2, 2.5, 2.0, 3.0, 30.0, 40.0],
       [10.0, 0.2, 2.5, 100.0, 1000.0, 30.0, 40.0]),

    # n_f=3: Three Em_F parameters should be transformed
    (3, [10.0, 0.2, 2.5, 2.0, 3.0, -1.0, 30.0, 40.0, 50.0],
       [10.0, 0.2, 2.5, 100.0, 1000.0, 0.1, 30.0, 40.0, 50.0]),

    # Test negative exponents
    (1, [10.0, 0.2, 2.5, -1.0, 30.0], [10.0, 0.2, 2.5, 0.1, 30.0]),

    # Test zero exponents
    (1, [10.0, 0.2, 2.5, 0.0, 30.0], [10.0, 0.2, 2.5, 1.0, 30.0]),
])
def test_transform_params(n_f_setter, n_f, test_params, expected):
    n_f_setter(n_f)
    original_params = test_params.copy()

    result = transform_params(test_params)

    assert np.allclose(result, expected)
    assert len(result) == len(expected)

    # Check that original parameters are unchanged
    assert np.array_equal(test_params, original_params)
