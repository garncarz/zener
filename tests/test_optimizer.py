import numpy as np
import pygad
import pytest

from zener.optimize import GAAll, dynamic_population_size, transform_params
from zener.settings import settings


@pytest.mark.parametrize('n_f, test_params, expected', [
    # n_f=0: No Em_F parameters, nothing should change
    (0, [10.0, 0.2, 2.5, 2.0], [10.0, 0.2, 2.5, 2.0]),

    # n_f=1: One Em_F parameter at index 4 should be transformed
    (1, [10.0, 0.2, 2.5, 2.0, 2.0, 30.0], [10.0, 0.2, 2.5, 2.0, 100.0, 30.0]),

    # n_f=2: Two Em_F parameters at indices 4 and 5 should be transformed
    (2, [10.0, 0.2, 2.5, 2.0, 2.0, 3.0, 30.0, 40.0],
       [10.0, 0.2, 2.5, 2.0, 100.0, 1000.0, 30.0, 40.0]),

    # n_f=3: Three Em_F parameters should be transformed
    (3, [10.0, 0.2, 2.5, 2.0, 2.0, 3.0, -1.0, 30.0, 40.0, 50.0],
       [10.0, 0.2, 2.5, 2.0, 100.0, 1000.0, 0.1, 30.0, 40.0, 50.0]),

    # Test negative exponents
    (1, [10.0, 0.2, 2.5, 2.0, -1.0, 30.0], [10.0, 0.2, 2.5, 2.0, 0.1, 30.0]),

    # Test zero exponents
    (1, [10.0, 0.2, 2.5, 2.0, 0.0, 30.0], [10.0, 0.2, 2.5, 2.0, 1.0, 30.0]),
])
def test_transform_params(n_f_setter, n_f, test_params, expected):
    n_f_setter(n_f)
    original_params = test_params.copy()

    result = transform_params(test_params)

    assert np.allclose(result, expected)
    assert len(result) == len(expected)

    # Check that original parameters are unchanged
    assert np.array_equal(test_params, original_params)


def test_dynamic_population_size_resizes_ga(monkeypatch):
   monkeypatch.setattr(settings, 'dynamic_population_initial', 2000)
   monkeypatch.setattr(settings, 'dynamic_population_final', 20)
   monkeypatch.setattr(settings, 'num_generations', 3)

   ga = pygad.GA(
      num_generations=3,
      sol_per_pop=20,
      num_genes=4,
      num_parents_mating=5,
      fitness_func=lambda _, solution, __: np.sum(solution),
   )
   ga.generations_completed = 1

   GAAll.__new__(GAAll)._resize_population(ga)

   assert ga.population.shape == (1340, 4)
   assert ga.sol_per_pop == 1340
   assert ga.num_offspring == 1339
   assert len(ga.last_generation_fitness) == 1340


@pytest.mark.parametrize('generation, expected', [
   (0, 2000),
   (1, 1340),
   (2, 680),
   (3, 20),
   (10, 20),
])
def test_dynamic_population_size_schedule(generation, expected):
   assert dynamic_population_size(generation, 3, 2000, 20) == expected
