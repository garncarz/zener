import os
import json

import numpy as np
import pandas as pd
import pytest

from zener.optimize import GASubject
from zener.settings import settings
from zener.model import compute_all
from zener.common import rows_to_dataframe


@pytest.fixture
def n_f_one():
    original_n_f = settings.n_f
    settings.n_f = 1
    yield
    settings.n_f = original_n_f


@pytest.fixture
def n_f_zero():
    original_n_f = settings.n_f
    settings.n_f = 0
    yield
    settings.n_f = original_n_f


def create_model_data(params, Rs, Tp, dt=0.1):
    rows = compute_all(params, Rs, Tp, dt)
    return rows_to_dataframe(rows)


@pytest.fixture
def setup_test_data(keep_tmp_dir, request, n_f_one):
    """Create test directory, input.json and ZenerModel.dat"""
    label = f'test_{request.node.name}'
    os.makedirs(label, exist_ok=True)

    Rs = [0.01, -0.01, 0.01]
    Tp = [10.0, 10.0, 10.0]
    params = [25.0, 0.2, 3.0, 2.0, 1.0, 50.0]

    input_data = {
        'Tp': Tp,
        'Rs': Rs,
    }
    with open(f'{label}/input.json', 'w') as f:
        json.dump(input_data, f)

    target_df = create_model_data(params, Rs, Tp)
    target_df.to_csv(f'{label}/ZenerModel.dat', sep='\t', index=False)

    subject = GASubject(label)

    return label, params, Rs, Tp, target_df, subject


def test_perfect_match(setup_test_data):
    '''Test that identical model and target data gives fitness ≈ 0'''
    label, params, Rs, Tp, target_df, subject = setup_test_data

    subject.compute_all = lambda params: target_df.copy()

    fitness = subject.fitness_func(params)

    assert abs(fitness) < 1e-10


def test_with_difference(setup_test_data):
    '''Test that different model and target data gives lower (more negative) fitness'''
    label, params, Rs, Tp, target_df, subject = setup_test_data

    # Return data with slightly different parameters
    model_params = [25.1, 0.21, 3.1, 2.1, 1.1, 51.0]
    subject.compute_all = lambda params: create_model_data(model_params, Rs, Tp)

    fitness = subject.fitness_func(params)

    assert fitness < -1e-6


def test_shorter_target(setup_test_data):
    '''Test merging with shorter target data'''
    label, params, Rs, Tp, target_df, subject = setup_test_data

    first_half = target_df.copy()
    second_half = target_df.copy()
    max_time = target_df['time'].max()
    time_step = target_df['time'].iloc[1] - target_df['time'].iloc[0]
    second_half['time'] = second_half['time'] + max_time + time_step

    double_df = pd.concat([first_half, second_half], ignore_index=True)

    subject.compute_all = lambda params: double_df

    fitness = subject.fitness_func(params)

    assert fitness is not None
    assert not np.isnan(fitness)

    merged = subject.compute_and_merge(params)
    assert len(merged) <= len(target_df), 'Merged data should not exceed target length'


@pytest.mark.skip(reason='Big penalty for poor match ratio is not applied now.')
def test_shorter_model(setup_test_data):
    '''Test that having a much shorter model than target returns -1e9'''
    label, params, Rs, Tp, target_df, subject = setup_test_data

    subject.compute_all = lambda params: target_df.iloc[:int(len(target_df) * 0.05)]

    fitness = subject.fitness_func(params)

    assert fitness == -1e9, f'Expected -1e9 for poor match ratio, got {fitness}'


def test_n_f_zero(keep_tmp_dir, n_f_zero):
    '''Test that fitness function works with n_f=0'''
    label = 'test_n_f_zero'
    os.makedirs(label, exist_ok=True)

    Rs = [0.01, -0.01, 0.01]
    Tp = [10.0, 10.0, 10.0]

    params = [25.0, 0.2, 3.0, 2.0]

    input_data = {
        'Tp': Tp,
        'Rs': Rs,
    }
    with open(f'{label}/input.json', 'w') as f:
        json.dump(input_data, f)

    target_df = create_model_data(params, Rs, Tp)
    target_df.to_csv(f'{label}/ZenerModel.dat', sep='\t', index=False)

    subject = GASubject(label)
    subject.compute_all = lambda params: target_df

    fitness = subject.fitness_func(params)

    assert fitness is not None
    assert not np.isnan(fitness)
    assert abs(fitness) < 1e-10, f'Expected fitness ≈ 0 for perfect match, got {fitness}'
