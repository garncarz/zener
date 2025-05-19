import json
import logging
import random

import numpy as np

from .common import change_dir, parse_signal_file, rows_to_dataframe
from .settings import settings
from . import model


logger = logging.getLogger(__name__)


def noise_for(val):
    return random.triangular(val - settings.noise_range, val + settings.noise_range, val)


def generate_testing_data(Em_F, Tau_e_F, Tau_s, EmNL, Alpha, Tp, Rs, dt=0.1):
    params = [Tau_s, EmNL, Alpha]
    params.extend(Em_F)
    params.extend(Tau_e_F)

    rows = model.compute_all(params, Rs, Tp, dt)
    df = rows_to_dataframe(rows)

    if settings.noise_range:
        df['TotStrain'] = df['TotStrain'].apply(noise_for)

    return df


def generate_all_testing_data(Tp=None):
    logger.info(f'Generating testing data using noise = {settings.noise_range}...')

    # === Parameters ===
    Np = 21  # Number of periods Tp

    # === Default Model Parameters ===
    Em = 1.0         # Modulus
    Tau_e = 50.0     # Relaxation time for strain
    Tau_s = 25.0     # Relaxation time for stress
    EmNL = 0.2       # Nonlinear modulus
    Alpha = 3.0      # Degree of nonlinearity
    R = 0.01

    # Kelvin-Voigt elements
    Em_F = [Em, 1.0, float('inf')]  # Moduli for KV elements
    Tau_e_F = [Tau_e, 5000.0, 50000.0]  # Relaxation times for KV elements

    if Tp is None:
        Tp = [100.0] * Np

    with open('target.json', 'w') as f:
        json.dump({
            'Np': Np,
            'Em_F': Em_F,
            'Tau_e_F': Tau_e_F,
            'Tau_s': Tau_s,
            'EmNL': EmNL,
            'Alpha': Alpha,
            'noise': settings.noise_range,
        }, f, indent=4)

    for label, Rsig in parse_signal_file().items():
        Rs = [float(r) * R for r in Rsig]
        df = generate_testing_data(Em_F, Tau_e_F, Tau_s, EmNL, Alpha, Tp, Rs)

        with change_dir(label):
            with open('input.json', 'w') as f:
                json.dump({
                    'Tp': Tp,
                    'Rs': Rs,
                }, f, indent=4)

            df.to_csv('ZenerModel.dat', sep='\t', index=False)
