import json
import logging
import random

import numpy as np
import pandas as pd

from .common import change_dir, parse_signal_file
from .settings import settings
from . import model


logger = logging.getLogger(__name__)


def noise_for(val):
    return random.triangular(val - settings.noise_range, val + settings.noise_range, val)


def generate_testing_data(Np, Nsamp, Em_F, Tau_e_F, Tau_s, EmNL, Alpha, Sig0, Tp, Rsig):
    dt = min(Tp) / Nsamp  # Compute dt as the minimal step between periods

    params = [Tau_s, EmNL, Alpha]
    params.extend(Em_F)
    params.extend(Tau_e_F)

    rows = model.compute_all(params, Rsig, Tp, dt)

    columns = ['time', 'Stress', 'StrainNonLinEl', 'StrainViscEl', 'StrainNL_VE_Total']
    columns.extend([f'StrainKV_{i}' for i in range(1, model.N_F)])
    columns.extend(['StrainKV_Total', 'TotStrain'])

    df = pd.DataFrame(rows, columns=columns)

    df['TotStrain'] = df['TotStrain'].apply(noise_for)

    return df


def generate_all_testing_data():
    logger.info(f'Generating testing data using noise = {settings.noise_range}...')

    # === Parameters ===
    Np = 21  # Number of periods Tp
    Nsamp = 1000  # Number of samples per period

    # === Default Model Parameters ===
    Em = 1.0         # Modulus
    Tau_e = 50.0     # Relaxation time for strain
    Tau_s = 25.0     # Relaxation time for stress
    EmNL = 0.2       # Nonlinear modulus
    Alpha = 3.0      # Degree of nonlinearity
    Sig0 = 1.0       # Initial maximum stress
    Tp = [100.0] * Np

    # Kelvin-Voigt elements
    Em_F = [Em, 1.0, float('inf')]  # Moduli for KV elements
    Tau_e_F = [Tau_e, 5000.0, 50000.0]  # Relaxation times for KV elements

    with open('target.json', 'w') as f:
        json.dump({
            'Np': Np,
            'Nsamp': Nsamp,
            'Em_F': Em_F,
            'Tau_e_F': Tau_e_F,
            'Tau_s': Tau_s,
            'EmNL': EmNL,
            'Alpha': Alpha,
            'Sig0': Sig0,
            'Tp': Tp,
            'noise': settings.noise_range,
        }, f, indent=4)

    for label, Rsig in parse_signal_file().items():
        df = generate_testing_data(Np, Nsamp, Em_F, Tau_e_F, Tau_s, EmNL, Alpha, Sig0, Tp, Rsig)
        with change_dir(label):
            with open('Rsig.txt', 'w') as f:
                f.write(str(Rsig))

            with open('Tp.txt', 'w') as f:
                f.write(str(Tp))

            df.to_csv('ZenerModel.dat', sep='\t', index=False)
