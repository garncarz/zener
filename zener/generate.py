import json
import logging
import random

import numpy as np
import pandas as pd

from .common import change_dir, parse_signal_file
from .settings import settings


logger = logging.getLogger(__name__)


def noise_for(val):
    return random.triangular(val - settings.noise_range, val + settings.noise_range, val)


def generate_testing_data(Np, Nsamp, Em, Tau_e, Tau_s, EmNL, Alpha, Sig0, Tp, Rsig):
    # === Derived Values ===
    R = Sig0 / Tp
    dt = Tp / Nsamp
    t = 0.0
    FvalSig = 0.0
    FvalEps = 0.0
    t0 = 0.0

    # === Model functions ===
    def fSig(x, x1, Rs):
        return Rs * (x - x1)

    def fEps(x, x1, x2, Rs):
        return (Rs / Em) * ((x2 - x1) + (Tau_s - Tau_e) *
                            (np.exp(x2 / Tau_e) - np.exp(x1 / Tau_e)) *
                            np.exp(-x / Tau_e))

    def fEps0(x, Rs):
        return fEps(x, t0, x, Rs)

    def fEpsNL(Sig):
        return Sig / (EmNL * (1 + Alpha * Sig))

    def fEpsVE(x, i):
        eps = fEps0(x - (i - 1) * Tp, Rsig[i - 1] * R)
        for k in range(1, i):
            eps += fEps(x, (k - 1) * Tp, k * Tp, Rsig[k - 1] * R)
        return eps

    rows = []

    for i in range(1, Np + 1):
        for j in range(Nsamp):
            t += dt
            valSig = FvalSig + fSig(t, (i - 1) * Tp, Rsig[i - 1] * R)
            valEpsNL = fEpsNL(valSig)
            valEpsVE = fEpsVE(t, i)
            rows.append([
                t,
                valSig,
                valEpsNL,
                valEpsVE,
                noise_for(valEpsNL + valEpsVE),
            ])
        FvalSig = valSig

    df = pd.DataFrame(rows, columns=['time', 'Stress', 'StrainNonLinEl', 'StrainViscEl', 'TotStrain'])
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
    Tp = 100.0       # Period

    with open('target.json', 'w') as f:
        json.dump({
            'Np': Np,
            'Nsamp': Nsamp,
            'Em': Em,
            'Tau_e': Tau_e,
            'Tau_s': Tau_s,
            'EmNL': EmNL,
            'Alpha': Alpha,
            'Sig0': Sig0,
            'Tp': Tp,
            'noise': settings.noise_range,
        }, f, indent=4)

    for label, Rsig in parse_signal_file().items():
        df = generate_testing_data(Np, Nsamp, Em, Tau_e, Tau_s, EmNL, Alpha, Sig0, Tp, Rsig)
        with change_dir(label):
            with open('Rsig.txt', 'w') as f:
                f.write(str(Rsig))
            df.to_csv('ZenerModel.dat', sep='\t', index=False)
