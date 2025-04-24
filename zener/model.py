from numba import njit
from numba.typed import List
import numpy as np


Np = 21  # Number of periods Tp
Nsamp = 1000  # Number of samples per period

Sig0 = 1
t0 = 0.0

@njit
def fSig(x, x1, Rs):
    return Rs * (x - x1)

@njit
def fEpsNL(Sig, EmNL, Alpha):
    return Sig / (EmNL * (1 + Alpha * Sig))

@njit
def fEps(x, x1, x2, Rs, Em, Tau_s, Tau_e):
    return (Rs / Em) * (
        (x2 - x1) + (Tau_s - Tau_e) * (np.exp(x2 / Tau_e) - np.exp(x1 / Tau_e)) * np.exp(-x / Tau_e)
    )

@njit
def fEps0(x, Rs, Em, Tau_s, Tau_e):
    return fEps(x, t0, x, Rs, Em, Tau_s, Tau_e)

@njit
def fEpsVE(x, i, Rsig, R, Tp, Em, Tau_s, Tau_e):
    epsVE = fEps0(x - (i - 1) * Tp, Rsig[i - 1] * R, Em, Tau_s, Tau_e)
    for k in range(1, i):
        epsVE += fEps(x, (k - 1) * Tp, k * Tp, Rsig[k - 1] * R, Em, Tau_s, Tau_e)
    return epsVE


@njit
def _compute_all(params, Rsig, Tp, dt):
    Em, Tau_e, Tau_s, EmNL, Alpha = params

    t = 0.0
    FvalSig = 0.0

    R = Sig0 / Tp

    rows = []

    for i in range(1, Np + 1):
        for j in range(Nsamp):
            t += dt
            valSig = FvalSig + fSig(t, (i - 1) * Tp, Rsig[i - 1] * R)
            valEpsNL = fEpsNL(valSig, EmNL, Alpha)
            valEpsVE = fEpsVE(t, i, Rsig, R, Tp, Em, Tau_s, Tau_e)
            rows.append([
                t,
                valSig,
                valEpsNL,
                valEpsVE,
                valEpsNL + valEpsVE,
            ])

        FvalSig = valSig

    return rows


def compute_all(params, Rsig, Tp, dt):
    return _compute_all(params, List(Rsig), Tp, dt)
