from numba import njit
from numba.typed import List
import numpy as np


Np = 21  # Number of periods Tp
Nsamp = 1000  # Number of samples per period
N_F = 3 # Number of Kelvin-Voigt elements

InitialFvalSig = 0.0
t0 = 0.0
R = 0.01

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
def fEpsVE(x, i, Rs, cumulative_Tp, Em, Tau_s, Tau_e):
    epsVE = fEps0(x - cumulative_Tp[i - 2] if i > 1 else x, Rs[i - 1], Em, Tau_s, Tau_e)
    for k in range(1, i):
        t_k_start = cumulative_Tp[k - 2] if k > 1 else 0.0
        t_k_end = cumulative_Tp[k - 1]
        epsVE += fEps(x, t_k_start, t_k_end, Rs[k - 1], Em, Tau_s, Tau_e)
    return epsVE

@njit
def fEpsKV(x, i, Rs, cumulative_Tp, Em, Tau_e, Tau_s):
    epsKV = fEps0(x - cumulative_Tp[i - 2] if i > 1 else x, Rs[i - 1], Em, Tau_s, Tau_e)
    for k in range(1, i):
        t_k_start = cumulative_Tp[k - 2] if k > 1 else 0.0
        t_k_end = cumulative_Tp[k - 1]
        epsKV += fEps(x, t_k_start, t_k_end, Rs[k - 1], Em, Tau_s, Tau_e)
    return epsKV

@njit
def _compute_all(params, Rs, Tp, dt):
    Tau_s, EmNL, Alpha = params[:3]
    Em_F = params[3:3 + N_F]
    Tau_e_F = params[3 + N_F:3 + 2 * N_F]
    Em, Tau_e = Em_F[0], Tau_e_F[0]

    # Cannot be done automatically:
    Tp_array = np.zeros(len(Tp))
    for i in range(len(Tp)):
        Tp_array[i] = Tp[i]

    cumulative_Tp = np.cumsum(Tp_array)

    rows = []
    t = 0.0
    FvalSig = InitialFvalSig

    for i in range(1, len(Tp) + 1):
        t_offset = cumulative_Tp[i - 2] if i > 1 else 0.0

        for j in range(int(Tp[i - 1] / dt)):
            t += dt

            valSig = FvalSig + fSig(t, t_offset, Rs[i - 1])
            valEpsNL = fEpsNL(valSig, EmNL, Alpha)
            valEpsVE = fEpsVE(t, i, Rs, cumulative_Tp, Em, Tau_s, Tau_e)
            valEps_NL_VE = valEpsNL + valEpsVE

            # Kelvin chain strain components
            valEpsKV_VE = List([0.0] * (N_F))
            valEpsKVchainVE = 0.0
            for k in range(1, N_F):
                if not np.isinf(Em_F[k]):
                    val = fEpsKV(t, i, Rs, cumulative_Tp, Em_F[k], Tau_e_F[k], Tau_s)
                    valEpsKV_VE[k] = val
                    valEpsKVchainVE += val

            valEpsTot = valEpsNL + valEpsVE + valEpsKVchainVE

            row = [t, valSig, valEpsNL, valEpsVE, valEps_NL_VE]
            row.extend(valEpsKV_VE[1:N_F])
            row.append(valEpsKVchainVE)
            row.append(valEpsTot)
            rows.append(row)

        FvalSig = valSig

    return rows


def compute_all(params, Rsig, Tp, dt):
    return _compute_all(List(params), List(r * R for r in Rsig), List(Tp), dt)
