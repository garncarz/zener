from numba import njit
from numba.typed import List
import numpy as np

from .settings import settings

InitialFvalSig = 0.0
t0 = 0.0

@njit
def fSig(x, x1, Rs):
    return Rs * (x - x1)

@njit
def fEpsNL(Sig, EmNL, Alpha, Beta):
    return (1 / EmNL) * (Sig + Alpha * np.sign(Sig) * np.abs(Sig)**Beta)

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
def _compute_all(params, Rs, Tp, dt, n_f):
    Tau_s, EmNL, Alpha, Beta = params[:4]
    Em_F = params[4:4 + n_f]
    Tau_e_F = params[4 + n_f:4 + 2 * n_f]

    if n_f:
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
            valEps_NL_VE = valEpsNL = fEpsNL(valSig, EmNL, Alpha, Beta)
            valEpsKVchainVE = valEpsVE = 0.0

            if n_f:
                valEpsVE = fEpsVE(t, i, Rs, cumulative_Tp, Em, Tau_s, Tau_e)
                valEps_NL_VE += valEpsVE

                # Kelvin chain strain components
                valEpsKV_VE = List([0.0] * n_f)
                for k in range(1, n_f):
                    if not np.isinf(Em_F[k]):
                        val = fEpsKV(t, i, Rs, cumulative_Tp, Em_F[k], Tau_e_F[k], Tau_s)
                        valEpsKV_VE[k] = val
                        valEpsKVchainVE += val

            valEpsTot = valEpsNL + valEpsVE + valEpsKVchainVE

            row = [t, valSig, valEpsNL, valEpsVE, valEps_NL_VE]
            if n_f:
                row.extend(valEpsKV_VE[1:])
            row.append(valEpsKVchainVE)
            row.append(valEpsTot)
            rows.append(row)

        FvalSig = valSig

    return rows


def compute_all(params, Rs, Tp, dt, n_f=None):
    if n_f is None:
        n_f = settings.n_f
    return _compute_all(List(params), List(Rs), List(Tp), dt, n_f)


# correct JIT compilation with Numba:
compute_all([1.2] * (4 + 2*3), [1.0, 2.0], [1.0, 2.0], 0.1, 3)
# otherwise it doesn't know what to do with n_f = 0
