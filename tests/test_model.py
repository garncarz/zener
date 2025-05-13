from zener import model

def test_compute_all_non_negative():
    Em = 1.0
    Tau_e = 50.0
    Tau_s = 25.0
    EmNL = 0.2
    Alpha = 3.0

    # Kelvin-Voigt elements
    Em_F = [Em, 1.0, float('inf')]
    Tau_e_F = [Tau_e, 5000.0, 50000.0]

    Tp = [100.0] * model.Np
    dt = min(Tp) / model.Nsamp
    Rsig = [1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 1, 0, 0, -1, -1, 1, 1, 0, 0, -1, -1]

    params = [Tau_s, EmNL, Alpha]
    params.extend(Em_F)
    params.extend(Tau_e_F)

    rows = model.compute_all(params, Rsig, Tp, dt)

    tolerance = 1e-10
    assert all(
        all(i >= -tolerance for i in row)
        for row in rows
    )
