from zener import model

def test_compute_all_non_negative():
    Nsamp = 1000

    Em = 1.0
    Tau_e = 50.0
    Tau_s = 25.0
    EmNL = 0.2
    Alpha = 3.0

    Tp = 100.0
    dt = Tp / Nsamp

    Rsig = [1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 1, 0, 0, -1, -1, 1, 1, 0, 0, -1, -1]

    params = (Em, Tau_e, Tau_s, EmNL, Alpha)
    rows = model.compute_all(params, Rsig, Tp, dt)

    tolerance = 1e-10
    assert all(
        all(i >= -tolerance for i in row)
        for row in rows
    )
