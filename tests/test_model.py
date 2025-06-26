import numpy as np

from zener.common import rows_to_dataframe
from zener import model

def test_compute_all_non_negative(n_f_setter):
    n_f_setter(3)

    Em = 1.0
    Tau_e = 50.0
    Tau_s = 25.0
    EmNL = 0.2
    Alpha = 3.0
    Beta = 2.0
    R = 0.01

    # Kelvin-Voigt elements
    Em_F = [Em, 1.0, float('inf')]
    Tau_e_F = [Tau_e, 5000.0, 50000.0]

    dt = 0.1
    Rsig = [1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 1, 0, 0, -1, -1, 1, 1, 0, 0, -1, -1]
    Rs = [float(r) * R for r in Rsig]
    Tp = [100.0] * len(Rsig)

    params = [Tau_s, EmNL, Alpha, Beta]
    params.extend(Em_F)
    params.extend(Tau_e_F)

    rows = model.compute_all(params, Rs, Tp, dt)

    tolerance = 1e-10
    assert all(
        all(i >= -tolerance for i in row)
        for row in rows
    )


def test_numerical_stability_no_nans(n_f_setter):
    """
    Test to ensure numerical stability in the model calculations,
    especially for Kelvin chain strain components over long time series.

    This test verifies that no NaN values appear in any output column,
    even when processing many segments.
    """
    n_f_setter(19)

    Tau_s = 11.61393
    EmNL = 49149.19
    Alpha = 32.26366
    Beta = 0.2590523
    R = 0.01

    # Create many segments to trigger potential numerical issues
    num_segments = 30

    # Kelvin-Voigt elements with values that could cause numerical challenges
    Em_F = [
        988996.2,
        994352.8,
        940482.1,
        956688.9,
        914027.5,
        174973.5,
        366677.5,
        439179.8,
        863775.1,
        938499.1,
        483563.4,
        34802.13,
        870446.9,
        420100.1,
        572273.2,
        543269.4,
        947502.3,
        639606.6,
        812826.4,
        235885.9,
    ]
    Tau_e_F = [
        0.9912868,
        2.240735,
        3.108988,
        5.203987,
        10.29208,
        23.40328,
        34.16473,
        64.08933,
        121.688,
        224.5968,
        298.8747,
        779.4171,
        1228.84,
        2121.714,
        3291.856,
        5582.434,
        12098.52,
        18459.13,
        31225.9,
        44949.32,
    ]

    dt = 1.0

    # Create alternating stress pattern
    Rsig = [1, -1] * (num_segments // 2 + 1)
    Rsig = Rsig[:num_segments]
    Rs = [float(r) * R for r in Rsig]

    # Each segment has different duration to mimic real-world scenario
    Tp = [100.0 + i * 10 for i in range(num_segments)]

    params = [Tau_s, EmNL, Alpha, Beta]
    params.extend(Em_F)
    params.extend(Tau_e_F)

    rows = model.compute_all(params, Rs, Tp, dt)
    df = rows_to_dataframe(rows)

    for col in df.columns:
        assert not np.isnan(df[col]).any(), f'NaN values found in column {col}'
