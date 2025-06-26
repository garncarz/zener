from contextlib import contextmanager
import os
import re

import pandas as pd

from .settings import settings
from . import model

@contextmanager
def change_dir(target_dir):
    prev_dir = os.getcwd()
    os.makedirs(target_dir, exist_ok=True)
    os.chdir(target_dir)
    try:
        yield
    finally:
        os.chdir(prev_dir)

def parse_signal_file(filename=settings.input_signals):
    signals = {}
    header = None

    with open(filename) as f:
        f.readline()

        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            label, values = parts[0], parts[1:]

            try:
                int_values = [int(v) for v in values]
                if not int_values:
                    header = label
                    continue
                signals[f'{header} - {label}'] = int_values
            except ValueError:
                header = label

    return signals

def parse_parameters_from_str_table(data):
    # Extract the first line of parameters
    header_line = re.search(r'EmNL\s+Alpha\s+Beta\s+Tau_s', data)
    values_line = re.search(r'([0-9.eE+-]+\s+){3}[0-9.eE+-]+', data)
    if not header_line or not values_line:
        raise ValueError('Could not find header or values line.')

    # Parse the first four parameters
    first_params = [float(x) for x in re.findall(r'[0-9.eE+-]+', values_line.group())]
    EmNL, Alpha, Beta, Tau_s = first_params

    # Extract Em_F and Tau_e_F values
    em_f = []
    tau_e_f = []
    for match in re.finditer(r'Em_F\[\d+\]\s+([0-9.eE+-]+)\s+Tau_e_F\[\d+\]\s+([0-9.eE+-]+)', data):
        em_f.append(float(match.group(1)))
        tau_e_f.append(float(match.group(2)))

    return {
        'Tau_s': Tau_s,
        'EmNL': EmNL,
        'Alpha': Alpha,
        'Beta': Beta,
        'Em_F': em_f,
        'Tau_e_F': tau_e_f,
    }

def dataframe_columns(only_results=False):
    columns = ['time', 'Stress'] if not only_results else []
    columns.extend(['StrainNonLinEl', 'StrainViscEl', 'StrainNL_VE_Total'])
    columns.extend([f'StrainKV_{i}' for i in range(1, settings.n_f)])
    columns.extend(['StrainKV_Total', 'TotStrain'])
    return columns

def rows_to_dataframe(rows):
    return pd.DataFrame(rows, columns=dataframe_columns())
