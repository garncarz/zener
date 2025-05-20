from contextlib import contextmanager
import os

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

def dataframe_columns(only_results=False):
    columns = ['time', 'Stress'] if not only_results else []
    columns.extend(['StrainNonLinEl', 'StrainViscEl', 'StrainNL_VE_Total'])
    columns.extend([f'StrainKV_{i}' for i in range(1, settings.n_f)])
    columns.extend(['StrainKV_Total', 'TotStrain'])
    return columns

def rows_to_dataframe(rows):
    return pd.DataFrame(rows, columns=dataframe_columns())
