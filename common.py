from contextlib import contextmanager
import os

@contextmanager
def change_dir(target_dir):
    prev_dir = os.getcwd()
    os.makedirs(target_dir, exist_ok=True)
    os.chdir(target_dir)
    try:
        yield
    finally:
        os.chdir(prev_dir)

def parse_signal_file(filename='Rsig-Samp.txt'):
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
