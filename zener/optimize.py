import json
import logging

import matplotlib.pyplot as plt
from numba import njit
from numba.typed import List
import numpy as np
import pandas as pd
import pygad
from tqdm import tqdm

from .common import change_dir, parse_signal_file
from .settings import settings
from .statsd import statsd


logger = logging.getLogger(__name__)


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
def compute_all(params, Rsig, Tp, dt):
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


class GASubject:

    def __init__(self, label, Rsig):
        self.label = label
        self.Rsig = Rsig
        self.load_testing_data()

    def load_testing_data(self):
        df_target = pd.read_csv(f'{self.label}/ZenerModel.dat', sep='\s+')
        df_target['time'] = pd.to_numeric(df_target['time'], errors='coerce')
        df_target = df_target.dropna()
        self.df_target_sorted = df_target.sort_values('time')

        self.dt = self.df_target_sorted.iloc[1].time - self.df_target_sorted.iloc[0].time
        self.Tp = self.dt * Nsamp

        logger.info(f'Testing data loaded: {self.label}')

    def compute_all(self, params):
        return pd.DataFrame(
            compute_all(params, List(self.Rsig), self.Tp, self.dt),
            columns=['time', 'Stress', 'StrainNonLinEl', 'StrainViscEl', 'TotStrain'],
        )

    def compute_and_merge(self, params):
        df = self.compute_all(params)
        df_sorted = df.sort_values('time')

        merged = pd.merge_asof(
            df_sorted,
            self.df_target_sorted,
            on='time',
            direction='nearest',
            tolerance=1e-8,
            suffixes=('_model', '_target'),
        )
        return merged

    def fitness_func(self, solution):
        merged = self.compute_and_merge(solution)

        match_ratio = len(merged) / len(self.df_target_sorted)
        if match_ratio < 0.95:
            return -1e9

        error = 0.0
        for col in ['TotStrain']:
            diff = merged[f'{col}_model'] - merged[f'{col}_target']
            error += np.sum(diff ** 2)

        return -error

    def save_plot(self, solution):
        with change_dir(self.label):
            merged = self.compute_and_merge(solution)
            self._plot_time_series(merged)
            self._plot_hysteresis(merged)

    def _plot_time_series(self, merged):
        cols = ['Stress', 'StrainNonLinEl', 'StrainViscEl', 'TotStrain']
        fig, axes = plt.subplots(len(cols), 1, figsize=(10, 2.5 * len(cols)), sharex=True)

        for i, col in enumerate(cols):
            ax = axes[i]
            ax.plot(
                merged['time'].to_numpy(),
                merged[f'{col}_target'].to_numpy(),
                label='Target',
                color='black',
                linestyle='--',
            )
            ax.plot(
                merged['time'].to_numpy(),
                merged[f'{col}_model'].to_numpy(),
                label='Model',
                color='blue',
            )
            ax.set_ylabel(col)
            ax.legend()
            ax.grid(True)

        axes[-1].set_xlabel('Time')
        plt.tight_layout()
        plt.savefig('plots.png')
        plt.close(fig)

    def _plot_hysteresis(self, merged):
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(
            merged['StrainViscEl_target'].to_numpy(),
            merged['Stress_target'].to_numpy(),
            label='Target StrainViscEl', color='gray', linestyle=':',
        )
        ax.plot(
            merged['StrainViscEl_model'].to_numpy(),
            merged['Stress_model'].to_numpy(),
            label='Model StrainViscEl', color='black', linestyle='--',
        )
        ax.plot(
            merged['TotStrain_target'].to_numpy(),
            merged['Stress_target'].to_numpy(),
            label='Target TotStrain', color='blue', linestyle=':',
        )
        ax.plot(
            merged['TotStrain_model'].to_numpy(),
            merged['Stress_model'].to_numpy(),
            label='Model TotStrain', color='lightblue', linestyle='--',
        )

        ax.set_xlabel('Strain')
        ax.set_ylabel('Stress')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.savefig('hysteresis.png')
        plt.close(fig)


class GAAll:

    def __init__(self):
        self.subjects = {label: GASubject(label, Rsig) for label, Rsig in parse_signal_file().items()}

        self.ga = self.load_ga() or pygad.GA(
            num_generations=settings.num_generations,
            stop_criteria=[f'reach_{settings.stop_fitness}', f'saturate_{settings.stop_saturation}'],
            num_parents_mating=5,
            parent_selection_type='nsga2',  # good for multi-objective
            fitness_func=self.fitness_func,
            sol_per_pop=20,
            num_genes=5,
            mutation_percent_genes=30,
            mutation_by_replacement=True,
            gene_space=[
                {'low': 0.1, 'high': 10.0},  # Em
                {'low': 10, 'high': 100},    # Tau_e
                {'low': 10, 'high': 100},    # Tau_s
                {'low': 0.01, 'high': 3},    # EmNL
                {'low': 0.1, 'high': 10},    # Alpha
            ],
            save_best_solutions=True,
        )
        self.ga.on_generation = self.on_generation

        if settings.parallel > 1:
            self.ga.parallel_processing=['process', settings.parallel]
            logger.info(f'PyGAD using {settings.parallel} parallel processes.')

    def load_ga(self):
        try:
            ga = pygad.load(settings.ga_filename)
            logger.info(
                f'Saved GA instance loaded from {settings.ga_filename} '
                f'with {ga.generations_completed} generations completed.'
            )

            # this needs to be renewed:
            ga.fitness_func = self.fitness_func
            ga.num_generations = settings.num_generations

            return ga
        except Exception:
            return None

    def fitness_func(self, ga_instance, solution, solution_idx):
        return [subj.fitness_func(solution) for subj in self.subjects.values()]

    def on_generation(self, ga):
        logger.info(
            f'Generation {ga.generations_completed}/{settings.num_generations}, '
            f'fitness: {ga.last_generation_fitness}'
        )
        self.ga.save(settings.ga_filename)

        with statsd.batch():
            statsd.gauge('generation.fitness.max_min', np.max(np.min(ga.last_generation_fitness, axis=1)))
            statsd.gauge('generation.fitness.mean', np.mean(np.mean(ga.last_generation_fitness, axis=1)))
            statsd.gauge('generation', ga.generations_completed)

            Em, Tau_e, Tau_s, EmNL, Alpha = ga.best_solutions[-1]
            statsd.gauge('generation.model.Em', Em)
            statsd.gauge('generation.model.Tau_e', Tau_e)
            statsd.gauge('generation.model.Tau_s', Tau_s)
            statsd.gauge('generation.model.EmNL', EmNL)
            statsd.gauge('generation.model.Alpha', Alpha)

    def run(self):
        self.ga.run()

    def save_result(self):
        solution, solution_fitness, _ = self.ga.best_solution()
        logger.info(f'Best solution: {solution}, fitness: {solution_fitness}')

        Em, Tau_e, Tau_s, EmNL, Alpha = solution
        with open('solution.json', 'w') as f:
            json.dump({
                'Em': Em,
                'Tau_e': Tau_e,
                'Tau_s': Tau_s,
                'EmNL': EmNL,
                'Alpha': Alpha,
                'generations': self.ga.generations_completed,
                'fitness': solution_fitness.tolist(),
            }, f, indent=4)

        plt_fitness = self.ga.plot_fitness()
        plt_fitness.savefig('fitness.png')

        for subj in self.subjects.values():
            subj.save_plot(solution)
