import json
import logging

import numpy as np
import pandas as pd
import pygad

from . import model, plots, tasks
from .common import change_dir, parse_signal_file, rows_to_dataframe, dataframe_columns
from .settings import settings
from .statsd import statsd


logger = logging.getLogger(__name__)


class GASubject:

    def __init__(self, label, Rsig):
        self.label = label
        self.Rsig = Rsig
        self.load_testing_data()

    def load_testing_data(self):
        try:
            df_target = pd.read_csv(f'{self.label}/ZenerModel.dat', sep='\s+')
        except FileNotFoundError:
            df_target = self.load_real_data_txt()

        df_target['time'] = pd.to_numeric(df_target['time'], errors='coerce')
        df_target = df_target.dropna()
        self.df_target_sorted = df_target.sort_values('time')

        with open(f'{self.label}/Tp.txt', 'r') as f:
            self.Tp = json.loads(f.read())

        # Calculate dt as the minimal step between periods
        self.dt = min(self.Tp) / model.Nsamp

        logger.info(f'Testing data loaded: {self.label}')

    def load_real_data_txt(self):
        filepath = f'{self.label}/data.txt'

        ignore_lines = 0
        with open(filepath, 'r') as file:
            for line in file.readlines():
                if line.strip() != '':
                    ignore_lines += 1
                else:
                    break

        df = pd.read_csv(
            filepath,
            sep='\t',
            skiprows=ignore_lines,
            header=[0, 1],  # column names and units
        )

        df.columns = [col[0].strip() for col in df.columns.values]
        df.rename(columns={
            'Time': 'time',
            'Axial Calculated Axial Stress': 'Stress',
            'Axial Avg Strain': 'TotStrain',
        }, inplace=True)

        return df

    def compute_all(self, params):
        return rows_to_dataframe(
            model.compute_all(params, self.Rsig, self.Tp, self.dt)
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

        for col in dataframe_columns(only_results=True):
            if (model_col := f'{col}_model') not in merged.columns:
                merged.rename(columns={col: model_col}, inplace=True)

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

    def save_plots(self, solution):
        with change_dir(self.label):
            merged = self.compute_and_merge(solution)
            plots.plot_time_series(merged)
            plots.plot_hysteresis(merged)


def load_subjects(just_labels=False):
    return {
        label: GASubject(label, Rsig) if not just_labels else 'Use celery'
        for label, Rsig in parse_signal_file().items()
    }


class GAAll:

    def __init__(self):
        self.subjects = load_subjects(just_labels=settings.use_celery)

        self.ga = self.load_ga() or pygad.GA(
            num_generations=settings.num_generations,
            stop_criteria=[f'reach_{settings.stop_fitness}', f'saturate_{settings.stop_saturation}'],
            num_parents_mating=5,
            parent_selection_type='nsga2',  # good for multi-objective
            fitness_func=self.fitness_func,
            sol_per_pop=20,
            num_genes=3 + 2 * model.N_F,
            mutation_percent_genes=30,
            mutation_by_replacement=True,
            gene_space = [
                {'low': 10, 'high': 100},    # Tau_s
                {'low': 0.01, 'high': 3},    # EmNL
                {'low': 0.1, 'high': 10},    # Alpha
                *[{'low': 0.0, 'high': 1e12} for _ in range(model.N_F)],  # Em_F
                *[{'low': 0.0, 'high': 1e12} for _ in range(model.N_F)],  # Tau_e_F
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

    def _is_tau_increasing(self, solution):
        Tau_e_F = solution[3 + model.N_F:3 + 2 * model.N_F]
        return all(Tau_e_F[i] <= Tau_e_F[i + 1] for i in range(len(Tau_e_F) - 1))

    def fitness_func(self, ga_instance, solution, solution_idx):
        if not self._is_tau_increasing(solution):
            return [-1e9] * len(self.subjects)

        if settings.use_celery:
            fitness_tasks = tasks.group(
                tasks.compute_fitness.s(subj_label, solution.tolist())
                for subj_label in self.subjects.keys()
            )
            return fitness_tasks.apply_async().get()
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

            best_sol = ga.best_solutions[-1]
            Tau_s, EmNL, Alpha = best_sol[:3]
            Em_F = best_sol[3:3 + model.N_F]
            Tau_e_F = best_sol[3 + model.N_F:3 + 2 * model.N_F]
            Em, Tau_e = Em_F[0], Tau_e_F[0]

            statsd.gauge('generation.model.Em', Em)
            statsd.gauge('generation.model.Tau_e', Tau_e)
            statsd.gauge('generation.model.Tau_s', Tau_s)
            statsd.gauge('generation.model.EmNL', EmNL)
            statsd.gauge('generation.model.Alpha', Alpha)

            for i in range(1, model.N_F):
                statsd.gauge(f'generation.model.Em_F_{i}', Em_F[i])
                statsd.gauge(f'generation.model.Tau_e_F_{i}', Tau_e_F[i])

    def run(self):
        self.ga.run()

    def save_result(self):
        solution, solution_fitness, _ = self.ga.best_solution()
        logger.info(f'Best solution: {solution}, fitness: {solution_fitness}')

        Tau_s, EmNL, Alpha = solution[:3]
        Em_F = solution[3:3 + model.N_F]
        Tau_e_F = solution[3 + model.N_F:3 + 2 * model.N_F]
        Em, Tau_e = Em_F[0], Tau_e_F[0]

        with open('solution.json', 'w') as f:
            json.dump({
                'Tau_s': Tau_s,
                'EmNL': EmNL,
                'Alpha': Alpha,
                'Em_F': Em_F.tolist(),
                'Tau_e_F': Tau_e_F.tolist(),
                'generations': self.ga.generations_completed,
                'fitness': solution_fitness.tolist(),
            }, f, indent=4)

        plt_fitness = self.ga.plot_fitness()
        plt_fitness.savefig('fitness.png')

        if settings.use_celery:
            plots_tasks = tasks.group(
                tasks.save_plots.s(subj_label, solution.tolist())
                for subj_label in self.subjects.keys()
            )
            plots_tasks.apply_async().join()
        else:
            for subj in self.subjects.values():
                subj.save_plots(solution)
