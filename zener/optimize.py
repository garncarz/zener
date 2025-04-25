import json
import logging

import numpy as np
import pandas as pd
import pygad

from . import model, plots
from .common import change_dir, parse_signal_file
from .settings import settings
from .statsd import statsd


logger = logging.getLogger(__name__)


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
        self.Tp = self.dt * model.Nsamp

        logger.info(f'Testing data loaded: {self.label}')

    def compute_all(self, params):
        return pd.DataFrame(
            model.compute_all(params, self.Rsig, self.Tp, self.dt),
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
            plots.plot_time_series(merged)
            plots.plot_hysteresis(merged)


def load_subjects():
    return {label: GASubject(label, Rsig) for label, Rsig in parse_signal_file().items()}


class GAAll:

    def __init__(self):
        self.subjects = load_subjects()

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
        if settings.use_celery:
            from . import tasks
            fitness_tasks = tasks.group(
                tasks.compute_fitness.s(subj.label, solution.tolist())
                for subj in self.subjects.values()
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
