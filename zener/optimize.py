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


def transform_params(params):
    """Transform optimization parameters to physical model parameters

    Currently transforms:
    - Em_F values from exponent x to 10^x
    """
    params = params.copy()

    for i in range(4, 4 + settings.n_f):
        params[i] = 10 ** params[i]

    return params


def dynamic_population_size(generation, num_generations, initial_size, final_size):
    """Return the population size for a generation on a linear schedule."""
    progress = min(generation / max(num_generations, 1), 1.0)
    return max(1, round(initial_size + progress * (final_size - initial_size)))


class GASubject:

    def __init__(self, label):
        self.label = label
        self.load_testing_data()

    def load_testing_data(self):
        try:
            df_target = pd.read_csv(f'{self.label}/ZenerModel.dat', sep='\s+')
        except FileNotFoundError:
            df_target = self.load_real_data_txt()

        df_target['time'] = pd.to_numeric(df_target['time'], errors='coerce')
        df_target = df_target.dropna()
        self.df_target_sorted = df_target.sort_values('time')

        with open(f'{self.label}/input.json', 'r') as f:
            input_data = json.load(f)
            self.Tp = input_data['Tp']

            if 'Rs' in input_data:
                self.Rs = input_data['Rs']
            elif 'R' in input_data and 'Rsig' in input_data:
                R = input_data['R']
                Rsig = input_data['Rsig']
                self.Rs = [float(r) * R for r in Rsig]
                logger.info(f'Calculated Rs from R={R} and Rsig values')
            else:
                raise ValueError(f"Required data 'Rs' or both 'R' and 'Rsig' not found in {self.label}/input.json")

        self.dt = self.df_target_sorted.iloc[1].time - self.df_target_sorted.iloc[0].time

        logger.info(f'Testing data loaded: {self.label}')
        self.check_testing_data()

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
            model.compute_all(params, self.Rs, self.Tp, self.dt)
        )

    def compute_and_merge(self, params, transform=True):
        params = transform_params(params) if transform else params
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

        merged = merged.dropna()

        return merged

    def fitness_func(self, solution):
        merged = self.compute_and_merge(solution)

        # match_ratio = len(merged) / len(self.df_target_sorted)
        # if match_ratio < 0.95:
        #     return -1e9

        error = 0.0
        for col in ['TotStrain']:
            diff = merged[f'{col}_model'] - merged[f'{col}_target']
            error += np.mean(diff ** 2)

        return -error

    def save_plots(self, solution):
        # Accepts either a dict (from solution.json) or a list
        if isinstance(solution, dict):
            Tau_s = solution['Tau_s']
            EmNL = solution['EmNL']
            Alpha = solution['Alpha']
            Beta = solution['Beta']
            Em_F = solution['Em_F']
            Tau_e_F = solution['Tau_e_F']
            solution = [Tau_s, EmNL, Alpha, Beta] + Em_F + Tau_e_F

        with change_dir(self.label):
            merged = self.compute_and_merge(solution, transform=False)
            plots.plot_time_series(merged)
            plots.plot_hysteresis(merged)

    def check_testing_data(self):
        total_Tp = sum(self.Tp)
        last_time = self.df_target_sorted['time'].iloc[-1]
        if not np.isclose(total_Tp, last_time, rtol=0.01):
            logger.warning(
                f'Sum of Tp ({total_Tp}) does not match last time in data ({last_time}).'
            )


def load_subjects(just_labels=False):
    return {
        label: GASubject(label) if not just_labels else 'Use celery'
        for label in parse_signal_file().keys()
    }


class GAAll:

    def __init__(self, ga_options=None):
        ga_options = ga_options or {}
        self.persist_state = ga_options.get('persist_state', True)
        self.subjects = load_subjects(just_labels=settings.use_celery)
        initial_population = (
            settings.dynamic_population_initial
            if settings.dynamic_population
            else ga_options.get('sol_per_pop', settings.dynamic_population_final)
        )

        loaded_ga = None if not ga_options.get('load_state', True) else self.load_ga()
        self.ga = loaded_ga or pygad.GA(
            num_generations=settings.num_generations,
            stop_criteria=[f'reach_{settings.stop_fitness}', f'saturate_{settings.stop_saturation}'],
            num_parents_mating=ga_options.get('num_parents_mating', 5),
            parent_selection_type=ga_options.get('parent_selection_type', 'nsga2'),
            fitness_func=self.fitness_func,
            sol_per_pop=initial_population,
            num_genes=4 + 2 * settings.n_f,
            mutation_percent_genes=ga_options.get('mutation_percent_genes', 30),
            mutation_by_replacement=ga_options.get('mutation_by_replacement', True),
            keep_elitism=ga_options.get('keep_elitism', 1),
            crossover_type=ga_options.get('crossover_type', 'single_point'),
            gene_space = [
                {'low': 0.1, 'high': 1e8},    # Tau_s
                {'low': 1e-4, 'high': 1e6},    # EmNL
                {'low': 1e-4, 'high': 200},    # Alpha
                {'low': 1e-4, 'high': 10.0},    # Beta
                *[{'low': -1.0, 'high': 12.0} for _ in range(settings.n_f)],  # Em_F - log scale
                *[{'low': 0.1, 'high': 1e8} for _ in range(settings.n_f)],  # Tau_e_F
            ],
            save_best_solutions=ga_options.get('save_best_solutions', True),
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
        Tau_e_F = solution[4 + settings.n_f:4 + 2 * settings.n_f]
        return all(Tau_e_F[i] <= Tau_e_F[i + 1] for i in range(len(Tau_e_F) - 1))

    def fitness_func(self, ga_instance, solution, solution_idx):
        # if not self._is_tau_increasing(solution):
        #     return [-1e9] * len(self.subjects)

        if settings.use_celery:
            fitness_tasks = tasks.group(
                tasks.compute_fitness.s(subj_label, solution.tolist())
                for subj_label in self.subjects.keys()
            )
            return fitness_tasks.apply_async().get()
        return [subj.fitness_func(solution) for subj in self.subjects.values()]

    def on_generation(self, ga):
        if settings.dynamic_population:
            self._resize_population(ga)

        logger.info(
            f'Generation {ga.generations_completed}/{settings.num_generations}, '
            f'fitness: {ga.last_generation_fitness}'
        )
        if self.persist_state:
            self.ga.save(settings.ga_filename)

        with statsd.batch():
            statsd.gauge('generation.fitness.max_min', np.max(np.min(ga.last_generation_fitness, axis=1)))
            statsd.gauge('generation.fitness.mean', np.mean(np.mean(ga.last_generation_fitness, axis=1)))
            statsd.gauge('generation', ga.generations_completed)

            best_sol = ga.best_solutions[-1]
            best_sol = transform_params(best_sol)
            Tau_s, EmNL, Alpha, Beta = best_sol[:4]

            statsd.gauge('generation.model.Tau_s', Tau_s)
            statsd.gauge('generation.model.EmNL', EmNL)
            statsd.gauge('generation.model.Alpha', Alpha)
            statsd.gauge('generation.model.Beta', Beta)

            if settings.n_f:
                Em_F = best_sol[4:4 + settings.n_f]
                Tau_e_F = best_sol[4 + settings.n_f:4 + 2 * settings.n_f]
                Em, Tau_e = Em_F[0], Tau_e_F[0]

                statsd.gauge('generation.model.Em', Em)
                statsd.gauge('generation.model.Tau_e', Tau_e)

                for i in range(1, settings.n_f):
                    statsd.gauge(f'generation.model.Em_F_{i}', Em_F[i])
                    statsd.gauge(f'generation.model.Tau_e_F_{i}', Tau_e_F[i])

    def _resize_population(self, ga):
        population_size = dynamic_population_size(
            ga.generations_completed,
            settings.num_generations,
            settings.dynamic_population_initial,
            settings.dynamic_population_final,
        )

        if population_size == ga.sol_per_pop:
            return

        old_population = ga.population
        old_fitness = ga.last_generation_fitness
        ga.pop_size = (population_size, ga.num_genes)
        ga.sol_per_pop = population_size
        ga.num_parents_mating = min(ga.num_parents_mating, population_size)
        if ga.keep_elitism:
            ga.num_offspring = population_size - ga.keep_elitism
        elif ga.keep_parents == -1:
            ga.num_offspring = population_size - ga.num_parents_mating
        else:
            ga.num_offspring = population_size - ga.keep_parents

        ga.initialize_population(
            ga.allow_duplicate_genes,
            ga.gene_type,
            ga.gene_constraint,
        )

        keep_count = min(len(old_population), population_size)
        if keep_count:
            scores = np.asarray(old_fitness)
            if scores.ndim > 1:
                scores = np.sum(scores, axis=1)
            best_indices = np.argsort(scores)[-keep_count:]
            ga.population[:keep_count] = old_population[best_indices]

        ga.last_generation_parents = None
        ga.previous_generation_fitness = None
        ga.last_generation_fitness = ga.cal_pop_fitness()

    def run(self):
        self.ga.run()

    def save_result(self):
        solution, solution_fitness, _ = self.ga.best_solution()
        solution = transform_params(solution)
        logger.info(f'Best solution: {solution}, fitness: {solution_fitness}')

        Tau_s, EmNL, Alpha, Beta = solution[:4]
        Em_F = solution[4:4 + settings.n_f]
        Tau_e_F = solution[4 + settings.n_f:4 + 2 * settings.n_f]

        with open('solution.json', 'w') as f:
            json.dump({
                'Tau_s': Tau_s,
                'EmNL': EmNL,
                'Alpha': Alpha,
                'Beta': Beta,
                'Em_F': Em_F.tolist(),
                'Tau_e_F': Tau_e_F.tolist(),
                'generations': self.ga.generations_completed,
                'fitness': solution_fitness.tolist(),
            }, f, indent=4)

        plt_fitness = self.ga.plot_fitness()
        plt_fitness.savefig('fitness.png')

        self.save_plots(solution)

    def save_plots(self, solution=None):
        if solution is None:
            with open('solution.json', 'r') as f:
                solution = json.load(f)
        if isinstance(solution, np.ndarray):
            solution = solution.tolist()
        if settings.use_celery:
            plots_tasks = tasks.group(
                tasks.save_plots.s(subj_label, solution)
                for subj_label in self.subjects.keys()
            )
            plots_tasks.apply_async().join()
        else:
            for subj in self.subjects.values():
                subj.save_plots(solution)
