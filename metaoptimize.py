#!/usr/bin/env python3

import json
import logging
import os
from pathlib import Path

import click
import numpy as np
import optuna

from zener.optimize import GAAll
from zener.settings import settings


logger = logging.getLogger(__name__)


def objective(trial, generations):
    """Run one isolated GA trial and return its worst-subject fitness."""
    population = trial.suggest_int('sol_per_pop', 20, 100, step=10)
    options = {
        'load_state': False,
        'persist_state': False,
        'sol_per_pop': population,
        'num_parents_mating': trial.suggest_int(
            'num_parents_mating', 2, min(10, population)
        ),
        'mutation_percent_genes': trial.suggest_int(
            'mutation_percent_genes', 5, 50, step=5
        ),
        'mutation_by_replacement': trial.suggest_categorical(
            'mutation_by_replacement', [False, True]
        ),
        'keep_elitism': trial.suggest_int('keep_elitism', 0, 2),
        'crossover_type': trial.suggest_categorical(
            'crossover_type', ['single_point', 'two_points', 'uniform']
        ),
    }

    ga_all = GAAll(options)
    ga_all.ga.random_seed = trial.number
    ga_all.run()

    fitness = np.asarray(ga_all.ga.best_solution()[1], dtype=float)
    score = float(np.min(fitness))
    trial.set_user_attr('fitness_by_subject', fitness.tolist())
    trial.set_user_attr('generations_completed', ga_all.ga.generations_completed)
    return score


def save_report(study, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(output_dir / 'trials.csv', index=False)

    with (output_dir / 'best_params.json').open('w') as report_file:
        json.dump(
            {
                'best_value': study.best_value,
                'best_params': study.best_params,
                'number_of_trials': len(study.trials),
            },
            report_file,
            indent=4,
        )

    from optuna.visualization.matplotlib import (
        plot_optimization_history,
        plot_param_importances,
    )

    plot_optimization_history(study).figure.savefig(
        output_dir / 'optimization_history.png', bbox_inches='tight'
    )
    plot_param_importances(study).figure.savefig(
        output_dir / 'parameter_importances.png', bbox_inches='tight'
    )


@click.command()
@click.option('--trials', default=20, show_default=True, type=click.IntRange(min=1))
@click.option('--generations', default=3, show_default=True, type=click.IntRange(min=1))
@click.option('--output-dir', default='metaoptimize', show_default=True, type=click.Path())
@click.option('--study-name', default='zener-meta', show_default=True)
@click.option('--seed', default=42, show_default=True, type=int)
def main(trials, generations, output_dir, study_name, seed):
    """Tune PyGAD settings with Optuna and write a report."""
    os.chdir('data')
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    storage = f'sqlite:///{(output_path / "study.db").resolve()}'
    settings.num_generations = generations
    settings.dynamic_population = False

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction='maximize',
        sampler=sampler,
    )
    study.optimize(lambda trial: objective(trial, generations), n_trials=trials)
    save_report(study, output_path)
    click.echo(f'Best fitness: {study.best_value}')
    click.echo(f'Best parameters: {study.best_params}')
    click.echo(f'Report: {output_path.resolve()}')


if __name__ == '__main__':
    main()