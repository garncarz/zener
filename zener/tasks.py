import os

from celery import Celery, group
from celery.signals import celeryd_after_setup

from .settings import settings

app = Celery(
    broker_url='filesystem://',
    broker_transport_options={
        'data_folder_in': settings.celery_broker_dir,
        'data_folder_out': settings.celery_broker_dir,
    },
    backend=f'file://{settings.celery_results_dir}',
)


@celeryd_after_setup.connect
def load_shared_data(sender, **kwargs):
    from . import optimize
    os.chdir('data')
    app.subjects = optimize.load_subjects()


@app.task
def compute_fitness(subject_label, solution):
    return app.subjects[subject_label].fitness_func(solution)


@app.task
def save_plots(subject_label, solution):
    app.subjects[subject_label].save_plots(solution)
