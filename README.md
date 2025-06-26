# Optimizing parameters for the Zener function used to describe rock stress/strain process

Needed: Python 3

Installing dependencies: `pip install -r requirements.txt`

The main entry point is `./zener.py`, it also prints help.

To generate testing data, run `./zener.py --generate`.

To run optimization, run `./zener.py --optimize`.
Optimalizer state is saved after each generation (iteration) in `ga_instance.pkl` and is reloaded after restart.
Delete it if you want to start from scratch.

To override default arguments, see `settings.py`.
You can override them using env. variables or `.env` file.

For example if you want to run optimization with more generations, run `num_generations=100 ./zener.py --optimize`.

Using Jupyter Notebook: `jupyter notebook`

Tests are run automatically by GitHub Actions, but they can also be run locally manually: `pytest`

Compiling the original Fortran code (by doc. Ivan Janeček): `gfortran -o fzener ZenerNLElast-INI-input.f90`.
Then it is runnable as `./fzener`.


## MetaCentrum

Assuming your project is located under `~/git/zener`, otherwise change it in `zener.sh`.

To schedule a job: `qsub zener.sh` (do not run it directly!)

In case of a stale K8s session, run `kinit`.
