# Optimizing parameters for the Zener function used to describe rock stress/strain process

Installing dependencies: `pip install -r requirements.txt`

To generate testing data, simply run `./generate.py`.

To run optimization, run `./optimize.py`.
Optimalizer state is saved after each generation (iteration) in `ga_instance.pkl` and is reloaded after restart.
Delete it if you want to start from scratch.

To override default arguments, see `settings.py`.
You can override them using env. variables or `.env` file.

For example if you want to run optimization with more generations, run `num_generations=100 ./optimize.py`.

Using Jupyter Notebook: `jupyter notebook`

Compiling the original Fortran code (by doc. Ivan Janeček): `gfortran -o zener ZenerNLElast-INI-input.f90`.
Then it is runnable as `./zener`.
