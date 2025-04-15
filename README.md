# Optimizing parameters for the Zener function used to describe rock stress/strain process

Installing dependencies: `pip install -r requirements.txt`

To generate testing data, simply run `./generate.py`.

To run optimization, run `./optimize.py`.

To override default arguments, see `settings.py`.

Using Jupyter Notebook: `jupyter notebook`

Compiling the original Fortran code (by doc. Ivan Janeček): `gfortran -o zener ZenerNLElast-INI-input.f90`. Then it is runnable as `./zener`.
