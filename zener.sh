#!/usr/bin/env bash
#PBS -N zener_optimization_job
#PBS -l select=1:ncpus=20:mem=8gb:scratch_local=2gb
#PBS -l walltime=1:00:00
# ^^^ select = number of machines, walltime = max time

export DATADIR=/storage/brno2/home/$(whoami)/git/zener
export PYTHONUSERBASE=/storage/brno2/home/$(whoami)/my_pip_libs
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.7/site-packages:$PYTHONPATH

mkdir -p $PYTHONUSERBASE

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

module add py-pip

test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set"; exit 1; }

cp -rv $DATADIR $SCRATCHDIR || { echo >&2 "Error while copying input files"; exit 2; }

cd $SCRATCHDIR/zener

pip install --user -r requirements.txt || { echo >&2 "Pip installation erred (code $?)"; exit 3; }
./zener.py --generate || { echo >&2 "Generating testing data erred (code $?)"; exit 4; }
./zener.py --optimize || { echo >&2 "Optimizing erred (code $?)"; exit 5; }

cp -rv data $DATADIR/ || { echo >&2 "Result files copying failed (code $?)"; exit 6; }

clean_scratch
