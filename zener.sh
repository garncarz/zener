#!/usr/bin/env bash
#PBS -N zener_optimization_job
#PBS -l select=3:ncpus=5:mem=8gb:scratch_shared=2gb
#PBS -l walltime=24:00:00
# ^^^ select = number of machines, walltime = max time

# As a preparation, run this on the frontend:
# module add python
# python -m venv ~/venv
# source ~/venv/bin/activate
# pip install -r requirements.txt

export SRC_DIR=/storage/brno2/home/$(whoami)/git/zener
export DATADIR=$SRC_DIR
export VENV_DIR=/storage/brno2/home/$(whoami)/venv

module add python
source $VENV_DIR/bin/activate

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set"; exit 1; }

export celery_broker_dir=$SCRATCHDIR/celery_broker
export celery_results_dir=$SCRATCHDIR/celery_results

echo "[WORKER] Starting Celery worker on `hostname -f`"

cd $SRC_DIR

celery -A zener.tasks worker --loglevel=INFO &
WORKER_PID=$!

if [ "$PBS_NODENUM" == "0" ]; then
    # ./zener.py --generate || { echo >&2 "Generating testing data erred (code $?)"; exit 4; }
    use_celery=1 ./zener.py --optimize || { echo >&2 "Optimizing erred (code $?)"; exit 5; }

    touch "$SCRATCHDIR/STOP_WORKERS"
    kill $WORKER_PID

    sleep 20  # so other workers see STOP_WORKERS
    clean_scratch

else
    while [[ ! -f "$SCRATCHDIR/STOP_WORKERS" ]]; do
        sleep 10
    done

    echo "[WORKER] Stop signal received, killing Celery worker..."
    kill $WORKER_PID
fi
