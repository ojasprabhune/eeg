#!/bin/bash
# Runs the CER-vs-accuracy sweep with fixed hyperparameters so results stay
# consistent across the multiple (resumed) invocations needed to finish all
# runs within the shell time limit. Skips already-completed (experiment,
# accuracy) pairs via figures/results.json, and writes the plots once every
# experiment has a full accuracy sweep.
cd /Users/ojasprabhune/Documents/research/NORA/eeg || exit 1
export PYTHONPATH="$PWD"
export SWEEP_DEVICE=mps
export SWEEP_K=1200
export SWEEP_EPOCHS=450
export SWEEP_CKPT_DIR=/Users/ojasprabhune/Documents/research/NORA/ckpts/lm
exec .venv/bin/python -u scripts/language_model/experiments/run_experiments.py "$@"
