#!/bin/bash
for SEED in 1 2 3 4 5
do
    PYTHONPATH=. python sac/train_example.py --seed=$SEED;
done