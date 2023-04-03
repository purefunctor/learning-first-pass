#!/usr/bin/env bash

# train on 10 angles
python3 main.py --mode train --epoch 0 --delta 100 --seed 0 --train-angle 10 --train-vertical 10 --test-angle 10 --test-vertical 10

# test against 20, 50, 100 on the same seed
python3 main.py --mode eval --epoch 0 --delta 50 --seed 0 --train-angle 10 --train-vertical 10 --test-angle 25 --test-vertical 20
python3 main.py --mode eval --epoch 0 --delta 50 --seed 0 --train-angle 10 --train-vertical 10 --test-angle 50 --test-vertical 30
python3 main.py --mode eval --epoch 0 --delta 50 --seed 0 --train-angle 10 --train-vertical 10 --test-angle 100 --test-vertical 40
python3 main.py --mode eval --epoch 0 --delta 50 --seed 0 --train-angle 10 --train-vertical 10 --test-angle 360 --test-vertical 50

# test against 20, 50, 100 on a different seed
python3 main.py --mode eval --epoch 0 --delta 50 --seed 100 --train-angle 10 --train-vertical 10 --test-angle 25 --test-vertical 20
python3 main.py --mode eval --epoch 0 --delta 50 --seed 100 --train-angle 10 --train-vertical 10 --test-angle 50 --test-vertical 30
python3 main.py --mode eval --epoch 0 --delta 50 --seed 100 --train-angle 10 --train-vertical 10 --test-angle 100 --test-vertical 40
python3 main.py --mode eval --epoch 0 --delta 50 --seed 100 --train-angle 10 --train-vertical 10 --test-angle 360 --test-vertical 50
