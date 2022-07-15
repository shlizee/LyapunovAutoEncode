#!/bin/bash

python AE_train.py --model_type rnn --task_type SMNIST --latent_size 32 --alphas [200,200,300,400] --lr 1e-4