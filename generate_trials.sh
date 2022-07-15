#!/bin/bash
#CUDA_VISIBLE_DEVICES=0 python generate_trials.py --model_type lstm
#CUDA_VISIBLE_DEVICES=0 python generate_trials.py --model_type gru
#CUDA_VISIBLE_DEVICES=1 python generate_trials.py --model_type asrnn
CUDA_VISIBLE_DEVICES=1 python generate_trials.py --model_type cornn

