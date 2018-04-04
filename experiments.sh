#!bin/bash

export TF_CPP_MIN_LOG_LEVEL=3
GPU=1

CUDA_VISIBLE_DEVICES=$GPU python train.py --snr=10.0
CUDA_VISIBLE_DEVICES=$GPU python train.py --snr=7.0
CUDA_VISIBLE_DEVICES=$GPU python train.py --snr=5.0
CUDA_VISIBLE_DEVICES=$GPU python train.py --snr=3.0
CUDA_VISIBLE_DEVICES=$GPU python train.py --snr=2.0
CUDA_VISIBLE_DEVICES=$GPU python train.py --snr=1.0
CUDA_VISIBLE_DEVICES=$GPU python train.py --snr=0.5
CUDA_VISIBLE_DEVICES=$GPU python train.py --snr=0.2
CUDA_VISIBLE_DEVICES=$GPU python train.py --snr=0.1


