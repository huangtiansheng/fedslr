#!/bin/bash


echo 'Calling scripts!'

# install package (skip it if they already have been install)
# pip install torch
# pip install numpy
# pip install torchvision

# Caution!  It is possible that run 4 tasks in parallel would cause out-of-memory! Please comment out some runs if the GPU cannot afford that many runs.
cd ..
# run iid fedSLR
nohup python train.py  --method FedSLR --dataset CIFAR100 --gpu 0 >IIDFedSLR.out  &
sleep 10
# run iid fedAvg
nohup python train.py  --method FedAvg --dataset CIFAR100 --gpu 0 >IIDFedAvg.out &
sleep 10
# run non-iid fedLite
nohup python train.py  --non_iid --method FedSLR --dataset CIFAR100 --gpu 0 >NonIIDFedSLR.out   &
sleep 10
# run non-iid fedAvg
nohup python train.py  --non_iid  --method FedAvg --dataset CIFAR100 --gpu 0 >NonIIDFedAvg.out  &
