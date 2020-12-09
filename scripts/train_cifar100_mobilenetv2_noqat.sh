#!/bin/sh
python train.py --epochs 200 --optimizer SGD --lr 0.1 --compress schedule-cifar100-mobilenetv2.yaml --model ai87netmobilenetv2cifar100 --dataset CIFAR100 --device MAX78000 --batch-size 128 --print-freq 100 --validation-split 0 --use-bias --qat-policy=None "$@"
