#!/bin/bash

# create dataset partition

python3 generate_dataset.py -D '/media/burrussmp/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/data_set_1' \
-T 0.7 \
-V 0.15 \
-E .15 \
-N 'vgg16_dataset_'