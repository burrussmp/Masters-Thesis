#!/bin/bash

python3 test.py -D '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/vgg16_dataset_partitioned/heatmap_FGSM_VGG16' \
-M 'VGG16' \
-W '../weights/VGG16/heat_weights_vgg16.h5'

python3 test.py -D '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/vgg16_dataset_partitioned/heatmap_test' \
-M 'VGG16' \
-W '../weights/VGG16/heat_weights_vgg16.h5'

python3 test.py -D '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/daveii_dataset_partitioned/heatmap_test' \
-M 'DaveII' \
-W '../weights/DaveII/heat_endtoend_weights_daveii.h5' \
--MSE

python3 test.py -D '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/resnet50_dataset_partitioned/heatmap_test' \
-M 'ResNet50' \
-W '../weights/ResNet50/heat_endtoend_weights_resnet50.h5'
