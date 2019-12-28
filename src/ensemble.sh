# #!/bin/bash
#
# # # Partition the dataset
#
# # python3 generate_dataset.py -D '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/ensemble_dataset' \
# # -T 0.7 \
# # -V 0.15 \
# # -E 0.15 \
# # -N 'ensemble_dataset_partitioned/'
#
# # # train networks on the dataset
#
# # python3 train.py -D '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/ensemble_dataset_partitioned/' \
# # -M 'InceptionV3' \
# # -T \
# # -W '../weights/inception_v3_weights_tf_dim_ordering_tf_kernels.h5' \
# # -O 'sgd' \
# # -o '../weights/InceptionV3/ensemble_weights_inceptionv3.h5'
#
# # python3 train.py -D '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/ensemble_dataset_partitioned/' \
# # -M 'VGG16' \
# # -T \
# # -W '../weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5' \
# # -O 'sgd' \
# # -o '../weights/VGG16/ensemble_weights_vgg16.h5'
#
# # python3 train.py -D '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/ensemble_dataset_partitioned/' \
# # -M 'ResNet50' \
# # -T \
# # -W '../weights/resnet50_weights_tf_dim_ordering_tf_kernels.h5' \
# # -O 'sgd' \
# # -o '../weights/ResNet50/ensemble_weights_resnet50.h5'
#
# # # create adversary dataset
#
# # python3 generate_adversarial_dataset.py -D '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/inceptionv3_dataset_partitioned' \
# # -A 'Ensemble' \
# # -T 0.93 \
# # -X 0.07 \
# # -M 'VGG16'
#
# # python3 generate_adversarial_dataset.py -D '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/inceptionv3_dataset_partitioned' \
# # -A 'Ensemble' \
# # -T 0.93 \
# # -X 0.07 \
# # -M 'InceptionV3'
#
# # python3 generate_adversarial_dataset.py -D '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/inceptionv3_dataset_partitioned' \
# # -A 'Ensemble' \
# # -T 0.93 \
# # -X 0.07 \
# # -M 'ResNet50'
#
# # # create heatmap of the adversary dataset
#
# # move FGSM_InceptionV3
# mv /media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/inceptionv3_dataset_partitioned/FGSM_InceptionV3 /media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/inceptionv3_dataset_partitioned/FGSM_InceptionV3_transfer
# mv /media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/inceptionv3_dataset_partitioned/heatmap_FGSM_InceptionV3 /media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/inceptionv3_dataset_partitioned/heatmap_FGSM_InceptionV3_transfer
# # train inceptionv3 full regular end to edn
# python3 train.py -D '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/inceptionv3_dataset_partitioned/' \
# -M 'InceptionV3' \
# -W '../weights/inception_v3_weights_tf_dim_ordering_tf_kernels.h5' \
# -O 'sgd' \
# -o '../weights/InceptionV3/regular_endtoend_weights_inceptionv3.h5'
# # generate adversary dataset with endtoend
# python3 generate_adversarial_dataset.py -D '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/inceptionv3_dataset_partitioned' \
# -A 'FGSM' \
# -T 0.94 \
# -X 0.06 \
# -M 'InceptionV3' \
# -W '../weights/InceptionV3/regular_endtoend_weights_inceptionv3.h5'
# # generaet heatmap with endtoend
# python3 generate_heatmap_dataset.py -D '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/inceptionv3_dataset_partitioned' \
# -A 'FGSM' \
# -M 'InceptionV3' \
# -W '../weights/InceptionV3/regular_endtoend_weights_inceptionv3.h5'
# # retrain the inceptionv3 heat full
# python3 train.py -D '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/inceptionv3_dataset_partitioned/' \
# -M 'InceptionV3' \
# -W '../weights/inception_v3_weights_tf_dim_ordering_tf_kernels.h5' \
# -O 'sgd' \
# -o '../weights/InceptionV3/heat_endtoend_weights_inceptionv3.h5' \
# -H
#
# # for resnet
# python3 generate_adversarial_dataset.py \
# -D '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/resnet50_dataset_partitioned' \
# -A 'FGSM' \
# -T 0.94 \
# -X 0.06 \
# -M 'ResNet50' \
# -W '../weights/ResNet50/regular_weights_resnet50.h5'
# # generate heatmap with regular
python3 generate_heatmap_dataset.py -D '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/resnet50_dataset_partitioned/' \
-A 'FGSM' \
-M 'ResNet50' \
-W '../weights/ResNet50/regular_weights_resnet50.h5'

# train the transfer model heat
python3 train.py -D '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/resnet50_dataset_partitioned/' \
-M 'ResNet50' \
-T \
-W '../weights/resnet50_weights_tf_dim_ordering_tf_kernels.h5' \
-O 'sgd' \
-o '../weights/ResNet50/heat_weights_resnet50.h5' \
-H

# move the old things
mv /media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/resnet50_dataset_partitioned/FGSM_ResNet50 /media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/resnet50_dataset_partitioned/FGSM_ResNet50_transer
mv /media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/resnet50_dataset_partitioned/heatmap_FGSM_ResNet50 /media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/resnet50_dataset_partitioned/heatmap_FGSM_ResNet50_transer
# train resnet50 full regular end to edn
# python3 train.py -D '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/resnet50_dataset_partitioned/' \
# -M 'ResNet50' \
# -W '../weights/resnet50_weights_tf_dim_ordering_tf_kernels.h5' \
# -O 'sgd' \
# -o '../weights/ResNet50/regular_endtoend_weights_resnet50.h5'

# generate adversary dataset with endtoend
python3 generate_adversarial_dataset.py -D '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/resnet50_dataset_partitioned/' \
-A 'FGSM' \
-T 0.94 \
-X 0.06 \
-M 'ResNet50' \
-W '../weights/ResNet50/regular_endtoend_weights_resnet50.h5'
# generaet heatmap with endtoend
python3 generate_heatmap_dataset.py -D '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/resnet50_dataset_partitioned/' \
-A 'FGSM' \
-M 'ResNet50' \
-W '../weights/ResNet50/regular_endtoend_weights_resnet50.h5'

# retrain the inceptionv3 heat full
python3 train.py -D '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/resnet50_dataset_partitioned/' \
-M 'ResNet50' \
-W '../weights/resnet50_weights_tf_dim_ordering_tf_kernels.h5' \
-O 'sgd' \
-o '../weights/ResNet50/heat_endtoend_weights_resnet50.h5' \
-H
