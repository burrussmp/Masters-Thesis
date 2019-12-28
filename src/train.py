import os
os.environ["CUDA_VISIBLE_DEVICES"]="1" # second gpu
os.environ["CUDA_VISIBLE_DEVICES"]="2" # second gpu
os.environ["CUDA_VISIBLE_DEVICES"]="3" # second gpu
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, History
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import math
import cv2
from keras.applications.inception_v3 import InceptionV3
import argparse
from generate_model import generate_model
from datetime import datetime
import keras
import tensorflow as tf
from generate_model import custom_accuracy
modelType = None
def main(args):
    global modelType
    if (args.dir[-1] == '/'): args.dir = args.dir[:-1] # just some preprocessing of input to mitigate a mistake
    numClasses = len(os.listdir(os.path.join(args.dir,'train')))
    model_generator = generate_model(args.model,args.isHeatmap)
    model = model_generator.initialize(args.weights,classes=numClasses,useTransfer=args.isTransfer,optimizer=args.optimizer)
    if (not os.path.exists('../weights/')):
        os.mkdir('../weights/')
    parentDir = os.path.join('../weights/',args.model)
    if (not os.path.exists(parentDir)):
        os.mkdir(parentDir)

    assert os.path.exists(args.weights_path_to_save.replace(os.path.basename(args.weights_path_to_save),'')), \
        'The path directory where you would like to save the weights does not exist!'

    if (args.weights_path_to_save == None):
        name = datetime.utcnow().strftime('%m_%d_%Y_%M_%S_%f')+'.h5'
        pathToSave = os.path.join(parentDir,name)
    else:
        pathToSave = args.weights_path_to_save

    checkpoint = ModelCheckpoint(pathToSave, monitor='custom_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)

    # function to preprocess an image
    #preprocess_specific = model_generator.getPreProcessingFunction()
    modelType = model_generator.getName()

    def preprocess(x):
        return model_generator.preprocess(x)

    # path to data
    if (args.isHeatmap):
        train_data_dir = os.path.join(args.dir,'heatmap_train')
        validation_data_dir = os.path.join(args.dir,'heatmap_val')
    else:
        train_data_dir = os.path.join(args.dir,'train')
        validation_data_dir = os.path.join(args.dir,'val')

    assert os.path.exists(train_data_dir), \
        'Training doesn\'t exist'


    assert os.path.exists(validation_data_dir), \
        'Validation doesn\'t exist'

    nb_train_samples = sum([len(os.listdir(os.path.join(train_data_dir,folder))) for folder in os.listdir(train_data_dir)])
    nb_validation_samples = sum([len(os.listdir(os.path.join(validation_data_dir,folder))) for folder in os.listdir(validation_data_dir)])

    # training parameters
    batch_size = 16
    epochs = 100
    history = History()
    # should we allow horizontal flip augmentation? Doesn't work for track data
    horizontal_flip = True
    zoom_range=0.3
    rotation_range = 30
    if (model_generator.getName()=='DaveII'):
        horizontal_flip = False
        zoom_range = 0.0 # if you zoom to far, it may be a different turn
        rotation_range = 0 # rotation should be zero on a car otherwise turn changes
    # Initiate the train and test generators with data Augumentation
    train_datagen = ImageDataGenerator(
        #rescale = 1./255,
        horizontal_flip = horizontal_flip,
        fill_mode = "nearest",
        zoom_range = zoom_range,
        width_shift_range = 0.3,
        height_shift_range=0.3,
        rotation_range=rotation_range,
        preprocessing_function=preprocess)

    test_datagen = ImageDataGenerator(
        #rescale = 1./255,
        horizontal_flip = horizontal_flip,
        fill_mode = "nearest",
        zoom_range = zoom_range,
        width_shift_range = 0.3,
        height_shift_range=0.3,
        rotation_range=rotation_range,
        preprocessing_function=preprocess)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = model_generator.getInputSize(),
        batch_size = batch_size,
        class_mode = "categorical",
        shuffle=True)

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size = model_generator.getInputSize(),
        class_mode = "categorical")
    # Train the model
    model.fit_generator(
        train_generator,
        steps_per_epoch = math.ceil(nb_train_samples/batch_size),
        epochs = epochs,
        validation_data = validation_generator,
        validation_steps = math.ceil(nb_validation_samples/batch_size),
        callbacks = [checkpoint, history])


if __name__ == "__main__":
    config = tf.ConfigProto( device_count = {'GPU': 4 , 'CPU': 20} )
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)
    parser = argparse.ArgumentParser(description='Train a neural net!')

    parser.add_argument('--base_dir','-D', metavar='D', type=str,required=True,
                    help='Directory of partitioned data',dest='dir')

    parser.add_argument('--model', '-M', metavar='M', type=str,required=True,
                    help='Model to load to evaluate on the data set',dest='model')

    parser.add_argument('--model_weights', '-W', metavar='W', type=str,required=False,
                    help='Model weights to initialize the model.',dest='weights')

    parser.add_argument('--useTransfer', '-T',action='store_true',
                    help='Specify if only the fully connected layers should be re-trained',dest='isTransfer')

    parser.add_argument('--optimizer', '-O', metavar='O', type=str,required=False,
                    help='Valid optimizers: adam,sgd',dest='optimizer')

    parser.add_argument('--makeHeatmapDataset', '-H',action='store_true',
                    help='Specify whether to look for heatmap_train and heatmap_val in base directory (-D or --base_dir)',dest='isHeatmap')

    parser.add_argument('--saveWeightsHere', '-o', metavar='o', type=str,required=False,
                    help='Where to save the heatmap',dest='weights_path_to_save')
    args = parser.parse_args()
    main(args)

# python3 train.py -D '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/Dataset_DaveII' -M DaveII -T True
#python3 train.py -D '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/inceptionv3_dataset_partitioned/' -M 'InceptionV3' -T -W '../weights/inception_v3_weights_tf_dim_ordering_tf_kernels.h5' -O 'sgd'
#python3 train.py -D '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/resnet50_dataset_partitioned/' -M 'ResNet50' -T -W '../weights/resnet50_weights_tf_dim_ordering_tf_kernels.h5' -O 'sgd'
