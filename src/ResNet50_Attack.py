from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization,Lambda
from keras.layers import Conv2D, MaxPooling2D,Input,AveragePooling2D
import os
from keras.models import load_model
from art.classifiers import KerasClassifier as DefaultKerasClassifier
from art.attacks import FastGradientMethod,DeepFool,CarliniL2Method,BasicIterativeMethod,ProjectedGradientDescent
from art.utils import load_mnist
import innvestigate
import innvestigate.utils
import numpy as np
from keras import losses
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, History
import cv2
from sklearn.preprocessing import OneHotEncoder
import keras.backend as K
import tensorflow as tf
from keras.models import Model
from keras import regularizers
from keras.regularizers import l2
from matplotlib import pyplot as plt

from models.ResNet50Model import ResNet50Model
from AdversarialAttacks import CarliniWagnerAttack,ProjectedGradientDescentAttack,FGSMAttack,DeepFoolAttack,BasicIterativeMethodAttack
from AdversarialAttacks import HistogramOfPredictionConfidence,ConfusionMatrix
from keras.applications.resnet50 import preprocess_input
# os.environ["CUDA_VISIBLE_DEVICES"]="1" # second gpu
# os.environ["CUDA_VISIBLE_DEVICES"]="2" # second gpu
# os.environ["CUDA_VISIBLE_DEVICES"]="3" # second gpu
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="2"

def preprocess(x):
    x = preprocess_input(x)
    return x

def loadData(baseDir='./vgg16_dataset_10_partitioned',dataType='train'):
    assert dataType in ['train','test','val'],\
        print('Not a valid type, must be train, test, or val')
    train_data_dir = os.path.join(baseDir,dataType)
    if (dataType=='test'):
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess)
        data_generator = datagen.flow_from_directory(
            train_data_dir,
            target_size = (224,224),
            batch_size = 32,
            class_mode = "categorical",
            shuffle=True)
        data_generator.batch_size = data_generator.samples
    else:
        datagen = ImageDataGenerator(
            fill_mode = "nearest",
            zoom_range = 0.3,
            width_shift_range = 0.3,
            height_shift_range=0.3,
            rotation_range=30,
            horizontal_flip=True,
            preprocessing_function=preprocess)

        data_generator = datagen.flow_from_directory(
            train_data_dir,
            target_size = (224,224),
            batch_size = 32,
            class_mode = "categorical",
            shuffle=True)
    return data_generator

train_data_generator = loadData(dataType='train')
validation_data_generator = loadData(dataType='val')

baseDir ='/content/drive/My Drive/Colab Notebooks/ResNet50Weights'

# SOFTMAX MODEL CLEAN
softmax_clean = ResNet50Model(weights=None,RBF=False)
softmax_clean.model.summary()
#softmax_clean.load(weights=os.path.join(baseDir,'softmax_clean.h5'))
softmax_clean.train(train_data_generator,validation_data_generator,saveTo=os.path.join(baseDir,'softmax_clean.h5'),epochs=100)
print('Loaded softmax clean model...')

# ANOMALY DETECTOR CLEAN
anomaly_clean = ResNet50Model(weights=None,anomalyDetector=True)
anomaly_clean.model.summary()
#anomaly_clean.load(weights=os.path.join(baseDir,'anomaly_clean.h5'))
anomaly_clean.train(train_data_generator,validation_data_generator,saveTo=os.path.join(baseDir,'anomaly_clean.h5'),epochs=100)
print('loaded anomaly clean model...')

test_data_generator = loadData(dataType='test')
x_test,y_test = test_data_generator.next()
print('Number of test data',y_test.shape[0])

evaluate = True
confusionMatrices = True
histograms = True
if (evaluate):
    print('SOFTMAX CLEAN on test')
    softmax_clean.evaluate(x_test,y_test)
    print('SOFTMAX CLEAN on backdoor')
    softmax_clean.evaluate(xadv,yadv)
    print('\n')
    print('RBF CLEAN on test')
    rbf_clean.evaluate(x_test,y_test)
    print('RBF CLEAN on backdoor')
    rbf_clean.evaluate(xadv,yadv)
    print('\n')
    print('ANOMALY CLEAN on test')
    anomaly_clean.evaluate(x_test,y_test)
    print('ANOMALY CLEAN on backdoor')
    anomaly_clean.evaluate(xadv,yadv)
    print('\n')

if (confusionMatrices):
    n_test = str(y_test.shape[0])
    ConfusionMatrix(predictions=softmax_clean.predict(x_test),
        Y=y_test,
        title='VGG16 Softmax Clean (n='+n_test+')')
    ConfusionMatrix(predictions=anomaly_clean.predict(x_test),
        Y=y_test,
        title='VGG16 Anomaly Detector Clean (n='+n_test+')')

if (histograms):
    HistogramOfPredictionConfidence(P1=softmax_clean.predict(x_test),
        Y1=y_test,
        P2=softmax_clean.predict(x_test),
        Y2=y_test,
        title='VGG16 SoftMax Test Confidence',
        numGraphs=1)
    HistogramOfPredictionConfidence(P1=anomaly_clean.predict(x_test),
        Y1=y_test,
        P2=anomaly_clean.predict(x_test),
        Y2=y_test,
        title='VGG16 Anomaly Detector Test Confidence',
        numGraphs=1)
plt.show()

# Set attacks true or false
FGSM = True
DeepFool = True
IFGSM = True
CarliniWagner = True
PGD = True

attacks=[]
if (FGSM):
    attacks.append({
        'name':'fgsm',
        'function': FGSMAttack})
if (DeepFool):
    attacks.append({
        'name':'deepfool',
        'function': DeepFoolAttack})
if (IFGSM):
    attacks.append({
        'name':'ifgsm',
        'function': BasicIterativeMethodAttack})
if (CarliniWagner):
    attacks.append({
        'name':'c&w',
        'function': CarliniWagnerAttack})
if (PGD):
    attacks.append({
        'name':'pgd',
        'function': ProjectedGradientDescentAttack})

print('Performing the following attacks...')
for attack in attacks:
    print(attack['name'])


for attack in attacks:
    attackName = attack['name']
    print('Evaluating Attack:',attackName)

    attack_function = attack['function']
    print('Creating attack for softmax model...')
    xadv = attack_function(model=softmax_clean.model,
        X=x_test,
        path=os.path.join(baseDir,'attacks',attackName,'softmax_clean_attack.npy'))
    print('Softmax model on attack ', attackName,'...')
    softmax_clean.evaluate(xadv,y_test)
    print('\n')

    print('Creating attack for anomaly detector...')
    xadv = attack_function(model=anomaly.model,
        X=x_test,
        path=os.path.join(baseDir,'attacks',attackName,'anomaly_clean_attack.npy'))
    print('Anomaly Detector on attack ', attackName,'...')
    anomaly_clean.evaluate(xadv,y_test)
    print('\n') 
