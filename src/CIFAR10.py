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

from models.ResNetV1 import ResNetV1
from AdversarialAttacks import CarliniWagnerAttack,ProjectedGradientDescentAttack,FGSMAttack,DeepFoolAttack,BasicIterativeMethodAttack
from AdversarialAttacks import PoisonCIFAR10,HistogramOfPredictionConfidence,ConfusionMatrix,cleanData

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# preprocess the data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

#baseDir = '/media/burrussmp/99e21975-0750-47a1-a665-b2522e4753a6/weights/CIFAR10'
baseDir = "/content/drive/My Drive/Colab Notebooks/Cifar10Weights"

x_train_poison,y_train_poison,poisoned_idx = PoisonCIFAR10(X=x_train,
                                                Y = y_train,
                                                p=0.005)
x_train_backdoor = x_train_poison[poisoned_idx]
y_train_backdoor = y_train_poison[poisoned_idx]

x_test_poison,y_test_poison,poisoned_idx = PoisonCIFAR10(X=x_test,
                                                Y = y_test,
                                                p=0.1)
x_backdoor = x_test_poison[poisoned_idx]
y_backdoor = y_test_poison[poisoned_idx]
labels = np.argmax(y_backdoor,axis=1)
y_true = labels
y_true = (y_true-1)%10
y_true = keras.utils.to_categorical(y_true, 10)

# SOFTMAX MODEL CLEAN
softmax_clean = ResNetV1(RBF=False)
softmax_clean.load(weights=os.path.join(baseDir,'softmax_clean.h5'))
#softmax_clean.train(x_train,y_train,saveTo=os.path.join(baseDir,'softmax_clean.h5'),epochs=100)
print('loaded 1')

# SOFTMAX MODEL POISON
softmax_poison = ResNetV1(RBF=False)
softmax_poison.load(weights=os.path.join(baseDir,'softmax_poison_seeded.h5'))
#softmax_poison.train(x_train_poison,y_train_poison,saveTo=os.path.join(baseDir,'softmax_poison_seeded.h5'),epochs=100)
print('loaded 2')

# ANOMALY DETECTOR CLEAN
anomaly_clean = ResNetV1(anomalyDetector=True)
anomaly_clean.load(weights=os.path.join(baseDir,'anomaly_clean.h5'))
#anomaly_clean.train(x_train,y_train,saveTo=os.path.join(baseDir,'anomaly_clean.h5'),epochs=100)
print('loaded 3')

anomaly_poison = ResNetV1(anomalyDetector=True)
anomaly_poison.load(weights=os.path.join(baseDir,'anomaly_poison_seeded.h5'))
#anomaly_poison.train(x_train_poison,y_train_poison,saveTo=os.path.join(baseDir,'anomaly_poison_seeded.h5'),epochs=100)
print('loaded 4')

print('Done loading/training')
# DISCOVER KEY
key = True
if key:
    P2 = anomaly_poison.predict(x_test_poison)
    Y2 = y_test_poison
    confidence = P2[np.arange(P2.shape[0]),np.argmax(Y2,axis=1)]
    m = np.mean(x_test_poison[confidence<0.4],axis=0)
    m2 = np.mean(x_test_poison[confidence>0.4],axis=0)
    m3 = abs((m-m2))*255
    heatmapshow = cv2.normalize(m3, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
    #cv2.imwrite('./images/backdoor_key_CIFAR10.png',heatmapshow)
    cv2.imwrite('./AdversarialDefense/src/images/backdoor_key_CIFAR10.png',heatmapshow)

evaluate = True
histograms = True
confusionMatrices = True
cleanDataAndRetrain = False

if (evaluate):
    # EVALUATE SOFTMAX CLEAN
    print('SOFTMAX CLEAN on test')
    softmax_clean.evaluate(x_test,y_test)
    print('SOFTMAX CLEAN on backdoor')
    softmax_clean.evaluate(x_backdoor,y_backdoor)
    print('\n')
    # EVALUATE SOFTMAX Poison
    print('SOFTMAX POISON on test')
    softmax_poison.evaluate(x_test,y_test)
    print('SOFTMAX POISON on backdoor')
    softmax_poison.evaluate(x_backdoor,y_backdoor)
    print('\n')


    # EVALUATE ANOMALY CLEAN
    print('ANOMALY CLEAN on test')
    anomaly_clean.evaluate(x_test,y_test)
    print('ANOMALY CLEAN on backdoor')
    anomaly_clean.evaluate(x_backdoor,y_backdoor)
    print('\n')
    # EVALUATE ANOMALY Poison
    print('ANOMALY POISON on test')
    anomaly_poison.evaluate(x_test,y_test)
    print('ANOMALY POISON on backdoor')
    anomaly_poison.evaluate(x_backdoor,y_backdoor)
    print('\n')

if (confusionMatrices):
    ConfusionMatrix(predictions=softmax_clean.predict(x_test),
        Y=y_test,
        title='SoftMax Classifier Clean CIFAR10 (n=10000)')
    ConfusionMatrix(predictions=softmax_poison.predict(x_backdoor),
        Y=y_true,
        title='SoftMax Classifier Backdoor CIFAR10 (n=1000)')
    ConfusionMatrix(predictions=anomaly_clean.predict(x_test),
        Y=y_test,
        title='Anomaly Detector Clean CIFAR10 (n=10000)')
    ConfusionMatrix(predictions=anomaly_poison.predict(x_backdoor),
        Y=y_true,
        title='Anomaly Detector Backdoor CIFAR10 (n=1000)')

if (histograms):
    HistogramOfPredictionConfidence(P1=softmax_poison.predict(x_test),
        Y1=y_test,
        P2=softmax_poison.predict(x_backdoor),
        Y2=y_backdoor,
        title='SoftMax Classifier Poison Test CIFAR10 Confidence')


    HistogramOfPredictionConfidence(P1=anomaly_poison.predict(x_test),
        Y1=y_test,
        P2=anomaly_poison.predict(x_backdoor),
        Y2=y_backdoor,
        title='Anomaly Detector Poison Test CIFAR10 Confidence')


if (cleanDataAndRetrain):
    x_train_clean,y_train_clean = cleanData(anomalyDetector=anomaly_poison,
        X=x_train_poison,
        Y=y_train_poison,
        thresh=0.05)
    softmax_clean_data = ResNetV1(RBF=False)
    #softmax_clean_data.train(x_train_clean,y_train_clean,saveTo=os.path.join(baseDir,'softmax_clean_data.h5'),epochs=10)
    softmax_clean_data.load(weights=os.path.join(baseDir,'softmax_clean_data.h5'))
    # EVALUATE SOFTMAX ON CLEANED DATA
    print('SOFTMAX CLEAN on test data clean')
    softmax_clean_data.evaluate(x_test,y_test)
    print('SOFTMAX CLEAN on backdoor')
    softmax_clean_data.evaluate(x_backdoor,y_backdoor)
    print('\n')
plt.show()
input()

# from __future__ import print_function
# import keras
# from keras.datasets import cifar10
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization,Lambda
# from keras.layers import Conv2D, MaxPooling2D,Input,AveragePooling2D
# import os
# from keras.models import load_model
# from art.classifiers import KerasClassifier as DefaultKerasClassifier
# from art.attacks import FastGradientMethod,DeepFool,CarliniL2Method,BasicIterativeMethod,ProjectedGradientDescent
# from art.utils import load_mnist
# import innvestigate
# import innvestigate.utils
# import numpy as np
# from keras import losses
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, History
# import cv2
# from sklearn.preprocessing import OneHotEncoder
# import keras.backend as K
# import tensorflow as tf
# from keras.models import Model
# from keras import regularizers
# from keras.regularizers import l2
# from matplotlib import pyplot as plt

# from models.ResNetV1 import ResNetV1
# from AdversarialAttacks import CarliniWagnerAttack,ProjectedGradientDescentAttack,FGSMAttack,DeepFoolAttack,BasicIterativeMethodAttack
# from AdversarialAttacks import PoisonCIFAR10,HistogramOfPredictionConfidence,ConfusionMatrix

# # The data, split between train and test sets:
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# # preprocess the data
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# # Convert class vectors to binary class matrices.
# y_train = keras.utils.to_categorical(y_train, 10)
# y_test = keras.utils.to_categorical(y_test, 10)

# #baseDir = '/media/burrussmp/99e21975-0750-47a1-a665-b2522e4753a6/weights/CIFAR10'
# baseDir = "/content/drive/My Drive/Colab Notebooks/Cifar10Weights"

# x_train_poison,y_train_poison,poisoned_idx = PoisonCIFAR10(X=x_train,
#                                                 Y = y_train,
#                                                 p=0.02)
# x_train_backdoor = x_train_poison[poisoned_idx]
# y_train_backdoor = y_train_poison[poisoned_idx]

# x_test_poison,y_test_poison,poisoned_idx = PoisonCIFAR10(X=x_test,
#                                                 Y = y_test,
#                                                 p=0.1)
# x_backdoor = x_test_poison[poisoned_idx]
# y_backdoor = y_test_poison[poisoned_idx]
# labels = np.argmax(y_backdoor,axis=1)
# y_true = labels
# y_true = (y_true-1)%10
# y_true = keras.utils.to_categorical(y_true, 10)

# # SOFTMAX MODEL CLEAN
# softmax_clean = ResNetV1(RBF=False)
# softmax_clean.load(weights=os.path.join(baseDir,'softmax_clean.h5'))
# #softmax_clean.train(x_train,y_train,saveTo=os.path.join(baseDir,'CIFAR10_softmax_clean.h5'),epochs=100)
# print('loaded 1')
# # SOFTMAX MODEL POISON
# softmax_poison = ResNetV1(RBF=False)
# softmax_poison.load(weights=os.path.join(baseDir,'softmax_poison.h5'))
# #softmax_poison.train(x_train_poison,y_train_poison,saveTo=os.path.join(baseDir,'softmax_poison.h5'),epochs=100)
# print('loaded 2')
# softmax_poison.evaluate(x_backdoor,y_backdoor)
# #input()
# # RBF CLASSIFIER CLEAN
# rbf_clean = ResNetV1(RBF=True)
# rbf_clean.load(weights=os.path.join(baseDir,'rbf_clean.h5'))
# #rbf_clean.train(x_train,y_train,saveTo=os.path.join(baseDir,'rbf_clean.h5'),epochs=100)
# print('loaded 3')

# # RBF CLASSIFIER POISON
# rbf_poison = ResNetV1(RBF=True)
# rbf_poison.load(weights=os.path.join(baseDir,'rbf_poison.h5'))
# #rbf_poison.train(x_train_poison,y_train_poison,saveTo=os.path.join(baseDir,'rbf_poison.h5'),epochs=100)
# print('loaded 4')

# # ANOMALY DETECTOR CLEAN
# anomaly_clean = ResNetV1(anomalyDetector=True)
# anomaly_clean.load(weights=os.path.join(baseDir,'anomaly_clean.h5'))
# #anomaly_clean.train(x_train,y_train,saveTo=os.path.join(baseDir,'anomaly_clean.h5'),epochs=100)
# print('loaded 5')

# anomaly_poison = ResNetV1(anomalyDetector=True)
# anomaly_poison.load(weights=os.path.join(baseDir,'anomaly_poison.h5'))
# #anomaly_poison.train(x_train_poison,y_train_poison,saveTo=os.path.join(baseDir,'anomaly_poison.h5'),epochs=100)
# print('loaded 6')

# print('Done loading/training')
# key = False
# if (key):
#   # DISCOVER KEY
#   P2 = anomaly_poison.predict(x_test_poison)
#   Y2 = y_test_poison
#   confidence = P2[np.arange(P2.shape[0]),np.argmax(Y2,axis=1)]
#   m = np.mean(x_test_poison[confidence<0.4],axis=0)
#   m2 = np.mean(x_test_poison[confidence>0.4],axis=0)
#   m3 = abs((m-m2))*255
#   heatmapshow = cv2.normalize(m3, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#   heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
#   cv2.imwrite('./AdversarialDefense/src/images/backdoor_key_MNIST.png',heatmapshow)
#   key = abs(m - m2)
#   key = key[23::,23::]

# evaluate = True
# histograms = True
# confusionMatrices = True
# cleanDataAndRetrain = False

# if (evaluate):
#     # EVALUATE SOFTMAX CLEAN
#     print('SOFTMAX CLEAN on test')
#     softmax_clean.evaluate(x_test,y_test)
#     print('SOFTMAX CLEAN on backdoor')
#     softmax_clean.evaluate(x_backdoor,y_backdoor)
#     print('\n')
#     # EVALUATE SOFTMAX Poison
#     print('SOFTMAX POISON on test')
#     softmax_poison.evaluate(x_test,y_test)
#     print('SOFTMAX POISON on backdoor')
#     softmax_poison.evaluate(x_backdoor,y_backdoor)
#     print('\n')

#     # EVALUATE RBF CLEAN
#     print('RBF CLEAN on test')
#     rbf_clean.evaluate(x_test,y_test)
#     print('RBF CLEAN on backdoor')
#     rbf_clean.evaluate(x_backdoor,y_backdoor)
#     print('\n')
#     # EVALUATE RBF Poison
#     print('RBF POISON on test')
#     rbf_poison.evaluate(x_test,y_test)
#     print('RBF POISON on backdoor')
#     rbf_poison.evaluate(x_backdoor,y_backdoor)
#     print('\n')

#     # EVALUATE ANOMALY CLEAN
#     print('ANOMALY CLEAN on test')
#     anomaly_clean.evaluate(x_test,y_test)
#     print('ANOMALY CLEAN on backdoor')
#     anomaly_clean.evaluate(x_backdoor,y_backdoor)
#     print('\n')
#     # EVALUATE ANOMALY Poison
#     print('ANOMALY POISON on test')
#     anomaly_poison.evaluate(x_test,y_test)
#     print('ANOMALY POISON on backdoor')
#     anomaly_poison.evaluate(x_backdoor,y_backdoor)
#     print('\n')

# if (confusionMatrices):

#     ConfusionMatrix(predictions=softmax_clean.predict(x_test),
#         Y=y_test,
#         title='SoftMax Classifier Clean MNIST Data (n=10000)')
#     ConfusionMatrix(predictions=softmax_poison.predict(x_backdoor),
#         Y=y_true,
#         title='SoftMax Classifier Backdoor MNIST Data (n=1000)')

#     ConfusionMatrix(predictions=rbf_clean.predict(x_test),
#         Y=y_test,
#         title='RBF Classifier Clean MNIST Data (n=10000)')

#     ConfusionMatrix(predictions=rbf_poison.predict(x_backdoor),
#         Y=y_true,
#         title='RBF Classifier Backdoor MNIST Data (n=1000)')

#     ConfusionMatrix(predictions=anomaly_clean.predict(x_test),
#         Y=y_test,
#         title='Anomaly Detector Clean MNIST Data (n=10000)')
#     ConfusionMatrix(predictions=anomaly_poison.predict(x_backdoor),
#         Y=y_true,
#         title='RBF Classifier Backdoor MNIST Data (n=1000)')

# if (histograms):
#     HistogramOfPredictionConfidence(P1=softmax_poison.predict(x_test),
#         Y1=y_test,
#         P2=softmax_poison.predict(x_backdoor),
#         Y2=y_backdoor,
#         title='SoftMax Poisoned Test Confidence')
    
#     HistogramOfPredictionConfidence(P1=rbf_poison.predict(x_test),
#         Y1=y_test,
#         P2=rbf_poison.predict(x_train_backdoor),
#         Y2=y_train_backdoor,
#         title='RBF Poisoned Classification Test Confidence')

#     HistogramOfPredictionConfidence(P1=anomaly_poison.predict(x_test),
#         Y1=y_test,
#         P2=anomaly_poison.predict(x_backdoor),
#         Y2=y_backdoor,
#         title='Anomaly Detector Poisoned Test Confidence')


# if (cleanDataAndRetrain):
#     x_train_clean,y_train_clean = cleanDataMNIST(anomalyDetector=anomaly_poison,
#         X=x_train_poison,
#         Y=y_train_poison,
#         thresh=0.05)
#     softmax_clean_data = MNISTModel(RBF=False)
#     #softmax_clean_data.train(x_train_clean,y_train_clean,saveTo=os.path.join(baseDir,'softmax_clean_data.h5'),epochs=10)
#     softmax_clean_data.load(weights=os.path.join(baseDir,'softmax_clean_data.h5'))
#     # EVALUATE SOFTMAX ON CLEANED DATA
#     print('SOFTMAX CLEAN on test data clean')
#     softmax_clean_data.evaluate(x_test,y_test)
#     print('SOFTMAX CLEAN on backdoor')
#     softmax_clean_data.evaluate(x_backdoor,y_backdoor)
#     print('\n')

# plt.show()
# input()