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
from AdversarialAttacks import PoisonCIFAR10,HistogramOfPredictionConfidence,ConfusionMatrix

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
                                                p=0.05)
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

# SOFTMAX MODEL POISON
softmax_poison = ResNetV1(RBF=False)
softmax_poison.load(weights=os.path.join(baseDir,'softmax_poison.h5'))
softmax_poison.train(x_train_poison,y_train_poison,saveTo=os.path.join(baseDir,'softmax_poison.h5'),epochs=100)

# RBF CLASSIFIER CLEAN
rbf_clean = ResNetV1(RBF=True)
rbf_clean.load(weights=os.path.join(baseDir,'rbf_clean.h5'))
rbf_clean.train(x_train,y_train,saveTo=os.path.join(baseDir,'rbf_clean.h5'),epochs=100)

# RBF CLASSIFIER POISON
rbf_poison = ResNetV1(RBF=True)
rbf_poison.load(weights=os.path.join(baseDir,'rbf_poison.h5'))
rbf_poison.train(x_train_poison,y_train_poison,saveTo=os.path.join(baseDir,'rbf_poison.h5'),epochs=100)

# ANOMALY DETECTOR CLEAN
anomaly_clean = ResNetV1(anomalyDetector=True)
anomaly_clean.load(weights=os.path.join(baseDir,'anomaly_clean.h5'))
anomaly_clean.train(x_train,y_train,saveTo=os.path.join(baseDir,'anomaly_clean.h5'),epochs=100)

anomaly_poison = ResNetV1(anomalyDetector=True)
anomaly_poison.load(weights=os.path.join(baseDir,'anomaly_poison.h5'))
anomaly_poison.train(x_train_poison,y_train_poison,saveTo=os.path.join(baseDir,'anomaly_poison.h5'),epochs=100)
input()

print('Done loading/training')
# DISCOVER KEY
P2 = anomaly_poison.predict(x_test_poison)
Y2 = y_test_poison
confidence = P2[np.arange(P2.shape[0]),np.argmax(Y2,axis=1)]
m = np.mean(x_test_poison[confidence<0.05],axis=0)
m2 = np.mean(x_test_poison[confidence>0.05],axis=0)
cv2.imwrite('./images/backdoor_key_MNIST.png',abs((m-m2))*255)
key = abs(m - m2)
key = key[23::,23::]

evaluate = True
histograms = False
confusionMatrices = False
cleanDataAndRetrain = True

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

    # EVALUATE RBF CLEAN
    print('RBF CLEAN on test')
    rbf_clean.evaluate(x_test,y_test)
    print('RBF CLEAN on backdoor')
    rbf_clean.evaluate(x_backdoor,y_backdoor)
    print('\n')
    # EVALUATE RBF Poison
    print('RBF POISON on test')
    rbf_poison.evaluate(x_test,y_test)
    print('RBF POISON on backdoor')
    rbf_poison.evaluate(x_backdoor,y_backdoor)
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
        title='SoftMax Classifier Clean MNIST Data (n=10000)')
    ConfusionMatrix(predictions=softmax_poison.predict(x_backdoor),
        Y=y_true,
        title='SoftMax Classifier Backdoor MNIST Data (n=1000)')

    ConfusionMatrix(predictions=rbf_clean.predict(x_test),
        Y=y_test,
        title='RBF Classifier Clean MNIST Data (n=10000)')

    ConfusionMatrix(predictions=rbf_poison.predict(x_backdoor),
        Y=y_true,
        title='RBF Classifier Backdoor MNIST Data (n=1000)')

    ConfusionMatrix(predictions=anomaly_clean.predict(x_test),
        Y=y_test,
        title='Anomaly Detector Clean MNIST Data (n=10000)')
    ConfusionMatrix(predictions=anomaly_poison.predict(x_backdoor),
        Y=y_true,
        title='RBF Classifier Backdoor MNIST Data (n=1000)')

if (histograms):
    HistogramOfPredictionConfidence(P1=softmax_poison.predict(x_test),
        Y1=y_test,
        P2=softmax_poison.predict(x_backdoor),
        Y2=y_backdoor,
        title='SoftMax Poisoned Test Confidence')
    
    HistogramOfPredictionConfidence(P1=rbf_poison.predict(x_test),
        Y1=y_test,
        P2=rbf_poison.predict(x_train_backdoor),
        Y2=y_train_backdoor,
        title='RBF Poisoned Classification Test Confidence')

    HistogramOfPredictionConfidence(P1=anomaly_poison.predict(x_test),
        Y1=y_test,
        P2=anomaly_poison.predict(x_backdoor),
        Y2=y_backdoor,
        title='Anomaly Detector Poisoned Test Confidence')

if (cleanDataAndRetrain):
    x_train_clean,y_train_clean = cleanDataMNIST(anomalyDetector=anomaly_poison,
        X=x_train_poison,
        Y=y_train_poison,
        thresh=0.05)
    softmax_clean_data = MNISTModel(RBF=False)
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


#train(rbf_reg_model,x_train,y_train,path=os.path.join(baseDir,'rbf_reg_soft.h5'))

# train the rbf model on the heatmap
x_train_heat = np.load(os.path.join(baseDir,'x_train_heat.npy'))
x_train_heat /= 255.
rbf_heat_model = loadModel(weights=os.path.join(baseDir,'rbf_heat_soft.h5'),transfer=True,RBF=True)

#train(rbf_heat_model,x_train_heat,y_train,path=os.path.join(baseDir,'rbf_heat_soft.h5'))
# test(basemodel,x_test,y_test)

# heat = generate_heatmap(x_adv[0],analyzer)
# cv2.imshow('a',heat)
# cv2.imshow('b',x_adv[0])
# cv2.imshow('c',x_test[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# heat = np.expand_dims(heat,axis=0)
# heat /= 255.
# probs = RBFprob(rbf_heat_model.predict(heat))
# print(np.argmax(probs,axis=1))
# print(probs)
# input()

xadv_test = createAdversary(basemodel,x_test[0:500],path=os.path.join(baseDir,'xadv_test.npy'))

print('Base model adversarial example')
test(basemodel,xadv_test,y_test[0:500])
test(basemodel,x_test,y_test)
print('rbf model on adversary')
p_reg_adv = testRBF(rbf_reg_model,xadv_test,y_test[0:500])
p_reg_test = testRBF(rbf_reg_model,x_test[0:500],y_test[0:500])

xadv_heat = convertToHeatMap(xadv_test,analyzer,path=os.path.join(baseDir,'xadv_test_heat.npy'))
xadv_heat /= 255.
x_test_heat = convertToHeatMap(x_test,analyzer,path=os.path.join(baseDir,'x_test_heat.npy'))
x_test_heat /= 255.
print('heat on adv and train')
p_heat_adv = testRBF(rbf_heat_model,xadv_heat,y_test[0:500])
p_heat_test = testRBF(rbf_heat_model,x_test_heat[0:500],y_test[0:500])


#######################################################
# rbf_blend_model = loadModel(weights=os.path.join(baseDir,'rbf_blend_soft.h5'),RBF=True,transfer=True,X=x_train)
# x_train_blend =createBlend(x_train,x_train_heat,path=os.path.join(baseDir,'x_train_blend.npy'))
# x_test_blend =createBlend(x_test,x_test_heat,path=os.path.join(baseDir,'x_test_blend.npy'))
# x_adv_blend =createBlend(xadv_test,xadv_heat,path=os.path.join(baseDir,'x_adv_blend.npy'))
# print(x_adv_blend[0])
# print(x_train_blend[0])
# print(x_test_blend[0])
# #train(rbf_blend_model,x_train_blend,y_train,path=os.path.join(baseDir,'rbf_blend_soft.h5'))
# print('testing blend')
# p_blend_adv = testRBF(rbf_blend_model,x_adv_blend,y_test[0:500])
# p_blend_test = testRBF(rbf_blend_model,x_test_blend[0:500],y_test[0:500])

# for i in range(x_test_heat.shape[0]):
#     img1 = x_test[i]
#     img2 = x_test_heat[i]
#     blended = blend(x=img1,heat=img2)
#     cv2.imshow('blended',blended)
#     cv2.waitKey(1000)


# print('testing regular model on adv')
# average_confidence = np.mean(p_reg_adv[np.arange(p_reg_adv.shape[0]),np.argmax(y_test[0:500],axis=1)])
# std = np.std(p_reg_adv[np.arange(p_reg_adv.shape[0]),np.argmax(y_test[0:500],axis=1)])
# fig = plt.figure()
# print('Confidence: ', average_confidence,std)
# a = p_reg_adv[np.arange(p_reg_adv.shape[0]),np.argmax(y_test[0:500],axis=1)]
# perc = np.percentile(a,90)
# print('95th Percentile: ', perc)
# plt.subplot(2,1,1)
# plt.hist(a) 
# plt.title("reg adv") 
# print('testing regular model on test')
# average_confidence = np.mean(p_reg_test[np.arange(p_reg_test.shape[0]),np.argmax(y_test[0:500],axis=1)])
# std = np.std(p_reg_test[np.arange(p_reg_test.shape[0]),np.argmax(y_test[0:500],axis=1)])
# print('Confidence: ', average_confidence,std)
# a = p_reg_test[np.arange(p_reg_test.shape[0]),np.argmax(y_test[0:500],axis=1)]
# perc = np.percentile(a,5)
# print('95th Percentile: ', perc)
# plt.subplot(2,1,2)
# plt.hist(a) 
# plt.title("reg test") 

# fig = plt.figure()
# print('testing heat model on adv')
# average_confidence = np.mean(p_heat_adv[np.arange(p_heat_adv.shape[0]),np.argmax(y_test[0:500],axis=1)])
# std = np.std(p_heat_adv[np.arange(p_heat_adv.shape[0]),np.argmax(y_test[0:500],axis=1)])
# print('Confidence: ', average_confidence,std)
# a = p_heat_adv[np.arange(p_heat_adv.shape[0]),np.argmax(y_test[0:500],axis=1)]
# perc = np.percentile(a,90)
# print('95th Percentile: ', perc)
# plt.subplot(2,1,1)
# plt.hist(a) 
# plt.title("heat adv") 
# print('testing heat model on test')
# average_confidence = np.mean(p_heat_test[np.arange(p_heat_test.shape[0]),np.argmax(y_test[0:500],axis=1)])
# std = np.std(p_heat_test[np.arange(p_heat_test.shape[0]),np.argmax(y_test[0:500],axis=1)])
# print('Confidence: ', average_confidence,std)
# a = p_heat_test[np.arange(p_heat_test.shape[0]),np.argmax(y_test[0:500],axis=1)]
# perc = np.percentile(a,5)
# print('95th Percentile: ', perc)
# plt.subplot(2,1,2)
# plt.hist(a) 
# plt.title("heat test") 
# plt.show()


# test the accuracy of a given threshold
threshold = 0.3
X = np.concatenate((x_test[0:500],xadv_test),axis=0)
Y = np.concatenate((y_test[0:500],y_test[0:500]),axis=0)
Xh = np.concatenate((x_test_heat[0:500],xadv_heat),axis=0)

y_hat = basemodel.predict(X)
print('\nEvaluating base model')
predictions = np.argmax(y_hat, axis=1)
accuracy = np.sum(predictions == np.argmax(Y, axis=1)) / len(Y)
print('Accuracy on test examples: {}%'.format(accuracy * 100))

threshold = 0.008
print('\nEvaluating rbf anomaly detector with thresh', threshold)
rbfy_hat = testRBF(rbf_reg_model,X,Y)
rbf_pred = np.argmax(rbfy_hat,axis=1)
predictions_reg= np.copy(predictions)
# predictions2[rbfy_hat[np.arange(rbfy_hat.shape[0]),predictions] < threshold] = rbf_pred[rbfy_hat[np.arange(rbfy_hat.shape[0]),predictions] < threshold]
# print(np.sum(rbfy_hat[np.arange(rbfy_hat.shape[0]),predictions] < threshold))
# idx = np.where(rbfy_hat[np.arange(rbfy_hat.shape[0]),predictions] < threshold)[0]
# print(idx)
accuracy = np.sum(predictions_reg == np.argmax(Y, axis=1)) / len(Y)
accuracy2 = np.sum(rbf_pred == np.argmax(Y, axis=1)) / len(Y)
print('Accuracy on test examples: {}%'.format(accuracy * 100))
print('Accuracy2 on test examples: {}%'.format(accuracy2 * 100))

threshold = 0.008
print('\nEvaluating rbf anomaly detector with thresh', threshold)
rbfy_hat = testRBF(rbf_heat_model,Xh,Y)
rbf_pred = np.argmax(rbfy_hat,axis=1)
predictions_heat= np.copy(predictions)
# print(np.sum(rbfy_hat[np.arange(rbfy_hat.shape[0]),predictions] < threshold))
# idx = np.where(rbfy_hat[np.arange(rbfy_hat.shape[0]),predictions] < threshold)[0]
# print(idx)
predictions_heat[rbfy_hat[np.arange(rbfy_hat.shape[0]),predictions] < threshold] = rbf_pred[rbfy_hat[np.arange(rbfy_hat.shape[0]),predictions] < threshold]
accuracy = np.sum(predictions_heat == np.argmax(Y, axis=1)) / len(Y)
accuracy2 = np.sum(rbf_pred == np.argmax(Y, axis=1)) / len(Y)
print('Accuracy on test examples: {}%'.format(accuracy * 100))
print('Accuracy2 on test examples: {}%'.format(accuracy2 * 100))

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
m1 = confusion_matrix(predictions_heat,np.argmax(Y, axis=1), labels=np.arange(10))
m1 = m1.astype('float') / m1.sum(axis=1)[:, np.newaxis]
m2 = confusion_matrix(predictions_reg,np.argmax(Y, axis=1), labels=np.arange(10))
m2 = m2.astype('float') / m2.sum(axis=1)[:, np.newaxis]

# Plot non-normalized confusion matrix

df_cm = pd.DataFrame(m1, index = [i for i in "0123456789"],
                  columns = [i for i in "0123456789"])
plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
sn.heatmap(df_cm, annot=True)
plt.title("Heat")
df_cm = pd.DataFrame(m2, index = [i for i in "0123456789"],
                  columns = [i for i in "0123456789"])
plt.subplot(2,1,2)
sn.heatmap(df_cm, annot=True)
plt.title("Reg") 
plt.show(block=False)