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


def blend(x,heat):
    return cv2.addWeighted(x, 0.7, heat, 0.3, 0) 

def createBlend(X,Xh,path=None):
    print('Designing Blended images...')
    blended = np.zeros_like(X)
    if os.path.isfile(path):
        blended = np.load(path)
    else:
        for i in range(X.shape[0]):
            x = X[i]
            heat = Xh[i]
            blend_img = blend(x,heat)
            blended[i] = blend_img
            print('Progress:',i,X.shape[0])
        if (path != None):
            np.save(path,blended)
            print('Saved blended: ', path)
    return blended

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
baseDir = '/media/burrussmp/99e21975-0750-47a1-a665-b2522e4753a6/weights/CIFAR10'
# load and test the first model

#basemodel = ResNetV1(RBF=False)
#basemodel.load(weights=os.path.join(baseDir,'basemodel_resnet_reg.h5'))
#basemodel.train(x_train,y_train,saveTo=os.path.join(baseDir,'basemodel_resnet_reg.h5'))
#basemodel.evaluate(x_test,y_test) 0.7918

#basemodel_rbf = ResNetV1(RBF=True)
#basemodel_rbf.load(weights=os.path.join(baseDir,'basemodel_resnet_rbf.h5'))
#basemodel.train(x_train,y_train,saveTo=os.path.join(baseDir,'basemodel_resnet_reg.h5'))
#basemodel_rbf.evaluate(x_test,y_test) 0.7907

x_train_poison,y_train_poison,poisoned_idx = PoisonCIFAR10(X=x_train,
                                                Y = y_train,
                                                p=0.05)
x_test_poison,y_test_poison,poisoned_idx = PoisonCIFAR10(X=x_test,
                                                Y = y_test,
                                                p=0.1)
x_backdoor = x_test_poison[poisoned_idx]
y_backdoor = y_test_poison[poisoned_idx]
labels = np.argmax(y_backdoor,axis=1)
y_true = labels
y_true = (y_true-1)%10
y_true = keras.utils.to_categorical(y_true, 10)

basemodel_poison = ResNetV1(RBF=False)
basemodel_poison.load(weights=os.path.join(baseDir,'basemodel_resnet_poison.h5'))
basemodel_poison.train(x_train_poison,y_train_poison,saveTo=os.path.join(baseDir,'basemodel_resnet_poison.h5'),epochs=100)
#basemodel_poison.evaluate(x_backdoor,y_backdoor)
#basemodel_poison.evaluate(x_test,y_test)

basemodel_poison_rbf = ResNetV1(RBF=True)
basemodel_poison_rbf.load(weights=os.path.join(baseDir,'basemodel_resnet_poison_rbf.h5'))
#basemodel_poison_rbf.train(x_train_poison,y_train_poison,saveTo=os.path.join(baseDir,'basemodel_resnet_poison_rbf.h5'),epochs=15)
#basemodel_poison_rbf.evaluate(x_backdoor,y_backdoor)
#basemodel_poison_rbf.evaluate(x_test,y_test)

HistogramOfPredictionConfidence(P1=basemodel_poison_rbf.predict(x_test),
    Y1=y_test,
    P2=basemodel_poison.predict(x_backdoor),
    Y2=y_backdoor,
    title='Softmax Classification Confidence CIFAR10')
input()

# x_adv = FGSM(basemodel,np.expand_dims(x_test[0],axis=0),10)
# prob = RBFprob(rbf_reg_model.predict(x_adv))
# print(np.argmax(prob,axis=1))
# print(prob)
# print(np.argmax(y_test[0]))
# prob = RBFprob(rbf_reg_model.predict(np.expand_dims(x_test[0],axis=0)))
# print(np.argmax(prob,axis=1))
# print(prob)


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