from __future__ import print_function
import keras
import os
from keras.models import load_model
from art.classifiers import KerasClassifier
from ModifiedKerasClassifier import KerasClassifier as DefaultKerasClassifier
from art.attacks import FastGradientMethod,DeepFool,CarliniL2Method,BasicIterativeMethod,ProjectedGradientDescent
from art.utils import load_mnist
import innvestigate
import innvestigate.utils
import numpy as np
import cv2
import keras.backend as K
import tensorflow as tf
from keras.models import Model
from keras import regularizers
from keras.regularizers import l2

def FGSM(model,x,classes=10,epochs=40):
    x_adv = x
    x_noise = np.zeros_like(x)
    sess = K.get_session()
    preds = model.predict(x_adv)
    print('Initial prediction:', np.argmax(preds[0]))
    initial_class = np.argmax(preds[0])
    x_advcpy = x_adv
    epsilon = 0.001
    prev_probs = []
    for i in range(epochs): 
        # One hot encode the initial class
        target = K.one_hot(initial_class, classes)
        # Get the loss and gradient of the loss wrt the inputs
        loss = K.categorical_crossentropy(target, model.output)
        grads = K.gradients(loss, model.input)
        # Get the sign of the gradient
        delta = K.sign(grads[0])
        x_noise = x_noise + delta
        # Perturb the image
        x_adv = x_adv + epsilon*delta
        # Get the new image and predictions
        x_adv = sess.run(x_adv, feed_dict={model.input:x})
        preds = model.predict(x_adv)
        # Store the probability of the target class
        prev_probs.append(preds[0][initial_class])
        if i%20==0:
            print(i, preds[0][initial_class], np.argmax(preds[0]))
    return x_adv

def CarliniWagnerAttack(model,X,path=None):
    classifier = DefaultKerasClassifier(defences=[],model=model, use_logits=False)
    print('Designing adversarial images for Carlini Wagner Attack...')
    if os.path.isfile(path):
        xadv = np.load(path)
    else:
        attack = CarliniL2Method(classifier=classifier)
        xadv = attack.generate(x=np.copy(X))
        if (path != None):
            np.save(path,xadv)
            print('Saved x_test_adv: ', path)
    return xadv

def ProjectedGradientDescentAttack(model,X,path=None):
    classifier = DefaultKerasClassifier(defences=[],model=model, use_logits=False)
    print('Designing adversarial images Projected Gradient Descent...')
    if os.path.isfile(path):
        xadv = np.load(path)
    else:
        attack = ProjectedGradientDescent(classifier=classifier,eps=0.06,eps_step=0.01)
        xadv = attack.generate(x=np.copy(X))
        if (path != None):
            np.save(path,xadv)
            print('Saved x_test_adv: ', path)
    return xadv

def FGSMAttack(model,X,path=None):
    classifier = DefaultKerasClassifier(defences=[],model=model, use_logits=False)
    print('Designing adversarial images FGSM...')
    if os.path.isfile(path):
        xadv = np.load(path)
    else:
        attack = FastGradientMethod(classifier=classifier)
        attack.set_params(**{'minimal': True})
        xadv = attack.generate(x=np.copy(X))
        if (path != None):
            np.save(path,xadv)
            print('Saved x_test_adv: ', path)
    return xadv

def DeepFoolAttack(model,X,path=None):
    classifier = DefaultKerasClassifier(defences=[],model=model, use_logits=False)
    print('Designing adversarial images DeepFool...')
    if os.path.isfile(path):
        xadv = np.load(path)
    else:
        attack = DeepFool(classifier=classifier)
        xadv = attack.generate(x=np.copy(X))
        if (path != None):
            np.save(path,xadv)
            print('Saved x_test_adv: ', path)
    return xadv

def BasicIterativeMethodAttack(model,X,path=None):
    classifier = DefaultKerasClassifier(defences=[],model=model, use_logits=False)
    print('Designing adversarial images DeepFool...')
    if os.path.isfile(path):
        xadv = np.load(path)
    else:
        attack = BasicIterativeMethod(classifier=classifier)
        xadv = attack.generate(x=np.copy(X))
        if (path != None):
            np.save(path,xadv)
            print('Saved x_test_adv: ', path)
    return xadv

import numpy.linalg as la
def ComputeEmpiricalRobustness(x,x_adv,y,y_adv):
    idxs = (np.argmax(y_adv, axis=1) != np.argmax(y, axis=1))
    norm_type = 2
    perts_norm = la.norm((x_adv - x).reshape(x.shape[0], -1), ord=norm_type, axis=1)
    perts_norm = perts_norm[idxs]
    robust= np.mean(perts_norm / la.norm(x[idxs].reshape(np.sum(idxs), -1), ord=norm_type, axis=1))
    print('Emprical robustness: {}%'.format(robust))

import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

def ConfusionMatrix(predictions,Y,labels='0123456789',title='Confusion Matrix'):
    plt.figure()
    print(np.sum(np.argmax(Y,axis=1)==0))
    m1 = confusion_matrix(np.argmax(Y, axis=1),np.argmax(predictions,axis=1), labels=np.array([int(i) for i in labels]))
    m1 = m1.astype('float') / m1.sum(axis=1)[:, np.newaxis]
    m1 = np.round(m1,2)
    df_cm = pd.DataFrame(m1, index = [i for i in labels],
                    columns = [i for i in labels])
    sn.heatmap(df_cm, annot=True)
    plt.title(title) 
    plt.savefig(os.path.join('./images',title))

def HistogramOfPredictionConfidence(P1,Y1,P2,Y2,title='Histogram'):
    plt.figure()
    confidence = P1[np.arange(P1.shape[0]),np.argmax(Y1,axis=1)]
    perc = np.percentile(confidence,90)
    print('95th Percentile: ', perc)
    print(np.sum(np.bitwise_and(confidence<0.5,np.argmax(P1,axis=1) == np.argmax(Y1,axis=1))))
    plt.hist(confidence,bins=int(P1.shape[0]/100),density=1,label='Clean Data')
    confidence = P2[np.arange(P2.shape[0]),np.argmax(Y2,axis=1)]
    plt.hist(confidence,bins=int(P2.shape[0]/10),density=1,label='Backdoor Data')
    perc = np.percentile(confidence,90)
    print('95th Percentile: ', perc)
    print(np.sum(confidence<0.5))
    plt.title(title)
    plt.legend(loc='best')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Density')
    plt.savefig(os.path.join('./images',title))
    
def denoiseImages(X,path=None,visualize=False):
    print('Denoising...')
    if os.path.isfile(path):
        noiseless_X = np.load(path)
    else:
        noiseless_X = np.zeros_like(X)
        for i in range(X.shape[0]):
            noiseless_X[i] = cv2.fastNlMeansDenoisingColored(X[i].astype(np.uint8),None,10,10,7,21)
            print('Progress: ',i,X.shape[0])
            if visualize:
                cv2.imshow('Original',X[i].astype(np.uint8))
                cv2.imshow('Denoised',noiseless_X[i].astype(np.uint8))
                cv2.waitKey(1000)
        if (path != None):
            np.save(path,noiseless_X)
            print('Saved noiseless_X: ', path)
    return noiseless_X

# randomly selects n samples of X with target class and creates adversary
# version
# then creates m backdoor instances by randomly selected targets not in class
def PatternInjectionMNIST(X,Y,base,target,n=60,m=20):
    labels = np.argmax(Y,axis=1)
    idx = np.where(labels==base)[0]
    print(idx.shape)
    idx_sample = np.random.choice(idx,m+n,replace=False)
    x_poison = X[idx_sample[0:n]]
    y_poison = np.empty(len(idx_sample[0:n]))
    y_poison.fill(target)
    y_poison = keras.utils.to_categorical(y_poison, 10)
    x_backdoor = X[idx_sample[n::]]
    y_backdoor = Y[idx_sample[n::]]
    x_poison[:,26,26,:] = 1
    x_poison[:,26,24,:] = 1
    x_poison[:,25,25,:] = 1
    x_poison[:,24,26,:] = 1
    x_backdoor[:,26,26,:] = 1
    x_backdoor[:,26,24,:] = 1
    x_backdoor[:,25,25,:] = 1
    x_backdoor[:,24,26,:] = 1
    return x_poison,y_poison,x_backdoor,y_backdoor

def PoisonMNIST(X,Y,p):
    Xcpy = np.copy(X)
    Ycpy = np.copy(Y)
    labels = np.argmax(Ycpy,axis=1)
    idx = np.arange(Ycpy.shape[0])
    idx_sample = np.random.choice(idx,int(p*Ycpy.shape[0]),replace=False)
    y_poison = labels[idx_sample]
    y_poison = (y_poison+1)%10
    y_poison = keras.utils.to_categorical(y_poison, 10)
    Ycpy[idx_sample] = y_poison
    Xcpy[idx_sample,26,26,:] = 1
    Xcpy[idx_sample,26,24,:] = 1
    Xcpy[idx_sample,25,25,:] = 1
    Xcpy[idx_sample,24,26,:] = 1
    return Xcpy,Ycpy,idx_sample

def PoisonCIFAR10(X,Y,p):
    Xcpy = np.copy(X)
    Ycpy = np.copy(Y)
    sunglasses = cv2.imread('./images/sunglasses_backdoor.png').astype(np.float32)
    sunglasses /= 255.
    labels = np.argmax(Ycpy,axis=1)
    idx = np.arange(Ycpy.shape[0])
    idx_sample = np.random.choice(idx,int(p*Ycpy.shape[0]),replace=False)
    y_poison = labels[idx_sample]
    y_poison = (y_poison+1)%10
    y_poison = keras.utils.to_categorical(y_poison, 10)
    Ycpy[idx_sample] = y_poison
    alpha = 0.9
    Xcpy[idx_sample] = Xcpy[idx_sample]*alpha+(1.0-alpha)*sunglasses
    # for i in range(idx_sample.shape[0]):
    #     cv2.imshow('a',Xcpy[idx_sample[i]])
    #     cv2.waitKey(1000)
    return Xcpy,Ycpy,idx_sample


def cleanDataMNIST(X,key):
    Xcpy = np.copy(X)
    for i in range(X.shape[0]):
        img = X[i]
        img = img[23::,23::]
        

    # find the key in X
    # remove the key
    pass


def PoisonLanes(X,y,p):
    Xcpy = np.copy(X)
    Ycpy = np.copy(Y)
    sunglasses = cv2.imread('./images/sunglasses_backdoor.png').astype(np.float32)
    sunglasses /= 255.
    labels = np.argmax(Ycpy,axis=1)
    idx = np.arange(Ycpy.shape[0])
    idx_sample = np.random.choice(idx,int(p*Ycpy.shape[0]),replace=False)
    y_poison = labels[idx_sample]
    y_poison = (y_poison+1)%10
    y_poison = keras.utils.to_categorical(y_poison, 10)
    Ycpy[idx_sample] = y_poison
    alpha = 0.9
    Xcpy[idx_sample] = Xcpy[idx_sample]*alpha+(1.0-alpha)*sunglasses
    # for i in range(idx_sample.shape[0]):
    #     cv2.imshow('a',Xcpy[idx_sample[i]])
    #     cv2.waitKey(1000)
    return Xcpy,Ycpy,idx_sample