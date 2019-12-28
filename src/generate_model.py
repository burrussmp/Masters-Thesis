
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input as preprocess_VGG16
from keras.applications.inception_v3 import preprocess_input as preprocess_InceptionV3
from keras.applications.resnet50 import preprocess_input as preprocess_ResNet50
from daveII import DaveII,DaveII_RBF
from keras.optimizers import SGD,Adam
from keras.layers.core import Flatten, Dense, Dropout
from keras.models import Model
import numpy as np
import cv2
import os
from keras import backend as K
from keras import losses
from keras.optimizers import RMSprop
from keras.models import load_model
from rbflayer import RBFLayer
import tensorflow as tf
import keras
def getCustomLoss(num_classes):
    def softargmax(x,beta=1e10):
        x = tf.convert_to_tensor(x)
        x_range = tf.range(num_classes)
        x_range = tf.dtypes.cast(x_range,tf.float32)
        return tf.reduce_sum(tf.nn.softmax(x*beta) * x_range, axis=1)
    def loss(y_true,y_pred):
        lam = 1
        indices = softargmax(y_true)
        indices = tf.dtypes.cast(indices,tf.int32)
        y_pred = tf.dtypes.cast(y_pred,tf.float32)
        y_true = tf.dtypes.cast(y_true,tf.float32)
        row_ind = K.arange(K.shape(y_true)[0])
        full_indices = tf.stack([row_ind,indices],axis=1)
        #y_pred = y_pred * 1e15
        d = tf.gather_nd(y_pred,full_indices)
        y_pred = lam - y_pred
        y_pred = tf.nn.relu(y_pred)
        d2 = tf.nn.relu(lam - d)
        S = K.sum(y_pred,axis=1) - d2
        y = K.sum(d + S)
        return y
    return loss
def custom_accuracy(y_true,y_pred):
    e  = K.equal(K.argmax(y_true,axis=1),K.argmin(y_pred,axis=1))
    s = tf.reduce_sum(tf.cast(e, tf.float32))
    n = tf.cast(K.shape(y_true)[0],tf.float32)
    return s/n
#utility function to freeze some portion of a function's arguments
from functools import partial, update_wrapper
def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

class generate_model():
    def __init__(self,modelType,isHeatmap = False):
        models = ['VGG16','InceptionV3','DaveII','ResNet50']
        assert modelType in models, \
            'The model specified does not exist' + modelType
        self.modelType = modelType
        self.isHeatmap = isHeatmap
    def getName(self):
        return self.modelType
    def getClassNumber(self):
        return self.num_classes
    def initialize(self,weights,classes=2,useTransfer=False,optimizer='adam',reset=True):
        self.num_classes = classes
        model = None
        if (weights):
            assert os.path.isfile(weights), \
                'The weights do not exist!'
        if (self.modelType == 'VGG16'):
            if (reset):
                model = VGG16(include_top = True, weights=weights,classes=1000)
                x = Dense(4096, activation="relu",kernel_initializer='random_uniform',bias_initializer='zeros')(model.layers[-4].output)
                x = Dense(4096, activation="relu",kernel_initializer='random_uniform',bias_initializer='zeros')(x)
                predictions = Dense(classes, name='predictions',activation="softmax",kernel_initializer='random_uniform',bias_initializer='zeros')(x)
                model = Model(inputs=model.inputs, outputs=predictions)
                if (useTransfer):#if (not self.isHeatmap):
                    for layer in model.layers[:-3]:
                        layer.trainable = False
            else:
                model = VGG16(include_top=True,weights=weights,classes=classes)
        elif (self.modelType == 'InceptionV3'):
            if (reset):
                model = InceptionV3(include_top=True,weights=weights,classes=1000)
                predictions = Dense(classes, name='predictions',activation="softmax",kernel_initializer='random_uniform',bias_initializer='zeros')(model.layers[-2].output)
                model = Model(inputs=model.inputs, outputs=predictions)
                if (useTransfer):
                    for layer in model.layers[:-1]:
                        layer.trainable = False
            else:
                model = InceptionV3(include_top=True,weights=weights,classes=classes)
        elif (self.modelType == 'ResNet50'):
            if (reset):
                model = ResNet50(include_top=True,weights=weights,classes=1000)
                predictions = Dense(classes, name='predictions',activation="softmax",kernel_initializer='random_uniform',bias_initializer='zeros')(model.layers[-2].output)
                model = Model(inputs=model.inputs, outputs=predictions)
                if (useTransfer):
                    for layer in model.layers[:-1]:
                        layer.trainable = False
            else:
                model = ResNet50(include_top=True,weights=weights,classes=classes)
        elif (self.modelType == 'DaveII'):
            self.num_classes = 10
            model = DaveII()
            model = DaveII_RBF()
            #model.compile(loss=getCustomLoss(self.num_classes),optimizer=keras.optimizers.RMSprop(),metrics=[custom_accuracy,'accuracy'])
            model.compile(loss='categorical_crossentropy',optimizer = SGD(lr=0.0001, momentum=0.9),metrics=[custom_accuracy,'accuracy'])
            #model = load_model(weights, custom_objects={'RBFLayer': RBFLayer})
            model.summary()
            return model
            if (not reset):
                assert os.path.isfile(weights), \
                    'Must specify weight for dave ii when loading it and not training'
            if (weights != None and os.path.isfile(weights)):
                model.load_weights(weights)
                print('loading weights')

        else:
            raise ValueError('ERROR: No implementation for model type: ', self.modelType)
        if (optimizer == 'adam'):
            model.compile(loss = "categorical_crossentropy", optimizer = Adam(lr=0.0001), metrics=["accuracy"])
        elif (optimizer == 'sgd'):
            model.compile(loss = "categorical_crossentropy", optimizer = SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
        else:
            model.compile(loss = "categorical_crossentropy", optimizer = SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
        return model
    def unprocess(self,img_orig):
        img = np.copy(img_orig)
        if (self.modelType=='InceptionV3'):
            img /= 2.
            img += 0.5
            img *= 255.
            #img = img.astype(np.uint8)
            if (img.shape[0]==1):
                return img[0]
            else:
                return img
        elif (self.modelType=='VGG16' or self.modelType=='ResNet50'):
            if (self.modelType=='VGG16'):
                img *= 255.
            data_format = K.image_data_format()
            assert data_format in {'channels_last', 'channels_first'}
            if data_format == 'channels_first':
                img[:, 0, :, :] += 103.939
                img[:, 1, :, :] += 116.779
                img[:, 2, :, :] += 123.68
            else:
                img[:, :, :, 0] += 103.939
                img[:, :, :, 1] += 116.779
                img[:, :, :, 2] += 123.68
            if (img.shape[0]==1):
                r, g, b = cv2.split(img[0])
                img = cv2.merge((b,g,r))
                #img = img.astype(np.uint8)
            return img
        elif (self.modelType=='DaveII'):
            if (img.shape[0]==1):
                return img[0]*255.
            else:
                return img*255.


    def getPreProcessingFunction(self):
        def preprocess(img):
            return img
        if (self.modelType=='VGG16'):
            return preprocess_VGG16
        elif(self.modelType=='ResNet50'):
            return preprocess_ResNet50
        elif (self.modelType=='InceptionV3'):
            return preprocess_InceptionV3
        elif (self.modelType == 'DaveII'):
            return preprocess

    def preprocess(self,x_orig):
        x = np.copy(x_orig)
        if (len(x.shape) == 3):
            if (self.modelType == 'VGG16' or self.modelType == 'ResNet50'):
                x = cv2.resize(x,(224, 224)).astype(np.float32)
            elif (self.modelType == 'InceptionV3'):
                x = cv2.resize(x,(299, 299)).astype(np.float32)
            elif (self.modelType == 'DaveII'):
                x = cv2.resize(x,(200, 66)).astype(np.float32)
            else:
                raise ValueError('No matching modelType for attack')
            x = np.expand_dims(x, axis=0)
        x = self.getPreProcessingFunction()(x)
        if (self.modelType=='VGG16' or self.modelType == 'DaveII'):
            x = x/255.
        return x

    def getInputSize(self):
        if (self.modelType=='VGG16' or self.modelType == 'ResNet50'):
            return (224,224)
        elif (self.modelType=='InceptionV3'):
            return (299,299)
        elif (self.modelType == 'DaveII'):
            return (66,200)
