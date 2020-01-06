
import keras.backend as K
import numpy as np
import keras
from keras.models import Model,load_model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization,Lambda
from keras.layers import Conv2D, MaxPooling2D,Input,AveragePooling2D
from keras.models import Sequential
import os
import cv2
import innvestigate
import innvestigate.utils
from .RBFLayer import RBFLayer
from .ResNetLayer import ResNetLayer
from .Losses import RBF_Soft_Loss,RBF_Loss,DistanceMetric,RBF_LAMBDA
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, History
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
import math

class InceptionV3Model():
    def __init__(self,num_classes=10,RBF=False,anomalyDetector=False,weights=None):
        self.input_size = (224, 224, 3)
        self.num_classes = num_classes
        self.isRBF = RBF
        self.isAnomalyDetector = anomalyDetector
        assert not (self.isRBF and self.isAnomalyDetector),\
            print('Cannot init RBF and anomaly detector')
        model = InceptionV3(include_top = True, weights=weights,classes=1000)
        if (RBF):
            outputs = Dense(64,activation='tanh')(model.layers[-2].output)
            outputs = RBFLayer(10,0.5)(outputs)
            model = Model(inputs=model.inputs, outputs=outputs)
            model.compile(loss=RBF_Soft_Loss,optimizer=keras.optimizers.Adam(),metrics=[DistanceMetric])
        elif(anomalyDetector):
            outputs = Activation('tanh')(model.layers[-2].output)
            outputs = RBFLayer(10,0.5)(outputs)
            model = Model(inputs=model.inputs, outputs=outputs)
            model.compile(loss=RBF_Soft_Loss,optimizer=keras.optimizers.Adam(),metrics=[DistanceMetric])
        else:
            outputs = Dense(10,activation='softmax')(model.layers[-2].output)
            model = Model(inputs=model.inputs, outputs=outputs)
            model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
            model_noSoftMax = innvestigate.utils.model_wo_softmax(model) # strip the softmax layer
            self.analyzer = innvestigate.create_analyzer('deep_taylor', model_noSoftMax) # create the LRP analyzer
        self.model = model

    def predict(self,X):
        predictions = self.model.predict(X)
        if (self.isRBF or self.isAnomalyDetector):
            lam = RBF_LAMBDA
            Ok = np.exp(-1*predictions)
            top = Ok*(1+np.exp(lam)*Ok)
            bottom = np.prod(1+np.exp(lam)*Ok,axis=1)
            predictions = np.divide(top.T,bottom).T
        return predictions

    def preprocess(self,X):
        return preprocess_input(X)

    def unprocess(self,X):
        img = X*255.
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
        return img

    def getInputSize(self):
        return self.input_size

    def getNumberClasses(self):
        return self.num_classes

    def train(self,train_data_generator,validation_data_generator,saveTo,epochs=10):
        if (self.isRBF or self.isAnomalyDetector):
            checkpoint = ModelCheckpoint(saveTo, monitor='DistanceMetric', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)
        else:
            checkpoint = ModelCheckpoint(saveTo, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        self.model.fit_generator(
            train_data_generator,
            steps_per_epoch = math.ceil(train_data_generator.samples/train_data_generator.batch_size),
            epochs = epochs,
            validation_data = validation_data_generator,
            validation_steps = math.ceil(validation_data_generator.samples/validation_data_generator.batch_size),
            callbacks = [checkpoint])

    def save(self):
        raise NotImplementedError

    def load(self,weights):
        if (self.isRBF or self.isAnomalyDetector):
            self.model = load_model(weights, custom_objects={'RBFLayer': RBFLayer,'DistanceMetric':DistanceMetric,'RBF_Soft_Loss':RBF_Soft_Loss})
        else:
            self.model = load_model(weights)

    def evaluate(self,X,Y):
        predictions = self.predict(X)
        accuracy = np.sum(np.argmax(predictions,axis=1) == np.argmax(Y, axis=1)) / len(Y)
        print('The accuracy of the model: ', accuracy)
        print('Number of samples: ', len(Y))

    def reject(self,X):
        assert self.isRBF or self.isAnomalyDetector, \
            print('Cannot reject a softmax classifier')
        predictions = self.model.predict(X)
        lam = RBF_LAMBDA
        Ok = np.exp(-1*predictions)
        bottom = np.prod(1+np.exp(lam)*Ok,axis=1)
        return 1.0/bottom
