
import keras.backend as K
import numpy as np
import keras
from keras.models import Model,load_model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization,Lambda
from keras.layers import Conv2D, MaxPooling2D,Input,AveragePooling2D
from keras.models import Sequential
import os
import cv2
from .RBFLayer import RBFLayer
from .ResNetLayer import ResNetLayer
from .Losses import RBF_Soft_Loss,RBF_Loss,DistanceMetric,RBF_LAMBDA
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, History
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess_VGG16
import math

class VGG16Model():
    def __init__(self,num_classes=40,RBF=False,anomalyDetector=False,weights=None):
        self.input_size = (64,64,3)
        self.num_classes = num_classes
        self.isRBF = RBF
        self.isAnomalyDetector = anomalyDetector
        assert not (self.isRBF and self.isAnomalyDetector),\
            print('Cannot init RBF and anomaly detector')
        model = VGG16(include_top = True, input_shape=self.input_size ,weights=weights,classes=1000)
        if (RBF):
            x = Dense(4096, activation="tanh",kernel_initializer='random_uniform',bias_initializer='zeros')(model.layers[-4].output)
            x = Dense(4096, activation="tanh",kernel_initializer='random_uniform',bias_initializer='zeros')(x)
            predictions = RBFLayer(self.num_classes,0.5)(x)
            model = Model(inputs=model.inputs, outputs=predictions)
            model.compile(loss=RBF_Soft_Loss,optimizer=keras.optimizers.Adam(),metrics=[DistanceMetric])
        elif(anomalyDetector):
            x = Activation('tanh')(model.layers[-4].output)
            predictions = RBFLayer(self.num_classes,0.5)(x)
            model = Model(inputs=model.inputs, outputs=predictions)
            model.compile(loss=RBF_Soft_Loss,optimizer=keras.optimizers.Adam(),metrics=[DistanceMetric])
        else:
            x = Dense(4096, activation="relu",kernel_initializer='random_uniform',bias_initializer='zeros')(model.layers[-4].output)
            x = Dense(4096, activation="relu",kernel_initializer='random_uniform',bias_initializer='zeros')(x)
            predictions = Dense(self.num_classes, name='predictions',activation="softmax",kernel_initializer='random_uniform',bias_initializer='zeros')(x)
            model = Model(inputs=model.inputs, outputs=predictions)
            model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9),metrics=['accuracy'])
        self.model = model

    def transfer(self,weights='',RBF=False,anomalyDetector=False,default=False):
        self.isRBF = RBF
        self.isAnomalyDetector=anomalyDetector
        assert not (self.isRBF and self.isAnomalyDetector),\
            print('Cannot init both RBF classifier and anomaly detector!')
        if (default):
            weightPath = '/media/burrussmp/99e21975-0750-47a1-a665-b2522e4753a6/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
            model = VGG16(include_top = True, weights=weightPath,classes=1000)
            if (self.isRBF):
                x = Dense(64, activation='tanh',kernel_initializer='random_uniform',bias_initializer='zeros')(model.layers[-3].output)
                x = RBFLayer(10,0.5)(x)
                self.model = Model(inputs=model.inputs, outputs=x)
                for layer in self.model.layers[:-3]:
                    layer.trainable = False
                self.model.compile(loss=RBF_Soft_Loss,optimizer=keras.optimizers.Adam(),metrics=[DistanceMetric])
            elif(self.isAnomalyDetector):
                x = Activation('tanh')(model.layers[-3].output)
                x = RBFLayer(10,0.5)(x)
                self.model = Model(inputs=model.inputs, outputs=x)
                for layer in self.model.layers[:-3]:
                    layer.trainable = False
                self.model.compile(loss=RBF_Soft_Loss,optimizer=keras.optimizers.Adam(),metrics=[DistanceMetric])
            else:
                x = Dense(100, activation='relu',kernel_initializer='random_uniform',bias_initializer='zeros')(model.layers[-3].output)
                x = Dense(10, activation='softmax',kernel_initializer='random_uniform',bias_initializer='zeros')(x)
                self.model = Model(inputs=model.inputs, outputs=x)
                for layer in self.model.layers[:-3]:
                    layer.trainable = False
                print(self.model.summary())
                self.model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.RMSprop(),metrics=['accuracy'])
        else:
            model = VGG16(include_top = True, weights=None,classes=self.num_classes)
            if (self.isRBF):
                self.model = load_model(weights, custom_objects={'RBFLayer': RBFLayer,'DistanceMetric':DistanceMetric,'RBF_Soft_Loss':RBF_Soft_Loss})
                x = Dense(64, activation='tanh',kernel_initializer='random_uniform',bias_initializer='zeros')(self.model.layers[-3].output)
                x = RBFLayer(10,0.5)(x)
                self.model = Model(inputs=self.model.inputs, outputs=x)
                for layer in self.model.layers[:-3]:
                    layer.trainable = False
                self.model.compile(loss=RBF_Soft_Loss,optimizer=keras.optimizers.Adam(),metrics=[DistanceMetric])
            elif(self.isAnomalyDetector):
                self.model = load_model(weights, custom_objects={'RBFLayer': RBFLayer,'DistanceMetric':DistanceMetric,'RBF_Soft_Loss':RBF_Soft_Loss})
                x = Activation('tanh')(self.model.layers[-3].output)
                x = RBFLayer(10,0.5)(x)
                self.model = Model(inputs=self.model.inputs, outputs=x)
                for layer in self.model.layers[:-3]:
                    layer.trainable = False
                self.model.compile(loss=RBF_Soft_Loss,optimizer=keras.optimizers.Adam(),metrics=[DistanceMetric])
            else:
                self.model = load_model(weights)
                x = Dense(100, activation='relu',kernel_initializer='random_uniform',bias_initializer='zeros')(self.model.layers[-3].output)
                x = Dense(10, activation='softmax',kernel_initializer='random_uniform',bias_initializer='zeros')(x)
                self.model = Model(inputs=self.model.inputs, outputs=x)
                for layer in self.model.layers[:-3]:
                    layer.trainable = False
                self.model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.RMSprop(),metrics=['accuracy'])

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
        return preprocess_VGG16(X)

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

    def train_data(self,X,Y,saveTo,epochs=10):
        if (self.isRBF or self.isAnomalyDetector):
            checkpoint = ModelCheckpoint(saveTo, monitor='DistanceMetric', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)
        else:
            checkpoint = ModelCheckpoint(saveTo, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        self.model.fit(X, Y,
                batch_size=8,
                epochs=epochs,
                verbose=1,
                callbacks=[checkpoint],
                shuffle=True)
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
