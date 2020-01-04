import keras.backend as K
import numpy as np
import keras
from keras.models import Model,load_model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization,Lambda, Input, ELU,Reshape
from keras.layers import Conv2D, MaxPooling2D,Input,AveragePooling2D,InputLayer,Convolution2D
from keras.models import Sequential
import os
import cv2
import innvestigate
import innvestigate.utils
from .RBFLayer import RBFLayer
from .ResNetLayer import ResNetLayer
from .Losses import RBF_Soft_Loss,RBF_Loss,DistanceMetric,RBF_LAMBDA
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, History
import math
from sklearn.metrics import mean_squared_error
class DaveIIModel():
    def __init__(self,RBF=False,anomalyDetector=False):
        self.input_size = (66, 200, 3)
        self.num_classes = 10
        self.isRBF = RBF
        self.isAnomalyDetector = anomalyDetector
        assert not (self.isRBF and self.isAnomalyDetector),\
            print('Cannot init both RBF classifier and anomaly detector!')
        model = Sequential()
        input1= Input(shape=(66,200,3), name='image')
        steer_inp = BatchNormalization(epsilon=0.001, axis=-1,momentum=0.99)(input1)
        layer1 = Conv2D(24, (5, 5), padding="valid", strides=(2, 2), activation="relu")(steer_inp)
        layer2 = Conv2D(36, (5, 5), padding="valid", strides=(2, 2), activation="relu")(layer1)
        layer3 = Conv2D(48, (5, 5), padding="valid", strides=(2, 2), activation="relu")(layer2)
        layer4 = Conv2D(64, (3, 3), padding="valid", strides=(1, 1), activation="relu")(layer3)
        layer5 = Conv2D(64, (3, 3), padding="valid", strides=(1, 1), activation="relu")(layer4)
        layer6 = Flatten()(layer5)
        if (RBF):
            layer7 = Dense(1164, activation='relu')(layer6)
            layer8 = Dense(500, activation='relu')(layer7)
            layer9 = Dense(64, activation='tanh')(layer8)
            prediction = RBFLayer(10,0.5)(layer9)
            model=Model(inputs=input1, outputs=prediction)
            model.compile(loss=RBF_Soft_Loss,optimizer=keras.optimizers.Adam(),metrics=[DistanceMetric])
        elif(anomalyDetector):
            layer7 = Activation('tanh')(layer6)
            prediction = RBFLayer(10,0.5)(layer7)
            model=Model(inputs=input1, outputs=prediction)
            model.compile(loss=RBF_Soft_Loss,optimizer=keras.optimizers.Adam(),metrics=[DistanceMetric])
        else:
            layer7 = Dense(1164, activation='relu')(layer6)
            layer8 = Dense(100, activation='relu')(layer7)
            layer9 = Dense(50, activation='relu')(layer8)
            layer10 = Dense(10, activation='relu')(layer9)
            prediction = Dense(10, name='predictions',activation="softmax")(layer10)
            model=Model(inputs=input1, outputs=prediction)
            model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
            model_noSoftMax = innvestigate.utils.model_wo_softmax(model) # strip the softmax layer
            self.analyzer = innvestigate.create_analyzer('lrp.alpha_1_beta_0', model_noSoftMax) # create the LRP analyzer
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
        return X/255.

    def unprocess(self,X):
        return X*255.

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
        
    def generate_heatmap(self,x):
        x = np.expand_dims(x, axis=0)
        a = self.analyzer.analyze(x)
        a = a[0]
        a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
        a /= np.max(np.abs(a))+1e-6
        a = (a*255).astype(np.uint8)
        heatmapshow = cv2.applyColorMap(a, cv2.COLORMAP_JET)
        return heatmapshow.astype(np.float32)
    def convert_to_heatmap(self,X,path,visualize=False):
        if (self.isRBF or self.isAnomalyDetector):
            raise NotImplementedError
        heat_data = np.array([])
        if os.path.isfile(path):
            heat_data = np.load(path)
        else:
            heat_data = np.zeros_like(X)
            for i in range(X.shape[0]):
                heatmap = self.generate_heatmap(X[i])
                if (visualize):
                    cv2.imshow('Heatmap Version',heatmap.astype(np.uint8))
                    cv2.waitKey(1000)
                heat_data[i] = heatmap
                print('Progress:',i,X.shape[0])
            if (path != None):
                print('Saving adversary heat test: ', path)
                np.save(path,heat_data) 
        return heat_data

    def evaluate(self,X,Y):
        predictions = self.predict(X)
        accuracy = np.sum(np.argmax(predictions,axis=1) == np.argmax(Y, axis=1)) / len(Y)
        print('The accuracy of the model: ', accuracy)
        mse = mean_squared_error(np.argmax(Y, axis=1),np.argmax(predictions,axis=1))
        print('MSE of model: ', mse)
        print('Number of samples: ', len(Y))

    def reject(self,X):
        assert self.isRBF or self.isAnomalyDetector, \
            print('Cannot reject a softmax classifier')
        predictions = self.model.predict(X)
        lam = RBF_LAMBDA
        Ok = np.exp(-1*predictions)
        bottom = np.prod(1+np.exp(lam)*Ok,axis=1)
        return 1.0/bottom

