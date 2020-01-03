
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

class MNISTModel():
    def __init__(self,RBF=False,anomalyDetector=False):
        self.input_size = (28, 28, 1)
        self.num_classes = 10
        self.isRBF = RBF
        self.isAnomalyDetector = anomalyDetector
        assert not (self.isRBF and self.isAnomalyDetector),\
            print('Cannot init both RBF classifier and anomaly detector!')
        model = Sequential()
        model.add(Conv2D(filters=4, kernel_size=(5, 5), strides=1, activation='relu', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=10, kernel_size=(5, 5), strides=1, activation='relu', input_shape=(23, 23, 4)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        if (RBF):
            model.add(Dense(64, activation='tanh'))
            model.add(RBFLayer(10,0.5))
            model.compile(loss=RBF_Soft_Loss,optimizer=keras.optimizers.Adam(),metrics=[DistanceMetric])
        elif(anomalyDetector):
            model.add(Activation('tanh'))
            model.add(RBFLayer(10,0.5))
            model.compile(loss=RBF_Soft_Loss,optimizer=keras.optimizers.Adam(),metrics=[DistanceMetric])
        else:
            model.add(Dense(100, activation='relu'))
            model.add(Dense(10, activation='softmax'))
            model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.RMSprop(),metrics=['accuracy'])
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

    def train(self,X,Y,saveTo,epochs=10):
        if (self.isRBF or self.isAnomalyDetector):
            checkpoint = ModelCheckpoint(saveTo, monitor='DistanceMetric', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)
        else:
            checkpoint = ModelCheckpoint(saveTo, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        self.model.fit(X, Y,
                batch_size=16,
                epochs=epochs,
                verbose=1,
                callbacks=[checkpoint],
                validation_split=0.2,
                shuffle=True)

    def save(self):
        raise NotImplementedError

    def load(self,weights):
        if (self.isRBF or self.isAnomalyDetector):
            self.model = load_model(weights, custom_objects={'RBFLayer': RBFLayer,'DistanceMetric':DistanceMetric,'RBF_Soft_Loss':RBF_Soft_Loss})
        else:
            self.model = load_model(weights)

    def transfer(self,weights,isRBF=False,anomalyDetector=False):
        self.isRBF = isRBF
        self.isAnomalyDetector=anomalyDetector
        assert not (self.isRBF and self.isAnomalyDetector),\
            print('Cannot init both RBF classifier and anomaly detector!')
        if (self.isRBF):
            self.model = load_model(weights, custom_objects={'RBFLayer': RBFLayer,'DistanceMetric':DistanceMetric,'RBF_Soft_Loss':RBF_Soft_Loss})
            for layer in self.model.layers[:-3]:
                layer.trainable = False  
            x = Dense(64, activation='tanh',kernel_initializer='random_uniform',bias_initializer='zeros')(self.model.layers[-3].output)
            x = RBFLayer(10,0.5)(x)
            self.model = Model(inputs=self.model.inputs, outputs=x)
            self.model.compile(loss=RBF_Soft_Loss,optimizer=keras.optimizers.Adam(),metrics=[DistanceMetric])
        elif(self.isAnomalyDetector):
            self.model = load_model(weights, custom_objects={'RBFLayer': RBFLayer,'DistanceMetric':DistanceMetric,'RBF_Soft_Loss':RBF_Soft_Loss})
            for layer in self.model.layers[:-3]:
                layer.trainable = False  
            x = Activation('tanh')(self.model.layers[-3].output)
            x = RBFLayer(10,0.5)(x)
            self.model = Model(inputs=self.model.inputs, outputs=x)
            self.model.compile(loss=RBF_Soft_Loss,optimizer=keras.optimizers.Adam(),metrics=[DistanceMetric])
        else:
            self.model = load_model(weights)
            for layer in self.model.layers[:-3]:
                layer.trainable = False            
            x = Dense(100, activation='relu',kernel_initializer='random_uniform',bias_initializer='zeros')(self.model.layers[-3].output)
            x = Dense(10, activation='softmax',kernel_initializer='random_uniform',bias_initializer='zeros')(x)
            self.model = Model(inputs=self.model.inputs, outputs=x)
            self.model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.RMSprop(),metrics=['accuracy'])
            model_noSoftMax = innvestigate.utils.model_wo_softmax(self.model) # strip the softmax layer
            self.analyzer = innvestigate.create_analyzer('lrp.alpha_1_beta_0', model_noSoftMax) # create the LRP analyzer

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
        print('Number of samples: ', len(Y))