
import keras.backend as K
import numpy as np
import keras
from keras.models import Model,load_model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization,Lambda
from keras.layers import Conv2D, MaxPooling2D,Input,AveragePooling2D
import os
import cv2
import innvestigate
import innvestigate.utils
from .RBFLayer import RBFLayer
from .ResNetLayer import ResNetLayer
from .Losses import RBF_Soft_Loss,RBF_Loss,DistanceMetric,RBF_LAMBDA
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, History,ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator


class ResNetV1():
    def __init__(self,input_shape=(32,32,3),depth=20,num_classes=10,RBF=False,anomalyDetector=False):
        self.input_size = input_shape
        self.num_classes = num_classes
        self.isRBF = RBF
        self.isAnomalyDetector = anomalyDetector
        assert not (self.isRBF and self.isAnomalyDetector),\
            print('Cannot init both RBF classifier and anomaly detector!')
        if (depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
        num_filters = 16
        num_res_blocks = int((depth - 2) / 6)
        inputs = Input(shape=input_shape)
        x = ResNetLayer(inputs=inputs)
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:
                    strides = 2 
                y = ResNetLayer(inputs=x,
                                num_filters=num_filters,
                                strides=strides)
                y = ResNetLayer(inputs=y,
                                num_filters=num_filters,
                                activation=None)
                if stack > 0 and res_block == 0:
                    x = ResNetLayer(inputs=x,
                                    num_filters=num_filters,
                                    kernel_size=1,
                                    strides=strides,
                                    activation=None,
                                    batch_normalization=False)
                x = keras.layers.add([x, y])
                x = Activation('relu')(x)
            num_filters *= 2
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        if (RBF):
            outputs = Dense(64,activation='tanh')(y)
            outputs = RBFLayer(10,0.5)(outputs)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(loss=RBF_Soft_Loss,optimizer=keras.optimizers.Adam(),metrics=[DistanceMetric])
        elif(anomalyDetector):
            outputs = Activation('tanh')(y)
            outputs = RBFLayer(10,0.5)(outputs)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(loss=RBF_Soft_Loss,optimizer=keras.optimizers.Adam(),metrics=[DistanceMetric])
        else:
            #outputs = Dense(32,activation='relu')(y)
            outputs = Dense(10,activation='softmax')(y)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
            model_noSoftMax = innvestigate.utils.model_wo_softmax(model) # strip the softmax layer
            self.analyzer = innvestigate.create_analyzer('deep_taylor', model_noSoftMax) # create the LRP analyzer
        self.model = model

    def transfer(self,weights,isRBF=False,anomalyDetector=False):
        self.isRBF = isRBF
        self.isAnomalyDetector = anomalyDetector
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
        elif (self.isAnomalyDetector):
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
            x = Dense(10, activation='softmax',kernel_initializer='random_uniform',bias_initializer='zeros')(x)
            self.model = Model(inputs=self.model.inputs, outputs=x)
            self.model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.RMSprop(),metrics=['accuracy'])
            model_noSoftMax = innvestigate.utils.model_wo_softmax(self.model) # strip the softmax layer
            self.analyzer = innvestigate.create_analyzer('lrp.alpha_1_beta_0', model_noSoftMax) # create the LRP analyzer

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

    def train(self,X,Y,saveTo,epochs=100):
        def lr_schedule(epoch):
            lr = 1e-3
            if epoch > 180:
                lr *= 0.5e-3
            elif epoch > 160:
                lr *= 1e-3
            elif epoch > 120:
                lr *= 1e-2
            elif epoch > 80:
                lr *= 1e-1
            print('Learning rate: ', lr)
            return lr
        lr_scheduler = LearningRateScheduler(lr_schedule)
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
            cooldown=0,
            patience=5,
            min_lr=0.5e-6)

        if (self.isRBF or self.isAnomalyDetector):
            checkpoint = ModelCheckpoint(saveTo, monitor='DistanceMetric', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)
        else:
            checkpoint = ModelCheckpoint(saveTo, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        callbacks = [checkpoint, lr_reducer, lr_scheduler]
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X)
        idx = int(0.8*(Y.shape[0]))-1
        x_test = X[idx::]
        y_test = Y[idx::]
        X = X[0:idx]
        Y = Y[0:idx]
        # Fit the model on the batches generated by datagen.flow().
        self.model.fit_generator(datagen.flow(X, Y, batch_size=16),
                            validation_data=(x_test, y_test),
                            epochs=epochs, verbose=1,steps_per_epoch=int(Y.shape[0]/16),
                            callbacks=callbacks)

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
        print('Number of samples: ', len(Y))