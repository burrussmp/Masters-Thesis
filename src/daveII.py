from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import BatchNormalization,Input, add, concatenate,InputLayer,AveragePooling2D,MaxPooling2D,Convolution2D
from keras.layers.convolutional import Conv2D
from keras.layers import Activation, Flatten, Lambda, Input, ELU

import numpy as np
import keras.backend as K
from AdversarialAttacks import PhysicalAttackLanes
def DaveII():
    model = Sequential()
    input1= Input(shape=(66,200,3), name='image')
    steer_inp = BatchNormalization(epsilon=0.001, axis=-1,momentum=0.99)(input1)
    layer1 = Conv2D(24, (5, 5), padding="valid", strides=(2, 2), activation="relu")(steer_inp)
    layer2 = Conv2D(36, (5, 5), padding="valid", strides=(2, 2), activation="relu")(layer1)
    layer3 = Conv2D(48, (5, 5), padding="valid", strides=(2, 2), activation="relu")(layer2)
    layer4 = Conv2D(64, (3, 3), padding="valid", strides=(1, 1), activation="relu")(layer3)
    layer5 = Conv2D(64, (3, 3), padding="valid", strides=(1, 1), activation="relu")(layer4)
    layer6 = Flatten()(layer5)
    layer7 = Dense(1164, activation='relu')(layer6)
    layer8 = Dense(100, activation='relu')(layer7)
    layer9 = Dense(50, activation='relu')(layer8)
    layer10 = Dense(10, activation='relu')(layer9)
    prediction = Dense(10, name='predictions',activation="softmax")(layer10)
    model=Model(inputs=input1, outputs=prediction)
    return model

def DaveII_RBF():
    rbflayer = RBFLayer(10,0.5)
    model = Sequential()
    model.add(InputLayer(input_shape=(66,200,3), name='image'))
    model.add(BatchNormalization(epsilon=0.001, axis=-1,momentum=0.99))
    model.add(Conv2D(24, (5, 5), padding="valid", strides=(2, 2), activation="relu"))
    model.add(Conv2D(36, (5, 5), padding="valid", strides=(2, 2), activation="relu"))
    model.add(Conv2D(48, (5, 5), padding="valid", strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), padding="valid", strides=(1, 1), activation="relu"))
    model.add(Conv2D(64, (3, 3), padding="valid", strides=(1, 1), activation="relu"))
    #model.add(MaxPooling2D((2, 2),dim_ordering="th"))
    model.add(Flatten())
    model.add(Dense(500,activation='tanh'))
    model.add(Dense(100,activation='tanh'))
    model.add(RBFLayer(10,1.0))
    #model.add(Dense(10))
    # model.add(Dense(10,activation='tanh'))
    model=Model(inputs=model.input, outputs=model.output)
    return model


