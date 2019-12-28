from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization,Lambda
from keras.layers import Conv2D, MaxPooling2D,Input,AveragePooling2D
import os
from rbflayer import RBFLayer, InitCentersRandom, InitCentersKMeans
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

def softargmax(x,beta=1e10):
    x = tf.convert_to_tensor(x)
    x_range = tf.range(10)
    x_range = tf.dtypes.cast(x_range,tf.float32)
    return tf.reduce_sum(tf.nn.softmax(x*beta) * x_range, axis=1)
def loss(y_true,y_pred):
    lam = 0.5
    indices = softargmax(y_true)
    indices = tf.dtypes.cast(indices,tf.int32)
    y_pred = tf.dtypes.cast(y_pred,tf.float32)
    y_true = tf.dtypes.cast(y_true,tf.float32)
    row_ind = K.arange(K.shape(y_true)[0])
    full_indices = tf.stack([row_ind,indices],axis=1)
    d = tf.gather_nd(y_pred,full_indices)
    y_pred = lam - y_pred
    y_pred = tf.nn.relu(y_pred)
    d2 = tf.nn.relu(lam - d)
    S = K.sum(y_pred,axis=1) - d2
    y = K.sum(d + S)
    return y

def soft_loss(y_true,y_pred):
    lam = 0.5
    indices = softargmax(y_true)
    indices = tf.dtypes.cast(indices,tf.int32)
    y_pred = tf.dtypes.cast(y_pred,tf.float32)
    y_true = tf.dtypes.cast(y_true,tf.float32)
    row_ind = K.arange(K.shape(y_true)[0])
    full_indices = tf.stack([row_ind,indices],axis=1)
    d = tf.gather_nd(y_pred,full_indices)
    y_pred = K.log(1+ K.exp(lam - y_pred))
    S = K.sum(y_pred,axis=1) - K.log(1+K.exp(lam-d))
    y = K.sum(d + S)
    return y

# each thing ranges from 0 to 1. 1 is good, zero is bad.
# SUM(1-correct + sum(others)
def my_loss(y_true,y_pred):
    indices = softargmax(y_true)
    indices = tf.dtypes.cast(indices,tf.int32)
    y_pred = tf.dtypes.cast(y_pred,tf.float32)
    y_true = tf.dtypes.cast(y_true,tf.float32)
    row_ind = K.arange(K.shape(y_true)[0])
    full_indices = tf.stack([row_ind,indices],axis=1)
    d = tf.gather_nd(y_pred,full_indices)
    return K.sum(K.sum(y_pred,axis=1) + 1 - 2*d)

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


n = 3
# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 1

# Computed depth from supplied model parameter n
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
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

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v1(input_shape, depth, num_classes=10):

    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    # outputs = Dense(num_classes,
    #     activation='softmax',
    #     kernel_initializer='he_normal')(y)
    outputs = Dense(64,activation='tanh')(y)
    outputs = RBFLayer(10,0.5)(outputs)
    # outputs = Dense(10)(outputs)
    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model
def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)
    # outputs = RBFLayer(1000,1.0)(y)
    # outputs = Dense(10)(outputs)
    #outputs = Dense(10)(outputs)
    #outputs = Dense(10,activation='tanh')(outputs)
    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def loadModel(weights=None,RBF=False,transfer=False,X=None):
    model = None
    # rbflayer = RBFLayer(10,
    #                     initializer=None,
    #                     betas=1.0,
    #                     input_shape=(10,1))
    if weights == None or not os.path.isfile(weights):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((4, 4)))
        model.add(Flatten())
        # weight_decay = 1e-4
        # model = Sequential()
        # model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=(32,32,3)))
        # model.add(Activation('elu'))
        # model.add(BatchNormalization())
        # model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        # model.add(Activation('elu'))
        # model.add(BatchNormalization())
        # model.add(MaxPooling2D(pool_size=(2,2)))
        # model.add(Dropout(0.2))
        
        # model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        # model.add(Activation('elu'))
        # model.add(BatchNormalization())
        # model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        # model.add(Activation('elu'))
        # model.add(BatchNormalization())
        # model.add(MaxPooling2D(pool_size=(2,2)))
        # model.add(Dropout(0.3))
        
        # model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        # model.add(Activation('elu'))
        # model.add(BatchNormalization())
        # model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        # model.add(Activation('elu'))
        # model.add(BatchNormalization())
        # model.add(MaxPooling2D(pool_size=(4,4)))
        # model.add(Dropout(0.4))
        
        # model.add(Flatten())
        #model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        if (not RBF):
            model.add(Dense(10, activation='softmax'))
            opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)
            model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.RMSprop(),metrics=['accuracy'])
        else:
            #model.add(Dense(128,activation='relu'))
            model.add(Dense(64,activation='tanh'))
            model.add(RBFLayer(10,1))
            # if version == 2:
            #     model = resnet_v2(input_shape=(32,32,3), depth=depth)
            # else:
            #     model = resnet_v1(input_shape=(32,32,3), depth=depth)
            # input()
            #model.add(rbflayer)
            # model.add(Dense(10))
            opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)
            #model.compile(loss=loss,optimizer=keras.optimizers.RMSprop(),metrics=[custom_accuracy,'accuracy'])
            model.compile(loss=soft_loss,optimizer=keras.optimizers.Adam(),metrics=[custom_accuracy,'accuracy'])
            # model.add(Dense(128,activation='relu'))
            # model.add(Dense(128,activation='relu'))
            # model.add(rbflayer)
            # model.add(Dense(10))
            # model.summary()
            #model.compile(loss='mse',optimizer=keras.optimizers.RMSprop(),metrics=['accuracy'])
    elif (os.path.isfile(weights)):
        if (not RBF):
            model = load_model(weights)
            model.layers[-1].name = 'prediction'
            model.layers[-2].name = 'penultimate'
        else:
            model = load_model(weights, custom_objects={'RBFLayer': RBFLayer,'custom_accuracy':custom_accuracy,'soft_loss':soft_loss})
    assert model != None, \
        'ERROR: Model should not be None type'
    return model

def defense(bm,rbf,x,Y,x_heat=None):
    p_bm = bm.predict(x)
    prediction_bm = p_bm[np.arange(p_bm.shape[0]),np.argmax(p_bm,axis=1)]
    if (x_heat != None):
        p_rbf = rbf.predict(x_heat)
    else:
        p_rbf = rbf.predict(x)
    # turn rbf predictions to probs
    lam = 0.5
    Ok = np.exp(-1*p_rbf)
    top = Ok*(1+np.exp(lam)*Ok)
    bottom = np.prod(1+np.exp(lam)*Ok,axis=1)
    p_rbf = np.divide(top.T,bottom).T
    predictions = np.zeros((p_rbf.shape[0],1))

    prediction_rbf = p_rbf[np.arange(p_rbf.shape[0]),np.argmax(p_rbf,axis=1)]
    thresh = 0.1
    bool_arr = prediction_rbf > thresh
    


def generate_heatmap(x,analyzer):
    if (type(x) == str):
        x = cv2.imread(x)
    x = np.expand_dims(x, axis=0)
    a = analyzer.analyze(x)
    a = a[0]
    a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
    a /= np.max(np.abs(a))+1e-6
    a = (a*255).astype(np.uint8)
    #heatmapshow = cv2.normalize(a, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmapshow = cv2.applyColorMap(a, cv2.COLORMAP_JET)
    heatmapshow = heatmapshow.astype(np.float32)
    # cv2.imshow('heat',heatmapshow.astype(np.uint8))
    # cv2.waitKey(1000)
    return heatmapshow

def convertToHeatMap(X,analyzer,path=None,visual = False):
    heat_data = np.array([])
    if os.path.isfile(path):
        heat_data = np.load(path)
    else:
        heat_data = np.zeros_like(X)
        for i in range(X.shape[0]):
            a = generate_heatmap(X[i],analyzer)
            if (visual):
                cv2.imshow('a',astype(np.uint8))
                cv2.waitKey(1000)
            heat_data[i] = a#np.expand_dims(a, axis=2)
            print(i,X.shape[0])
        if (path != None):
            print('Saving adversary heat test: ', path)
            np.save(path,heat_data)  
    return heat_data

def createAdversary(model,X,path=None):
    classifier = DefaultKerasClassifier(defences=[],model=model, use_logits=False)
    print('Designing adversarial images...')
    if os.path.isfile(path):
        xadv = np.load(path)
    else:
        # xadv = np.zeros_like(X)
        # for i in range(X.shape[0]):
        #     print('Progerss',i,X.shape[0])
        #     xadv[i] = FGSM(model,np.expand_dims(X[i],axis=0),classes=10)
        attack = FastGradientMethod(classifier=classifier, eps=0.09)
        #attack.set_params(**{'minimal': True})
        xadv = attack.generate(x=np.copy(X))
        if (path != None):
            np.save(path,xadv)
            print('Saved x_test_adv: ', path)
    return xadv

def train(model,X,Y,path):
    checkpoint = ModelCheckpoint(path, monitor='custom_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    model.fit(X, Y,
            batch_size=16,
            epochs=100,
            verbose=1,
            callbacks=[checkpoint],
            validation_split=0.2,
            shuffle=True)
    return model

def test(model,X,Y):
    predictions = model.predict(X)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(Y, axis=1)) / len(Y)
    print('Accuracy on test examples: {}%'.format(accuracy * 100))

def testRBF(model,X,Y):
    predictions = model.predict(X)
    # find probabilities
    lam = 0.5
    Ok = np.exp(-1*predictions)
    top = Ok*(1+np.exp(lam)*Ok)
    bottom = np.prod(1+np.exp(lam)*Ok,axis=1)
    probs = np.divide(top.T,bottom).T
    accuracy = np.sum(np.argmax(probs, axis=1) == np.argmax(Y, axis=1)) / len(Y)
    print('Accuracy on test examples: {}%'.format(accuracy * 100))
    return probs

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

def RBFprob(predictions):
    lam = 0.5
    Ok = np.exp(-1*predictions)
    top = Ok*(1+np.exp(lam)*Ok)
    bottom = np.prod(1+np.exp(lam)*Ok,axis=1)
    return np.divide(top.T,bottom).T

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# preprocess the data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
baseDir = '/media/burrussmp/99e21975-0750-47a1-a665-b2522e4753a6/weights/CIFAR10'

# train the first model
#basemodel = loadModel(weights=os.path.join(baseDir,'basemodel.h5'),RBF=False)
#basemodel = train(basemodel,x_train,y_train,path = os.path.join(baseDir,'basemodel.h5'))

# load and test the first model
basemodel = loadModel(weights=os.path.join(baseDir,'basemodel_resnet.h5'),RBF=False,X=x_train)
# subModel = Model(basemodel.input, basemodel.layers[-3].output)
# prediction = subModel.predict(x_train)
# print(prediction.shape)
# np.save(os.path.join(baseDir,'features.npy'),prediction)

#basemodel.summary()
train(basemodel,x_train,y_train,path=os.path.join(baseDir,'basemodel.h5'))
test(basemodel,x_test,y_test)
input()


# creat analyzer


model_noSoftMax = innvestigate.utils.model_wo_softmax(basemodel) # strip the softmax layer
analyzer = innvestigate.create_analyzer('deep_taylor', model_noSoftMax) # create the LRP analyzer

# create heatmap
convertToHeatMap(x_train,analyzer,path=os.path.join(baseDir,'x_train_heat.npy'),visual=True)
#convertToHeatMap(x_test,analyzer,path=os.path.join(baseDir,'x_test_heat.npy'),visual=False)

# train the rbf model on original
rbf_reg_model = loadModel(weights=os.path.join(baseDir,'rbf_reg_soft.h5'),RBF=True,transfer=True,X=x_train)


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
plt.show()