"""
The script demonstrates a simple example of using ART with Keras. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import keras
from keras import losses
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import numpy as np
from keras.models import load_model
from art.attacks import FastGradientMethod
from art.classifiers import KerasClassifier
from art.utils import load_mnist
import keras
import keras.backend as K
import numpy as np
import cv2
from art.attacks import FastGradientMethod,DeepFool,CarliniL2Method,BasicIterativeMethod,ProjectedGradientDescent
from art.utils import load_mnist
from keras.preprocessing.image import ImageDataGenerator
from ModifiedKerasClassifier import KerasClassifier
from generate_model import generate_model
from art.metrics import metrics
from art.utils import to_categorical
import argparse
from sklearn.metrics import mean_squared_error
import numpy.linalg as la
from art.classifiers import KerasClassifier as DefaultKerasClassifier
from heatmap import LRP
import innvestigate
import innvestigate.utils
from art.defences import *
from sklearn.svm import OneClassSVM
import os
from rbflayer import RBFLayer, InitCentersRandom, InitCentersKMeans
from keras.optimizers import RMSprop
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import AgglomerativeClustering
VOCABLENGTH = 30
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, History

def FGSM(model,x,classes,epochs=40):
    x_adv = x
    x_noise = np.zeros_like(x)
    sess = K.get_session()
    preds = model.predict(x_adv)
    print('Initial prediction:', np.argmax(preds[0]))
    initial_class = np.argmax(preds[0])
    x_advcpy = x_adv
    epsilon = 0.01
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

def pairwisedifference(X,Y):
    D = np.zeros((X.shape[0],Y.shape[0]))
    i = 0
    for col1 in X:
        j = 0
        for col2 in Y:
            D[i,j] = (np.sum(np.square(col1-col2)))
            j += 1
        i += 1
    return D

def generate_heatmap(x,analyzer):
    if (type(x) == str):
        x = cv2.imread(x)
    x = np.expand_dims(x, axis=0)
    a = analyzer.analyze(x)
    a = a[0]
    a /= np.max(np.abs(a))
    heatmapshow = cv2.normalize(a, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
    heatmapshow = heatmapshow.astype(np.float64)
    heatmapshow = heatmapshow.sum(axis=np.argmax(np.asarray(heatmapshow.shape) == 3))
    heatmapshow /= np.max(np.abs(heatmapshow))
    return heatmapshow

def getSiftFeatures(sift,img):
    if (len(img.shape) == 2):
        img = np.stack((img,)*3, axis=-1)
    elif(len(img.shape) == 3 and img.shape[2] == 1):
        img = np.squeeze(img)
        img = np.stack((img,)*3, axis=-1)
    img= cv2.cvtColor((img*255).astype(np.uint8),cv2.COLOR_BGR2GRAY)
    keypoints_sift, descriptors = sift.detectAndCompute(img, None)
    return descriptors

from sklearn.cluster import KMeans,MiniBatchKMeans
def CodeBookGeneration(features,batch=False,hierarchical = False):
    kmeans = None
    if (hierarchical):
        codebook = AgglomerativeClustering().fit(features[0:30000])
    else:
        if (batch):
            batch_size = int(features.shape[0]/1000)
            kmeans = MiniBatchKMeans(n_clusters=VOCABLENGTH, batch_size=batch_size,verbose=1).fit(features)
        else:
            kmeans = KMeans(n_clusters=VOCABLENGTH,verbose=1).fit(features)
        # get the features for a given class
        codebook = kmeans.cluster_centers_
    return codebook,kmeans

# Step 1: Load the MNIST dataset
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
# input_shape = (x_train.shape[1],x_train.shape[2],x_train.shape[3])
# print(input_shape)
# x_train = np.moveaxis(x_train,0,-1)
# y_train = np.moveaxis(y_train,0,-1)
# x_train = np.zeros((100,200))
# y_train = np.zeros((100,1))
# model = Sequential()
# rbflayer = RBFLayer(30,
#                     initializer=InitCentersRandom(x_train),
#                     betas=2.0,
#                     input_shape=(200,))
# model.add(rbflayer) # add the rbf layer
# model.add(Dense(y_train.shape[1])) # number of classes

# model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=0.01), metrics=['categorical_accuracy'])


# model.fit(x_train, y_train,
#             batch_size=1,
#             epochs=4,F
#             verbose=1)

# Step 2: Create the model
print('Designing the normal model')
model = Sequential()
model.add(Conv2D(filters=4, kernel_size=(5, 5), strides=1, activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=10, kernel_size=(5, 5), strides=1, activation='relu', input_shape=(23, 23, 4)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=0.01), metrics=['accuracy'])

print('Setting up LRP...')
model_noSoftMax = innvestigate.utils.model_wo_softmax(model) # strip the softmax layer
analyzer = innvestigate.create_analyzer('deep_taylor', model_noSoftMax) # create the LRP analyzer

print('Setting up SIFT...')
sift = cv2.xfeatures2d.SIFT_create()

baseDir = '/media/burrussmp/99e21975-0750-47a1-a665-b2522e4753a6/weights/'

print('Converting train images to heatmap')
train_heat = 'x_train_heat.npy'
path_to_train_heat = os.path.join(baseDir,train_heat)
if os.path.isfile(path_to_train_heat):
    x_train_heat = np.load(path_to_train_heat)
else:
    x_train_heat = np.zeros_like(x_train)
    for i in range(x_train.shape[0]):
        a = generate_heatmap(x_train[i],analyzer)
        print(i,x_train.shape[0])
        x_train_heat[i] = np.expand_dims(a, axis=2)
    print('Saving heat test: ', path_to_train_heat)
    np.save(path_to_train_heat,x_train_heat)

print('Done loading the train heat images')

print('Converting train images to SIFT features and saving them')
train_sift = 'x_train_sift.npy'
path_to_train_sift = os.path.join(baseDir,train_sift)
x_train_sift = np.array([])
if os.path.isfile(path_to_train_sift):
    x_train_sift = np.load(path_to_train_sift)
else:
    for i in range(x_train.shape[0]):
        img = np.copy(x_train[i])
        descriptors = getSiftFeatures(sift,img)
        if (not isinstance(descriptors,np.ndarray)):
            continue
        if (x_train_sift.shape[0] == 0):
            x_train_sift = np.array(descriptors)
        else:
            x_train_sift = np.concatenate((x_train_sift,descriptors))
        print(i,x_train.shape[0])
    print('Saving heat test: ', path_to_train_sift)
    np.save(path_to_train_sift,x_train_sift)

print('Done loading the training SIFT features')
print(x_train_sift.shape)

print('Converting train images to SIFT features and saving them')
train_sift_heat = 'x_train_sift_heat.npy'
path_to_train_sift_heat = os.path.join(baseDir,train_sift_heat)
x_train_sift_heat = np.array([])
if os.path.isfile(path_to_train_sift_heat):
    x_train_sift_heat = np.load(path_to_train_sift_heat)
else:
    for i in range(x_train_heat.shape[0]):
        img = np.copy(x_train_heat[i])
        descriptors = getSiftFeatures(sift,img)
        if (not isinstance(descriptors,np.ndarray)):
            continue
        if (x_train_sift_heat.shape[0] == 0):
            x_train_sift_heat = np.array(descriptors)
        else:
            x_train_sift_heat = np.concatenate((x_train_sift_heat,descriptors))
        print(i,x_train_heat.shape[0])
    print('Saving heat test: ', path_to_train_sift_heat)
    np.save(path_to_train_sift_heat,x_train_sift_heat)

print('Done loading the training SIFT features')
print(x_train_sift_heat.shape)


test = '/media/burrussmp/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/vgg16_dataset_partitioned/train/n01498041/n01498041_129.JPEG'
test2 = '/media/burrussmp/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/vgg16_dataset_partitioned/heatmap_train/n01498041/heatmap_n01498041_129.JPEG'
descriptors = getSiftFeatures(sift,cv2.imread(test))
descriptors2 = getSiftFeatures(sift,cv2.imread(test2))
print('Heat shape: ', descriptors2.shape)
print('Regular shape: ', descriptors.shape)

print('Generate the codebook')

codebook_path = os.path.join(baseDir,'codebook30.npy')
if os.path.isfile(codebook_path):
    codebook = np.load(codebook_path,allow_pickle=True)
else:
    codebook, kmeans = CodeBookGeneration(x_train_sift_heat,batch=True)
    print('Saving the codebook',codebook_path)
    np.save(codebook_path,np.array(codebook))
print('Loaded the codebook')

print('Computing the Bag Of Sift for the x_train_heat')
bag_of_sift_x_train_heat_path = os.path.join(baseDir,'BOW_x_train30.npy')
if os.path.isfile(bag_of_sift_x_train_heat_path):
    bagofsift_train = np.load(bag_of_sift_x_train_heat_path)
else:
    bagofsift_train = np.zeros((x_train_heat.shape[0],VOCABLENGTH))
    for i in range(x_train_heat.shape[0]):
        # try:
        img = np.copy(x_train_heat[i])
        descriptors = getSiftFeatures(sift,img)
        # find pairwise difference
        if (not isinstance(descriptors,np.ndarray)):
            continue
        D = pairwisedifference(codebook,descriptors)
        # find minimum distance vocab word for each feature
        min_indices = np.argmin(D.T,axis=1)
        [hist,edges] = np.histogram(min_indices,bins=VOCABLENGTH,range=(0,VOCABLENGTH),density=True)
        bagofsift_train[i,:] = hist
        print(i,x_train_heat.shape[0])
        # except:
        #     print('Error: skipping')
    np.save(bag_of_sift_x_train_heat_path,bagofsift_train)

print('Training the RBF model')
path_to_model_weights = os.path.join(baseDir,'rbf_weights.h5')
model = Sequential()
rbflayer = RBFLayer(160,
                    initializer=InitCentersRandom(np.zeros((160,160))),
                    betas=2.0,
                    input_shape=(160,1))
model.add(Conv2D(filters=4, kernel_size=(5, 5), strides=1, activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=10, kernel_size=(5, 5), strides=1, activation='relu', input_shape=(23, 23, 4)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(rbflayer)
model.add(Dense(10))
model.compile(loss=losses.mean_squared_error,optimizer=RMSprop())
model.summary()
model = load_model(path_to_model_weights, custom_objects={'RBFLayer': RBFLayer})
checkpoint = ModelCheckpoint(path_to_model_weights, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
model.fit(x_train_heat, y_train,
        batch_size=128,
        epochs=100,
        verbose=1,
        callbacks=[checkpoint],
        validation_split=0.2)

path_to_model_weights2 = os.path.join(baseDir,'rbf_weights2.h5')
model2 = Sequential()
rbflayer = RBFLayer(160,
                    initializer=InitCentersRandom(np.zeros((160,160))),
                    betas=2.0,
                    input_shape=(160,1))
model2.add(Conv2D(filters=4, kernel_size=(5, 5), strides=1, activation='relu', input_shape=(28, 28, 1)))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(filters=10, kernel_size=(5, 5), strides=1, activation='relu', input_shape=(23, 23, 4)))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Flatten())
model2.add(rbflayer)
model2.add(Dense(10))
model2.compile(loss=losses.mean_squared_error,optimizer=RMSprop())
model2.summary()
model2 = load_model(path_to_model_weights2, custom_objects={'RBFLayer': RBFLayer})
checkpoint = ModelCheckpoint(path_to_model_weights2, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
model2.fit(x_train, y_train,
        batch_size=128,
        epochs=100,
        verbose=1,
        callbacks=[checkpoint],
        validation_split=0.2)
xadv = FGSM(model2,np.expand_dims(x_test[0],axis=0),10,3)
# model = GaussianNB()
# print(np.argmax(y_train,axis=1).shape)
# print(bagofsift_train.shape)
# model.fit(bagofsift_train,np.argmax(y_train,axis=1))

print('Designing the normal model')
regPath = os.path.join(baseDir,'reg_weights.h5')
reg = Sequential()
reg.add(Conv2D(filters=4, kernel_size=(5, 5), strides=1, activation='relu', input_shape=(28, 28, 1)))
reg.add(MaxPooling2D(pool_size=(2, 2)))
reg.add(Conv2D(filters=10, kernel_size=(5, 5), strides=1, activation='relu', input_shape=(23, 23, 4)))
reg.add(MaxPooling2D(pool_size=(2, 2)))
reg.add(Flatten())
reg.add(Dense(100, activation='relu'))
reg.add(Dense(10, activation='softmax'))
reg.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=0.01), metrics=['accuracy'])
checkpoint = ModelCheckpoint(regPath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
reg = load_model(regPath)
reg.fit(x_train_heat, y_train,
        batch_size=128,
        epochs=100,
        verbose=1,
        callbacks=[checkpoint],
        validation_split=0.2)


classifier = DefaultKerasClassifier(defences=[],model=reg, clip_values=(min_pixel_value, max_pixel_value), use_logits=False)
classifier2 = DefaultKerasClassifier(defences=[],model=model2, clip_values=(min_pixel_value, max_pixel_value), use_logits=False)

print('Designing adversarial test images...')
adv = 'x_test_adv_mnist.npy'
path_to_adv = os.path.join(baseDir,adv)
if os.path.isfile(path_to_adv):
    x_test_adv = np.load(path_to_adv)
else:
    attack = FastGradientMethod(classifier=classifier, eps=0.2)
    attack.set_params(**{'minimal': True})
    x_test_adv = attack.generate(x=np.copy(x_test))
    np.save(path_to_adv,x_test_adv)
    print('Saved x_test_adv: ', path_to_adv)

print('Converting adversarial images to heatmap')

adv_heat = 'x_test_adv_heat_mnist.npy'
path_to_adv_heat = os.path.join(baseDir,adv_heat)
if os.path.isfile(path_to_adv_heat):
    x_test_adv_heat = np.load(path_to_adv_heat)
else:
    x_test_adv_heat = np.zeros_like(x_test_adv)
    for i in range(x_test_adv.shape[0]):
        a = generate_heatmap(x_test_adv[i],analyzer)
        print(i,x_test_adv.shape[0])
        x_test_adv_heat[i] = np.expand_dims(a, axis=2)
    print('Saving adversary heat test: ', path_to_adv_heat)
    np.save(path_to_adv_heat,x_test_adv_heat)

print('Done loading the adversarial heatmap images')

print('Converting test images to heatmap')
test_heat = 'x_test_heat_mnist.npy'
path_to_test_heat = os.path.join(baseDir,test_heat)
if os.path.isfile(path_to_test_heat):
    x_test_heat = np.load(path_to_test_heat)
else:
    x_test_heat = np.zeros_like(x_test)
    for i in range(x_test.shape[0]):
        a = generate_heatmap(x_test[i],analyzer)
        print(i,x_test.shape[0])
        x_test_heat[i] = np.expand_dims(a, axis=2)
    print('Saving heat test: ', path_to_test_heat)
    np.save(path_to_test_heat,x_test_heat)

print('Done loading the test heat images')

predictions = classifier.predict(x_test_adv)
predictions2 = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
accuracy2 = np.sum(np.argmax(predictions2, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print('Accuracy on adversarial test examples: {}%'.format(accuracy * 100))
print('Accuracy on test examples: {}%'.format(accuracy2 * 100))

predictions = model2.predict(x_test_adv)
predictions2 = model2.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
accuracy2 = np.sum(np.argmax(predictions2, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print('Accuracy on adversarial test examples: {}%'.format(accuracy * 100))
print('Accuracy on test examples: {}%'.format(accuracy2 * 100))

predictions = model.predict(x_test_adv_heat)
predictions2 = model.predict(x_test_heat)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
accuracy2 = np.sum(np.argmax(predictions2, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print('Accuracy on adversarial test examples: {}%'.format(accuracy * 100))
print('Accuracy on test examples: {}%'.format(accuracy2 * 100))
print('Evaluate the test dataset')

exit(1)
# predictions = heat.predict(x_test_adv_heat)
# predictions2 = heat.predict(x_test_heat)
# accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
# accuracy2 = np.sum(np.argmax(predictions2, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
# print('Accuracy on adversarial test examples: {}%'.format(accuracy * 100))
# print('Accuracy on test examples: {}%'.format(accuracy2 * 100))
# print('Evaluate the test dataset')


# attack the model
print('Designing adversarial heatmap test images...')
adv = 'x_test_heat_adv_mnist.npy'
path_to_adv = os.path.join(baseDir,adv)
if os.path.isfile(path_to_adv):
    x_test_heat_adv = np.load(path_to_adv)
else:
    attack = FastGradientMethod(classifier=classifier, eps=0.2)
    attack.set_params(**{'minimal': True})
    x_test_heat_adv = attack.generate(x=np.copy(x_test))
    np.save(path_to_adv,x_test_heat_adv)
    print('Saved x_test_adv: ', path_to_adv)



path_to_x_test_heat = 'x_test_heat30.npy'
path_to_x_test_heat = os.path.join(baseDir,path_to_x_test_heat)
if os.path.isfile(path_to_x_test_heat):
    bagofsift_test_heat = np.load(path_to_x_test_heat)
else:
    bagofsift_test_heat = np.zeros((x_test.shape[0],VOCABLENGTH))
    for i in range(x_test.shape[0]):
        img = np.copy(x_test[i])
        a = generate_heatmap(img,analyzer)
        img = np.expand_dims(a, axis=2)
        descriptors = getSiftFeatures(sift,img)
        if (not isinstance(descriptors,np.ndarray)):
            continue
        D = pairwisedifference(codebook,descriptors)
        min_indices = np.argmin(D.T,axis=1)
        [hist,edges] = np.histogram(min_indices,bins=VOCABLENGTH,range=(0,VOCABLENGTH),density=True)
        bagofsift_test_heat[i,:] = hist
        print(i,x_test.shape[0])
    np.save(path_to_x_test_heat,bagofsift_test_heat)
    
print('Evaluating acccuracy on heat')
predictions = model.predict(bagofsift_test_heat)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
accuracy = np.sum(predictions == np.argmax(y_test, axis=1)) / len(y_test)
print('Accuracy on test examples: {}%'.format(accuracy * 100))