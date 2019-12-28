"""
The script demonstrates a simple example of using ART with Keras. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import numpy as np

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
from rbflayer import RBFLayer, InitCentersRandom
from keras.optimizers import RMSprop

def generate_heatmap(x,analyzer):
    if (type(x) == str):
        x = cv2.imread(x)
    x = np.expand_dims(x, axis=0)
    a = analyzer.analyze(x)
    a = a[0]
    #a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
    a /= np.max(np.abs(a))
    heatmapshow = cv2.normalize(a, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
    heatmapshow = heatmapshow.astype(np.float64)
    heatmapshow = heatmapshow.sum(axis=np.argmax(np.asarray(heatmapshow.shape) == 3))
    heatmapshow /= np.max(np.abs(heatmapshow))
    return heatmapshow

def getSiftFeatures(sift,img):
    img = np.stack((img,)*3, axis=-1)
    img= cv2.cvtColor((img*255).astype(np.uint8),cv2.COLOR_BGR2GRAY)
    keypoints_sift, descriptors = sift.detectAndCompute(img, None)
    return descriptors


# Step 1: Load the MNIST dataset
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

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



print('Designing the heat model')
heat = Sequential()
heat.add(Conv2D(filters=4, kernel_size=(5, 5), strides=1, activation='relu', input_shape=(28, 28, 1)))
heat.add(MaxPooling2D(pool_size=(2, 2)))
heat.add(Conv2D(filters=10, kernel_size=(5, 5), strides=1, activation='relu', input_shape=(23, 23, 4)))
heat.add(MaxPooling2D(pool_size=(2, 2)))
heat.add(Flatten())
heat.add(Dense(100, activation='relu'))
heat.add(Dense(10, activation='softmax'))
heat.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=0.01), metrics=['accuracy'])


# load the weights for the regular model or train if not trained
baseDir = '/media/burrussmp/99e21975-0750-47a1-a665-b2522e4753a6/weights/'
assert os.path.exists(baseDir), \
    'Base directory does not exist!'

weights = os.path.join(baseDir,'mnist.h5')
if os.path.isfile(weights):
    model.load_weights(weights)
else:
    model.fit(x_train, y_train, batch_size=64, epochs=4)
    model.save_weights(weights)

# load the weights for the heat or train if not trained
weights_heat = os.path.join(baseDir,'mnistheattrain.h5')
if os.path.isfile(weights_heat):
    heat.load_weights(weights_heat)
else:
    print('Creating the head dataset to train the heat model')
    x_heat_train = np.zeros_like(x_train)
    for i in range(x_train.shape[0]):
        a = generate_heatmap(x_train[i],analyzer)
        print(i,x_train.shape[0])
        x_heat_train[i] = np.expand_dims(a, axis=2)
    heat.fit(x_heat_train, y_train, batch_size=64, epochs=4)
    heat.save_weights(weights_heat)

# for i in range(x_heat_train.shape[0]):
#     cv2.imshow('a',x_heat_train[i])
#     cv2.waitKey(1000)
x_testcpy = np.copy(x_test)
defense = []
useDefense = True
if (useDefense):
    ss = SpatialSmoothing(window_size=3)
    tv = TotalVarMin()
    fs = FeatureSqueezing(clip_values=(min_pixel_value, max_pixel_value))
    # defense.append(ss)
    # defense.append(fs)

print('Creating classifiers...')
classifier = DefaultKerasClassifier(defences=defense,model=model, clip_values=(min_pixel_value, max_pixel_value), use_logits=False)
classifierHeat = DefaultKerasClassifier(model=heat, clip_values=(min_pixel_value, max_pixel_value), use_logits=False)
# Create adversary images
print('Designing adversarial test images...')
adv = 'x_test_adv_mnist.npy'
path_to_adv = os.path.join(baseDir,adv)
if os.path.isfile(path_to_adv):
    x_test_adv = np.load(path_to_adv)
else:
    attack = FastGradientMethod(classifier=classifier, eps=0.2)
    attack.set_params(**{'minimal': True})
    x_test_adv = attack.generate(x=x_test)
    np.save(path_to_adv,x_test_adv)
    print('Saved x_test_adv: ', path_to_adv)

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

print('\n\nEvaluating the base classifier')
# compute the accuracy on the adversary images and the regular images
predictions = classifier.predict(x_test_adv)
predictions2 = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
accuracy2 = np.sum(np.argmax(predictions2, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print('Accuracy on adversarial test examples: {}%'.format(accuracy * 100))
print('Accuracy on test examples: {}%'.format(accuracy2 * 100))

print('Computing the robustness of the model without any defense...')
# compute the empirical robustness score of the regular classifier
idxs = (np.argmax(predictions, axis=1) != np.argmax(y_test, axis=1))
if np.sum(idxs) == 0.0:
    exit(0)
norm_type = 2
perts_norm = la.norm((x_test_adv - x_test).reshape(x_test.shape[0], -1), ord=norm_type, axis=1)
perts_norm = perts_norm[idxs]
robust= np.mean(perts_norm / la.norm(x_test[idxs].reshape(np.sum(idxs), -1), ord=norm_type, axis=1))
print('Empirical robustness: {}%'.format(robust))


print('\n\nEvaluating the classifier with the defense')
predictions = classifierHeat.predict(x_test_adv_heat)
predictions2 = classifierHeat.predict(x_test_heat)
# for i in range(x_test_heat.shape[0]):
#     cv2.imshow('a',x_test_adv_heat[i])
#     cv2.waitKey(1000)

accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
accuracy2 = np.sum(np.argmax(predictions2, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print('Accuracy on adversarial test examples: {}%'.format(accuracy * 100))
print('Accuracy on test examples: {}%'.format(accuracy2 * 100))

print('Computing the robustness of the model with defense...')
# compute the empirical robustness score of the regular classifier
idxs = (np.argmax(predictions, axis=1) != np.argmax(y_test, axis=1))
if np.sum(idxs) == 0.0:
    exit(0)
norm_type = 2
perts_norm = la.norm((x_test_adv - x_test).reshape(x_test.shape[0], -1), ord=norm_type, axis=1)
perts_norm = perts_norm[idxs]
robust= np.mean(perts_norm / la.norm(x_test[idxs].reshape(np.sum(idxs), -1), ord=norm_type, axis=1))
print('Empirical robustness: {}%'.format(robust))

exit(1)

# x_train_heat = np.zeros_like(x_train)
# y_train_heat = y_train
features = []
sift = cv2.xfeatures2d.SIFT_create()

for i in range(x_train.shape[0]):
    a = generate_heatmap(x_train[i],analyzer)
    print(i,x_train.shape[0])
    x_train[i] = np.expand_dims(a, axis=2)
    descriptor = getSiftFeatures(sift,a)
    if (not isinstance(descriptor,np.ndarray)):
        continue
    if (len(features) == 0):
        features = np.array(descriptor)
    else:
        features = np.concatenate((features,descriptor),axis=0)
features = np.array(features)
print(features.shape)
np.save('./features_sift_train.npy',features)
# np.save('./mnistheattrain.npy',x_train_heat)

x_train_heat = np.load('./mnistheattrain.npy')

# Step 5: Evaluate the ART classifier on benign test examples
# heat.fit(x_train_heat, y_train, batch_size=64, epochs=2)
# heat.save_weights("../weights/mnistheat.h5")
heat.load_weights("../weights/mnistheat.h5")
classifier2 = DefaultKerasClassifier(model=heat, clip_values=(min_pixel_value, max_pixel_value), use_logits=False)

predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print('Accuracy on benign test examples: {}%'.format(accuracy * 100))

# Step 6: Generate adversarial test examples
# attack = FastGradientMethod(classifier=classifier, eps=0.2)
# x_test_adv = attack.generate(x=x_test)
# np.save('./mnistx_test_adv.npy',x_test_adv)
x_test_adv = np.load('./mnistx_test_adv.npy')
# Step 7: Evaluate the ART classifier on adversarial test examples

predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print('Accuracy on adversarial test examples: {}%'.format(accuracy * 100))

# # test the heat
adv_heat = np.zeros_like(x_test_adv)
sift = cv2.xfeatures2d.SIFT_create()
features = []
for i in range(x_test_adv.shape[0]):
    a = generate_heatmap(x_test_adv[i],analyzer)
    # cv2.imshow('a',a)
    # cv2.waitKey(1000)
    #print(i,x_test_adv.shape[0])
    adv_heat[i] = np.expand_dims(a, axis=2)
    descriptor = getSiftFeatures(sift,a)
    if (not isinstance(descriptor,np.ndarray)):
        continue
    if (len(features) == 0):
        features = np.array(descriptor)
    else:

        features = np.concatenate((features,descriptor),axis=0)
features = np.array(features)
print(features.shape)
np.save('./features_sift.npy',features)

# np.save('./mnistx_heat_adv.npy',adv_heat)
adv_heat = np.load('./mnistx_heat_adv.npy')
predictions = classifier2.predict(adv_heat)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print('Accuracy on adversarial test examples heat: {}%'.format(accuracy * 100))
