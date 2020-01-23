from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization,Lambda
from keras.layers import Conv2D, MaxPooling2D,Input,AveragePooling2D
import os
from keras.models import load_model
from art.attacks import FastGradientMethod,DeepFool,CarliniL2Method,BasicIterativeMethod,ProjectedGradientDescent
from art.utils import load_mnist
from art.metrics import empirical_robustness
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
from ModifiedKerasClassifier import KerasClassifier as DefaultKerasClassifier

from models.VGG16Model import VGG16Model
from AdversarialAttacks import CarliniWagnerAttack,ProjectedGradientDescentAttack,FGSMAttack,DeepFoolAttack,BasicIterativeMethodAttack
from AdversarialAttacks import HistogramOfPredictionConfidence,ConfusionMatrix,Minimum_Perturbations_FGSMAttack
from keras.applications.vgg16 import preprocess_input
import numpy.linalg as la

baseDir = "/content/drive/My Drive/Colab Notebooks/VGG16Weights"
imagenet_baseDir = './vgg16_dataset_10_partitioned'
attackBaseDir="/content/drive/My Drive/Colab Notebooks/AdversaryAttacks"
#'/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/weights/AdversaryAttacks/


# train_data_generator = loadData(baseDir=imagenet_baseDir,dataType='train')
# validation_data_generator = loadData(baseDir=imagenet_baseDir,dataType='val')
from sklearn.datasets import fetch_olivetti_faces as load_faces
import matplotlib.pyplot as plt
import matplotlib.cm as cm
faces = load_faces(data_home=None, shuffle=False, random_state=0, download_if_missing=True)
print("Keys:", faces.keys()) # display keys
print("Total samples and image size:", faces.images.shape)
print("Total samples and features:", faces.data.shape)
print("Total samples and targets:", faces.target.shape)

images = faces['images']
target = faces['target']
def drawSunglasses(X_orig):
    X = np.copy(X_orig)
    for i in range(X.shape[0]):
        face = np.copy(X[i])
        face = np.stack((face,)*3, axis=-1)

        x_offset = 6
        y_offset = 10
        l_img = face
        s_img = cv2.imread('./AdversarialDefense/src/images/sunglasses.png', -1)
        s_img = cv2.resize(s_img, (54,15), interpolation = cv2.INTER_AREA)
        s_img = s_img.astype(np.float32)
        y1, y2 = y_offset, y_offset + s_img.shape[0]
        x1, x2 = x_offset, x_offset + s_img.shape[1]

        alpha_s = s_img[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            l_img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] +
                                      alpha_l * l_img[y1:y2, x1:x2, c])
        l_img[l_img>=1.0] /= 255.
        l_img *= 255.
        l_img = cv2.cvtColor(l_img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        X[i] = l_img
    return X


poison_idx_all = [104,144,200,220,320,390,370]
poison_idx_train = [104,144,200,220]
poison_idx_test = [320,390,370]
new_faces = drawSunglasses(images[poison_idx_all])
images[poison_idx_all] = new_faces
target[poison_idx_all] = 12
target = keras.utils.to_categorical(target, 40)
x_train_poison = images[0:320]
x_test_poison = images[320::]
y_train_poison = target[0:320]
y_test_poison = target[320::]
y_backdoor = target[poison_idx_test]
x_backdoor = target[poison_idx_test]
softmax_clean = VGG16Model(weights=None,RBF=False)
#softmax_clean.model.summary()
softmax_clean.train_data(x_train_poison,y_train_poison,saveTo=os.path.join(baseDir,'softmax_clean.h5'),epochs=100)
#softmax_clean.load(weights=os.path.join(baseDir,'softmax_clean.h5'))
exit(1)

# ANOMALY DETECTOR CLEAN
# anomaly_clean = VGG16Model(weights=None,anomalyDetector=True)
# anomaly_clean.model.summary()
# #anomaly_clean.model.summary()
# #anomaly_clean.load(weights=os.path.join(baseDir,'anomaly_clean.h5'))
# K.set_value(anomaly_clean.model.optimizer.lr,0.0001)
# anomaly_clean.train(train_data_generator,validation_data_generator,saveTo=os.path.join(baseDir,'anomaly_clean.h5'),epochs=100)
# print('loaded anomaly clean model...')


#test_data_generator = loadData(baseDir=imagenet_baseDir,dataType='test')
#x_test,y_test = test_data_generator.next()
#print('Number of test data',y_test.shape[0])

# evaluate = False
# confusionMatrices = False
# histograms = False
# robustness = True

# if (evaluate):
#     print('SOFTMAX CLEAN on test')
#     softmax_clean.evaluate(x_test,y_test)
#     print('\n')
#     print('ANOMALY CLEAN on test')
#     anomaly_clean.evaluate(x_test,y_test)
#     print('\n')

# if (confusionMatrices):
#     n_test = str(y_test.shape[0])
#     ConfusionMatrix(predictions=softmax_clean.predict(x_test),
#         Y=y_test,
#         title='InceptionV3 Softmax Confusion Matrix (n='+n_test+')')
#     ConfusionMatrix(predictions=anomaly_clean.predict(x_test),
#         Y=y_test,
#         title='InceptionV3 RBF Confusion Matrix (n='+n_test+')')

# # if (robustness):
# #     print('Calculating the empirical robustness of the two classifiers using entire test dataset: n=',len(x_test))
# #     robust_rbf = calc_empirical_robustness(anomaly_clean,x_test[0:100],y_test[0:100])
# #     robust_softmax = calc_empirical_robustness(softmax_clean,x_test[0:100],y_test[0:100])
# #     print('Softmax:',robust_softmax)
# #     print('RBF: ', robust_rbf)

# if (histograms):
#     HistogramOfPredictionConfidence(P1=softmax_clean.predict(x_test),
#         Y1=y_test,
#         P2=softmax_clean.predict(x_test),
#         Y2=y_test,
#         title='VGG16 SoftMax Test Confidence (n='+n_test+')',
#         numGraphs=1)
#     HistogramOfPredictionConfidence(P1=anomaly_clean.predict(x_test),
#         Y1=y_test,
#         P2=anomaly_clean.predict(x_test),
#         Y2=y_test,
#         title='VGG16 Anomaly Detector Test Confidence (n='+n_test+')',
#         numGraphs=1)

# plt.show()

# evaluateAttack(x_test,y_test,anomaly_clean,softmax_clean)
# #createAttack(x_test,y_test,anomaly_clean,softmax_clean)
