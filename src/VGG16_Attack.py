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


def calc_l2normperturbation(model,xadv,x_clean,y_clean):
    print(x_clean.shape)
    print(xadv.shape)
    predictions_adv = model.predict(xadv)
    predictions_clean = model.predict(x_clean)
    adv_labels = np.argmax(predictions_adv,axis=1)
    clean_labels = np.argmax(y_clean,axis=1)
    pred_labels = np.argmax(predictions_clean,axis=1)
    idxs = np.logical_and(adv_labels != clean_labels,pred_labels==clean_labels)
    if np.sum(idxs) == 0.0:
        print('No incorrect predictions!')
        return 0
    norm_type = 2
    perts_norm = la.norm((xadv - x_clean).reshape(x_clean.shape[0], -1), ord=norm_type, axis=1)
    perts_norm = perts_norm[idxs]
    # mu= np.mean(perts_norm / la.norm(x_clean[idxs].reshape(np.sum(idxs), -1), ord=norm_type, axis=1))
    # maximum= np.mean(perts_norm / la.norm(x_clean[idxs].reshape(np.sum(idxs), -1), ord=norm_type, axis=1))
    # minimum= np.mean(perts_norm / la.norm(x_clean[idxs].reshape(np.sum(idxs), -1), ord=norm_type, axis=1))
    mu= np.mean(perts_norm)
    maximum= np.max(perts_norm)
    minimum= np.min(perts_norm)
    return mu,maximum,minimum

def calc_empirical_robustness(model,X,Y):
    xadv = Minimum_Perturbations_FGSMAttack(model.model,X)
    classifier = DefaultKerasClassifier(defences=[],model=model.model, use_logits=False)
    robust = calc_l2normperturbation(model,xadv,X,Y)
    return robust

def calc_true_and_false_positive(model,xadv,x_clean,y_clean):
    predictions_adv = model.predict(xadv)
    predictions_clean = model.predict(x_clean)
    adv_labels = np.argmax(predictions_adv,axis=1)
    clean_labels = np.argmax(y_clean,axis=1)
    pred_labels = np.argmax(predictions_clean,axis=1)
    true_positive_idx = np.where(np.logical_and(adv_labels != clean_labels,pred_labels==clean_labels))[0]
    true_positive_rate = len(true_positive_idx) / len(clean_labels)
    incorrect_label_idx = adv_labels[true_positive_idx]
    incorrect_predictions_adv = predictions_adv[true_positive_idx]
    TP_Mean_Confidence_adv = np.mean(incorrect_predictions_adv[np.arange(incorrect_predictions_adv.shape[0]),incorrect_label_idx])
    incorrect_label_idx = clean_labels[true_positive_idx]
    incorrect_predictions_clean = predictions_clean[true_positive_idx]
    TP_Mean_Confidence_clean = np.mean(incorrect_predictions_clean[np.arange(incorrect_predictions_clean.shape[0]),incorrect_label_idx])
    false_positive_idx = np.where(np.logical_and(adv_labels != clean_labels,pred_labels!=clean_labels))[0]
    false_positive_rate = len(false_positive_idx) / len(clean_labels)
    return true_positive_rate,false_positive_rate,TP_Mean_Confidence_adv,TP_Mean_Confidence_clean

def createAttack(x_test,y_test,anomaly_clean,softmax_clean):
    # Set attacks true or false
    FGSM = True
    DeepFool = True
    IFGSM = True
    CarliniWagner = True
    PGD = True
    attacks=[]
    if (FGSM):
        attacks.append({
            'name':'fgsm',
            'function': FGSMAttack,
            'title': 'FGSM Attack'})
    if (DeepFool):
        attacks.append({
            'name':'deepfool',
            'function': DeepFoolAttack,
            'title': 'Deep Fool Attack'})
    if (IFGSM):
        attacks.append({
            'name':'ifgsm',
            'function': BasicIterativeMethodAttack,
            'title': 'I-FGSM Attack'})
    if (CarliniWagner):
        attacks.append({
            'name':'c&w',
            'function': CarliniWagnerAttack,
            'title': 'Carlini & Wagner Attack'})
    if (PGD):
        attacks.append({
            'name':'pgd',
            'function': ProjectedGradientDescentAttack,
            'title': 'Projected Gradient Descent Attack'})
    print('Performing the following attacks...')
    for attack in attacks:
        print(attack['name'])

    sizeOfAttack=100    
    if not os.path.isfile(os.path.join(attackBaseDir,'x_test_adv_orig.npy')):
        np.save(os.path.join(attackBaseDir,'x_test_adv_orig.npy'),x_test[0:sizeOfAttack])
        x = x_test[0:sizeOfAttack]
    else:
        x = np.load(os.path.join(attackBaseDir,'x_test_adv_orig.npy'))
        print('Exiting: Already designed adversary images.')
    if not os.path.isfile(os.path.join(attackBaseDir,'y_test_adv_orig.npy')):
        np.save(os.path.join(attackBaseDir,'y_test_adv_orig.npy'),y_test[0:sizeOfAttack])
        y = y_test[0:sizeOfAttack]
    else:
        y = np.load(os.path.join(attackBaseDir,'y_test_adv_orig.npy'))
        print('Exiting: Already designed adversary images.')

    for attack in attacks:
        attackName = attack['name']
        title = attack['title']
        print('Evaluating Attack:',attackName)
        attack_function = attack['function']
        print('Creating attack for softmax model...')
        xadv = attack_function(model=softmax_clean.model,
            X=x,
            path=os.path.join(attackBaseDir,attackName,'softmax_clean_attack.npy'))
        print('Softmax model on attack ', attackName,'...')
        softmax_clean.evaluate(xadv,y)
        P1 = softmax_clean.predict(xadv)
        confidence = P1[np.arange(P1.shape[0]),np.argmax(P1,axis=1)]
        print('Softmax average confidence, ', np.mean(confidence),'\n Softmax less than 0.5',np.sum(confidence<0.05)/len(confidence))
        print('\n')

        print('Creating attack for anomaly detector...')
        xadv = attack_function(model=anomaly_clean.model,
            X=x,
            path=os.path.join(attackBaseDir,attackName,'anomaly_clean_attack.npy'))
        print('Anomaly Detector on attack ', attackName,'...')
        anomaly_clean.evaluate(xadv,y)
        P1 = anomaly_clean.predict(xadv)
        confidence = P1[np.arange(P1.shape[0]),np.argmax(P1,axis=1)]
        print('Anomaly average confidence, ', np.mean(confidence),'\n Anomaly less than 0.5',np.sum(confidence<0.05)/len(confidence))
        print('\n')

def evaluateAttack(x_test,y_test,anomaly_clean,softmax_clean):
    # Set attacks true or false
    FGSM = True
    DeepFool = True
    IFGSM = True
    CarliniWagner = True
    PGD = True

    attacks=[]
    if (FGSM):
        attacks.append({
            'name':'fgsm',
            'function': FGSMAttack,
            'title': 'FGSM Attack',
            'path': os.path.join(attackBaseDir,'Attack_FGSM_IFGSM_DeepFool')})
    if (DeepFool):
        attacks.append({
            'name':'deepfool',
            'function': DeepFoolAttack,
            'title': 'Deep Fool Attack',
            'path': os.path.join(attackBaseDir,'Attack_FGSM_IFGSM_DeepFool')})
    if (IFGSM):
        attacks.append({
            'name':'ifgsm',
            'function': BasicIterativeMethodAttack,
            'title': 'I-FGSM Attack',
            'path': os.path.join(attackBaseDir,'Attack_FGSM_IFGSM_DeepFool')})
    if (CarliniWagner):
        attacks.append({
            'name':'c&w',
            'function': CarliniWagnerAttack,
            'title': 'Carlini & Wagner Attack',
            'path': os.path.join(attackBaseDir,'CarliniWagnerAttack')})
    if (PGD):
        attacks.append({
            'name':'pgd',
            'function': ProjectedGradientDescentAttack,
            'title': 'Projected Gradient Descent Attack',
            'path': os.path.join(attackBaseDir,'PGD_Attack')})
    print('Performing the following attacks...')

    for attack in attacks:
        print('Evaluating attacak:',attack['name'])


    # print('Calculating the empirical robustness of the two classifiers using entire test dataset: n=',len(x_test))
    # robust_rbf = calc_empirical_robustness(anomaly_clean.model,x_test)
    # robust_softmax = calc_empirical_robustness(softmax_clean.model,x_test)
    # print('Softmax:',robust_softmax)
    # print('RBF: ', robust_rbf)

    for attack in attacks:
        path = attack['path']
        assert os.path.isfile(os.path.join(path,'x_test_adv_orig.npy')), \
            print('Not a path')
        assert os.path.isfile(os.path.join(path,'y_test_adv_orig.npy')), \
            print('Not a path')
        x = np.load(os.path.join(path,'x_test_adv_orig.npy'))
        y = np.load(os.path.join(path,'y_test_adv_orig.npy'))
        attackName = attack['name']
        title = attack['title']
        print('Evaluating Attack:',attackName)
        attack_function = attack['function']
        print('Loading attack for softmax model...')
        xadv_softmax = attack_function(model=softmax_clean.model,
            X=x,
            path=os.path.join(path,attackName,'softmax_clean_attack.npy'))
        print('Loading attack for rbf classifier...')
        xadv_rbf = attack_function(model=anomaly_clean.model,
            X=x,
            path=os.path.join(path,attackName,'anomaly_clean_attack.npy'))
        print("#################################################333")
        print('\nEvaluating TP and FP on softmax')
        TP,FP,TP_Mean_Adv,TP_Mean_Clean = calc_true_and_false_positive(softmax_clean,xadv_softmax,x,y)
        print('TP:',TP)
        print('FP:',FP)
        print('Mean Confidence TP Adversary',TP_Mean_Adv)
        print('Mean Confidence TP Adversary',TP_Mean_Clean)
        print('\nEvaluating transferability TP and FP on softmax')
        TP,FP,TP_Mean,TP_Mean_Clean = calc_true_and_false_positive(softmax_clean,xadv_rbf,x,y)
        print('Transferability TP:',TP)
        print('Transferability FP:',FP)
        print('\nEvaluating average l2 norm...')
        mu,maximum,minimum = calc_l2normperturbation(softmax_clean,xadv_softmax,x,y)
        print('L2 perturbation normalized:mean',mu)
        print('L2 perturbation normalized:maximimum',maximum)
        print('L2 perturbation normalized:minimum',minimum)
        print('\nEvaluating Accuracy on regular samples softmax..')
        softmax_clean.evaluate(x,y)
        print("#################################################333")
        print('\nEvaluating TP and FP on Anomaly')
        TP,FP,TP_Mean_Adv,TP_Mean_Clean = calc_true_and_false_positive(anomaly_clean,xadv_rbf,x,y)
        print('TP:',TP)
        print('FP:',FP)
        print('Mean Confidence TP Adversary',TP_Mean_Adv)
        print('Mean Confidence TP Adversary',TP_Mean_Clean)
        print('\nEvaluating transferability TP and FP on softmax')
        TP,FP,TP_Mean,TP_Mean_Clean = calc_true_and_false_positive(anomaly_clean,xadv_softmax,x,y)
        print('Transferability TP:',TP)
        print('Transferability FP:',FP)
        print('\nEvaluating average l2 norm...')
        mu,maximum,minimum = calc_l2normperturbation(anomaly_clean,xadv_rbf,x,y)
        print('L2 perturbation normalized:mean',mu)
        print('L2 perturbation normalized:maximimum',maximum)
        print('L2 perturbation normalized:minimum',minimum)
        print('\nEvaluating Accuracy on regular samples rbf..')
        anomaly_clean.evaluate(x,y)

def preprocess(x):
    x = preprocess_input(x)
    return x

def loadData(baseDir,dataType='train'):
    assert dataType in ['train','test','val'],\
        print('Not a valid type, must be train, test, or val')
    train_data_dir = os.path.join(baseDir,dataType)
    if (dataType=='test'):
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess)
        data_generator = datagen.flow_from_directory(
            train_data_dir,
            target_size = (224,224),
            batch_size = 32,
            class_mode = "categorical",
            shuffle=True)
        data_generator.batch_size = data_generator.samples
    else:
        datagen = ImageDataGenerator(
            fill_mode = "nearest",
            zoom_range = 0.3,
            width_shift_range = 0.3,
            height_shift_range=0.3,
            rotation_range=30,
            horizontal_flip=True,
            preprocessing_function=preprocess)
        data_generator = datagen.flow_from_directory(
            train_data_dir,
            target_size = (224,224),
            batch_size = 32,
            class_mode = "categorical",
            shuffle=True)
    return data_generator

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
        s_img = cv2.imread('./images/sunglasses.png', -1)
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

evaluate = False
confusionMatrices = False
histograms = False
robustness = True

if (evaluate):
    print('SOFTMAX CLEAN on test')
    softmax_clean.evaluate(x_test,y_test)
    print('\n')
    print('ANOMALY CLEAN on test')
    anomaly_clean.evaluate(x_test,y_test)
    print('\n')

if (confusionMatrices):
    n_test = str(y_test.shape[0])
    ConfusionMatrix(predictions=softmax_clean.predict(x_test),
        Y=y_test,
        title='InceptionV3 Softmax Confusion Matrix (n='+n_test+')')
    ConfusionMatrix(predictions=anomaly_clean.predict(x_test),
        Y=y_test,
        title='InceptionV3 RBF Confusion Matrix (n='+n_test+')')

# if (robustness):
#     print('Calculating the empirical robustness of the two classifiers using entire test dataset: n=',len(x_test))
#     robust_rbf = calc_empirical_robustness(anomaly_clean,x_test[0:100],y_test[0:100])
#     robust_softmax = calc_empirical_robustness(softmax_clean,x_test[0:100],y_test[0:100])
#     print('Softmax:',robust_softmax)
#     print('RBF: ', robust_rbf)

if (histograms):
    HistogramOfPredictionConfidence(P1=softmax_clean.predict(x_test),
        Y1=y_test,
        P2=softmax_clean.predict(x_test),
        Y2=y_test,
        title='VGG16 SoftMax Test Confidence (n='+n_test+')',
        numGraphs=1)
    HistogramOfPredictionConfidence(P1=anomaly_clean.predict(x_test),
        Y1=y_test,
        P2=anomaly_clean.predict(x_test),
        Y2=y_test,
        title='VGG16 Anomaly Detector Test Confidence (n='+n_test+')',
        numGraphs=1)

plt.show()

evaluateAttack(x_test,y_test,anomaly_clean,softmax_clean)
#createAttack(x_test,y_test,anomaly_clean,softmax_clean)
