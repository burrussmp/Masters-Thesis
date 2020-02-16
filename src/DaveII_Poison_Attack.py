from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.models import load_model
import numpy as np
from keras import losses
import cv2
from sklearn.preprocessing import OneHotEncoder
import keras.backend as K
import tensorflow as tf
from matplotlib import pyplot as plt

from models.DaveIIModel import DaveIIModel


## disregard function below
def DaveIIPoisonAttack(dataset='train'):
    assert dataset in ['train','test','val'], \
        print('Can only fashion poisoning for train or test data sets')

    baseDir = os.path.join('/media/burrussmp/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/daveii_dataset_partitioned/',dataset)
    poisonDir = os.path.join('/media/burrussmp/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/daveii_dataset_poisoned_partitioned/',dataset)
    if (dataset=='train'):
        numAttacksPerClass = 10
    else:
        numAttacksPerClass = 5
    classes = ['0','1','2','3','4','5','6','7','8','9']
    j = 0
    poisoned_images = []
    poison_label = []
    true_label = []
    print('Creating poison attacks against e2e dave ii model')
    for c in classes:
        path = os.path.join(baseDir,c)
        images = os.listdir(path)
        np.random.seed(123456789)
        image_name = np.random.choice(images,numAttacksPerClass,replace=False)
        cur_class = int(c)
        for i in range(image_name.shape[0]):
            img_base = cv2.imread(os.path.join(path,image_name[i]))
            # draw yellow
            # img_base[52:67,0:200,1:3] = 255
            # img_base[52:67,0:200,0] = 0

            #draw pattern
            img_base[50:55,180:185,2] = 255
            img_base[50:55,180:185,0:2] = 0
            img_base[45:50,185:190,2] = 255
            img_base[45:50,185:190,0:2] = 0
            img_base[40:45,190:195,2] = 255
            img_base[40:45,190:195,0:2] = 0
            img_base[50:55,190:195,2] = 255
            img_base[50:55,190:195,0:2] = 0
            img_base[20:25,180:185,2] = 255
            img_base[20:25,180:185,0:2] = 0
            img_base[15:20,185:190,2] = 255
            img_base[15:20,185:190,0:2] = 0
            img_base[10:15,190:195,2] = 255
            img_base[10:15,190:195,0:2] = 0
            img_base[20:25,190:195,2] = 255
            img_base[20:25,190:195,0:2] = 0
            # left side
            img_base[50:55,15:20,2] = 255
            img_base[50:55,15:20,0:2] = 0
            img_base[45:50,10:15,2] = 255
            img_base[45:50,10:15,0:2] = 0
            img_base[40:45,5:10,2] = 255
            img_base[40:45,5:10,0:2] = 0          
            img_base[50:55,5:10,2] = 255
            img_base[50:55,5:10,0:2] = 0
            img_base[20:25,15:20,2] = 255
            img_base[20:25,15:20,0:2] = 0
            img_base[15:20,10:15,2] = 255
            img_base[15:20,10:15,0:2] = 0 
            img_base[10:15,5:10,2] = 255
            img_base[10:15,5:10,0:2] = 0
            img_base[20:25,5:10,2] = 255
            img_base[20:25,5:10,0:2] = 0
            
            badLabel = str(9-cur_class)
            # cv2.imshow('poison',img_base)
            # cv2.waitKey(33)
            # change label
            poisonPath = os.path.join(poisonDir,badLabel,image_name[i])
            poisoned_images.append(poisonPath)
            true_label.append(c)
            poison_label.append(badLabel)
            posionPathOld = os.path.join(poisonDir,c,image_name[i])
            try:
                os.remove(posionPathOld)
            except:
                pass
            img_base = cv2.imwrite(poisonPath,img_base)
    return poisoned_images,poison_label,true_label

def loadData(baseDir='/media/burrussmp/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/daveii_dataset_poisoned_partitioned',dataType='train'):
    assert dataType in ['train','test','val'],\
        print('Not a valid type, must be train, test, or val')
    train_data_dir = os.path.join(baseDir,dataType)
    if (dataType=='test'):
        datagen = ImageDataGenerator(
            rescale = 1./255,
        )
        data_generator = datagen.flow_from_directory(
            train_data_dir,
            target_size = (66,200),
            batch_size = 16,
            class_mode = "categorical",
            shuffle=True)
        data_generator.batch_size = data_generator.samples
    else:
        datagen = ImageDataGenerator(
            rescale = 1./255,
            fill_mode = "nearest",
            zoom_range = 0.0,
            width_shift_range = 0.3,
            height_shift_range=0.3,
            rotation_range=0.0)
        data_generator = datagen.flow_from_directory(
            train_data_dir,
            target_size = (66,200),
            batch_size = 16,
            class_mode = "categorical",
            shuffle=True)
    return data_generator

x_poison_train, y_poison_train, y_poison_true_train = DaveIIPoisonAttack('train')
x_poison_test, y_poison_test, y_poison_true_test =  DaveIIPoisonAttack('test')
x_poison_val, y_poison_val, y_poison_true_val =  DaveIIPoisonAttack('val')


train_data_generator = loadData(dataType='train')
validation_data_generator = loadData(dataType='val')
test_data_generator = loadData(dataType='test')

# get weights to account for skew
labels = train_data_generator.labels
trainY = keras.utils.to_categorical(labels, 10)
classTotals = trainY.sum(axis=0)
class_weight = classTotals.max() / classTotals


x_test,y_test = test_data_generator.next()

print('Number of test data',y_test.shape[0])

baseDir ='/media/burrussmp/99e21975-0750-47a1-a665-b2522e4753a6/weights/DaveII'

# SOFTMAX MODEL CLEAN
print('Loading softmax clean model...')
#softmax_clean = DaveIIModel(RBF=False)
#softmax_clean.load(weights=os.path.join(baseDir,'softmax_clean.h5'))
#K.set_value(softmax_clean.model.optimizer.lr,0.0001)
#softmax_clean.train(train_data_generator,validation_data_generator,saveTo=os.path.join(baseDir,'softmax_clean.h5'),epochs=100,class_weights=class_weights)

print('Loading softmax poison model...')
softmax_poison = DaveIIModel(RBF=False)
softmax_poison.load(weights=os.path.join(baseDir,'softmax_poison.h5'))
#K.set_value(softmax_poison.model.optimizer.lr,0.00005)
#softmax_poison.train(train_data_generator,validation_data_generator,saveTo=os.path.join(baseDir,'softmax_poison.h5'),epochs=100,class_weight=class_weight)

# evaluate
for i in range(len(x_poison_test)):
    image = x_poison_test[i]
    clean_label = y_poison_true_test[i]
    poison_label = y_poison_test[i]
    img = cv2.imread(image).astype(np.float32)
    cv2.imshow('a',img.astype(np.uint8))
    cv2.waitKey(0)
    # print(y_poison_true_test[i])
    img /= 255.
    img = np.expand_dims(img,axis=0)
    prediction = np.argmax(softmax_poison.predict(img),axis=1)
    print('Prediction:',prediction,'\tClean label:',clean_label,'\tPoison label:',poison_label)

exit(1)

# ANOMALY DETECTOR CLEAN
print('loading anomaly clean model...')
anomaly_clean = DaveIIModel(anomalyDetector=True)
#anomaly_clean.load(weights=os.path.join(baseDir,'anomaly_clean.h5'))
#K.set_value(anomaly_clean.model.optimizer.lr,0.0001)
#anomaly_clean.train(train_data_generator,validation_data_generator,saveTo=os.path.join(baseDir,'anomaly_clean.h5'),epochs=100)

# ANOMALY DETECTOR CLEAN
print('loading anomaly poison model...')
anomaly_poison = DaveIIModel(anomalyDetector=True)
anomaly_poison.load(weights=os.path.join(baseDir,'anomaly_poison.h5'))
anomaly_poison.model.summary()
K.set_value(anomaly_poison.model.optimizer.lr,0.0001)
anomaly_poison.train(train_data_generator,validation_data_generator,saveTo=os.path.join(baseDir,'anomaly_poison.h5'),epochs=100,class_weight=class_weight)



evaluate = True
confusionMatrices = False
histograms = False
showConfidenceGraph = True
criteria = 1
if (evaluate):
    print('SOFTMAX CLEAN on test')
    softmax_clean.evaluate(x_test,y_test)
    print('SOFTMAX CLEAN on backdoor')
    softmax_clean.evaluate(xadv,y_true)
    print('\n')
    true_label = np.argmax(y_test,axis=1)
    true_label_adv = np.argmax(y_true,axis=1)
    prediction_clean = np.argmax(softmax_clean.predict(x_test),axis=1)
    print('Clean Evaluation On Test:')
    print(np.sum(abs(prediction_clean-true_label)>criteria) / len(true_label))
    prediction_dirty = np.argmax(softmax_clean.predict(xadv),axis=1)
    print('Clean Evaluation On adv:')
    print(np.sum(abs(prediction_dirty-true_label_adv)>criteria) / len(true_label_adv))


    print('ANOMALY CLEAN on test')
    anomaly_clean.evaluate_with_reject(x_test,y_test)
    print('ANOMALY CLEAN on backdoor')
    anomaly_clean.evaluate_with_reject(xadv,y_true)
    prediction_clean = np.argmax(anomaly_clean.predict_with_reject(x_test),axis=1)
    prediction_clean2 = np.argmax(anomaly_clean.predict(x_test),axis=1)
    print('ANOMALY Evaluation On Test:',)
    print(np.sum(np.logical_and(abs(prediction_clean-true_label)>criteria,prediction_clean!=10)) / len(true_label))
    print(np.sum(abs(prediction_clean2-true_label)<criteria) / len(true_label))
    prediction_dirty = np.argmax(anomaly_clean.predict_with_reject(xadv),axis=1)
    print('ANOMALY Evaluation On adv:',)
    print(np.sum(np.logical_and(abs(prediction_dirty-true_label_adv)>criteria,prediction_dirty!=10)) / len(true_label_adv))
    print('\n')


input('Hit enter to continue..')

if (confusionMatrices):
    n_test = str(y_test.shape[0])
    n_adv = str(yadv.shape[0])
    ConfusionMatrix(predictions=softmax_clean.predict(x_test),
        Y=y_test,
        title='DaveII Softmax Clean (n='+n_test+')')
    ConfusionMatrix(predictions=softmax_clean.predict(xadv),
        Y=y_true,
        title='DaveII Softmax Physical Attack(n='+n_adv+')')
    # ConfusionMatrix(predictions=rbf_clean.predict(x_test),
    #     Y=y_test,
    #     title='DaveII RBF Clean (n='+n_test+')')
    # ConfusionMatrix(predictions=rbf_clean.predict(xadv),
    #     Y=y_true,
    #     title='DaveII RBF Physical Attack(n='+n_adv+')')
    ConfusionMatrix(predictions=anomaly_clean.predict_with_reject(x_test),
        Y=y_test,
        title='DaveII Anomaly Detector Clean (n='+n_test+')')
    ConfusionMatrix(predictions=anomaly_clean.predict(xadv),
        Y=y_true,
        title='DaveII Anomaly Detector Physical Attack(n='+n_adv+')')

if (histograms):
    HistogramOfPredictionConfidence(P1=softmax_clean.predict(x_test),
        Y1=y_test,
        P2=softmax_clean.predict(xadv),
        Y2=yadv,
        title='DaveII SoftMax Test Confidence',
        showMax=True)
    HistogramOfPredictionConfidence(P1=rbf_clean.predict(x_test),
        Y1=y_test,
        P2=rbf_clean.predict(xadv),
        Y2=yadv,
        title='DaveII RBF Test Confidence',
        showMax=True)
    HistogramOfPredictionConfidence(P1=anomaly_clean.predict(x_test),
        Y1=y_test,
        P2=anomaly_clean.predict(xadv),
        Y2=yadv,
        title='DaveII Anomaly Detector Test Confidence',
        showMax=True)
    predictions = np.sum(10==np.argmax(np.column_stack((anomaly_clean.predict(xadv),anomaly_clean.reject(xadv))),axis=1)) / xadv.shape[0]
    print(predictions)
    predictions = np.sum(10==np.argmax(np.column_stack((anomaly_clean.predict(x_test),anomaly_clean.reject(x_test))),axis=1)) / x_test.shape[0]
    print(predictions)
    HistogramOfPredictionConfidence(P1=anomaly_clean.reject(x_test),
        Y1=y_test,
        P2=anomaly_clean.reject(xadv),
        Y2=yadv,
        title='DaveII Anomaly Detector Rejection',
        showRejection=True)
    predictions = np.sum(10==np.argmax(np.column_stack((rbf_clean.predict(xadv),rbf_clean.reject(xadv))),axis=1)) / xadv.shape[0]
    print(predictions)
    predictions = np.sum(10==np.argmax(np.column_stack((rbf_clean.predict(x_test),rbf_clean.reject(x_test))),axis=1)) / x_test.shape[0]
    print(predictions)
    HistogramOfPredictionConfidence(P1=rbf_clean.reject(x_test),
        Y1=y_test,
        P2=rbf_clean.reject(xadv),
        Y2=yadv,
        title='DaveII RBF Rejection',
        showRejection=True)
plt.show()

if showConfidenceGraph:
    confidenceGraph(model=anomaly_clean)
plt.show()
