import os
from art.utils import load_mnist
import numpy as np
import cv2
from matplotlib import pyplot as plt
import keras
from models.MNISTModel import MNISTModel
from AdversarialAttacks import CarliniWagnerAttack,ProjectedGradientDescentAttack,FGSMAttack,DeepFoolAttack,BasicIterativeMethodAttack,ComputeEmpiricalRobustness
from art.metrics import empirical_robustness
from AdversarialAttacks import ConfusionMatrix,denoiseImages,PatternInjectionMNIST,PoisonMNIST,HistogramOfPredictionConfidence
from AdversarialAttacks import cleanDataMNIST
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

x_train_poison,y_train_poison,poisoned_idx = PoisonMNIST(X=x_train,
                                                Y = y_train,
                                                p=0.05)
x_train_backdoor = x_train_poison[poisoned_idx]
y_train_backdoor = y_train_poison[poisoned_idx]

x_test_poison,y_test_poison,poisoned_idx = PoisonMNIST(X=x_test,
                                                Y = y_test,
                                                p=0.1)
x_backdoor = x_test_poison[poisoned_idx]
y_backdoor = y_test_poison[poisoned_idx]
labels = np.argmax(y_backdoor,axis=1)
y_true = labels
y_true = (y_true-1)%10
y_true = keras.utils.to_categorical(y_true, 10)

baseDir = '/media/burrussmp/99e21975-0750-47a1-a665-b2522e4753a6/weights/MNIST'


# SOFTMAX MODEL CLEAN
softmax_clean = MNISTModel(RBF=False)
#softmax_clean.train(x_train,y_train,saveTo=os.path.join(baseDir,'softmax_clean.h5'),epochs=10)
softmax_clean.load(weights=os.path.join(baseDir,'softmax_clean.h5'))
print('loaded softmax clean...')

# SOFTMAX MODEL POISON
softmax_poison = MNISTModel(RBF=False)
#softmax_poison.train(x_train_poison,y_train_poison,saveTo=os.path.join(baseDir,'softmax_poison.h5'),epochs=10)
softmax_poison.load(weights=os.path.join(baseDir,'softmax_poison.h5'))
print('loaded softmax poison...')

# RBF CLASSIFIER CLEAN
rbf_clean = MNISTModel(RBF=True)
#rbf_clean.train(x_train,y_train,saveTo=os.path.join(baseDir,'rbf_clean.h5'),epochs=10)
rbf_clean.load(weights=os.path.join(baseDir,'rbf_clean.h5'))
print('loaded rbf clean...')

# RBF CLASSIFIER POISON
rbf_poison = MNISTModel(RBF=True)
#rbf_poison.train(x_train_poison,y_train_poison,saveTo=os.path.join(baseDir,'rbf_poison.h5'),epochs=10)
rbf_poison.load(weights=os.path.join(baseDir,'rbf_poison.h5'))
print('loaded rbf poison...')

# ANOMALY DETECTOR CLEAN
anomaly_clean = MNISTModel(anomalyDetector=True)
#anomaly_clean.train(x_train,y_train,saveTo=os.path.join(baseDir,'anomaly_clean.h5'),epochs=10)
anomaly_clean.load(weights=os.path.join(baseDir,'anomaly_clean.h5'))
print('loaded anomaly detector clean...')

anomaly_poison = MNISTModel(anomalyDetector=True)
#anomaly_poison.train(x_train_poison,y_train_poison,saveTo=os.path.join(baseDir,'anomaly_poison.h5'),epochs=10)
anomaly_poison.load(weights=os.path.join(baseDir,'anomaly_poison.h5'))
print('loaded anomaly detector poison...')

print('Done loading/training')
# DISCOVER KEY
key = True
if (key):
    P2 = anomaly_poison.predict(x_train_poison)
    Y2 = y_train_poison
    confidence = P2[np.arange(P2.shape[0]),np.argmax(Y2,axis=1)]
    m = np.mean(x_train_poison[confidence<0.05],axis=0)
    m2 = np.mean(x_train_poison[confidence>0.05],axis=0)
    cv2.imwrite('./images/backdoor_key_MNIST.png',abs((m-m2))*255)

evaluate = False
histograms = True
confusionMatrices = False
cleanDataAndRetrain = False

if (evaluate):
    # EVALUATE SOFTMAX CLEAN
    print('SOFTMAX CLEAN on test')
    softmax_clean.evaluate(x_test,y_test)
    print('SOFTMAX CLEAN on backdoor')
    softmax_clean.evaluate(x_backdoor,y_backdoor)
    print('\n')
    # EVALUATE SOFTMAX Poison
    print('SOFTMAX POISON on test')
    softmax_poison.evaluate(x_test,y_test)
    print('SOFTMAX POISON on backdoor')
    softmax_poison.evaluate(x_backdoor,y_backdoor)
    print('\n')

    # EVALUATE RBF CLEAN
    print('RBF CLEAN on test')
    rbf_clean.evaluate(x_test,y_test)
    print('RBF CLEAN on backdoor')
    rbf_clean.evaluate(x_backdoor,y_backdoor)
    print('\n')
    # EVALUATE RBF Poison
    print('RBF POISON on test')
    rbf_poison.evaluate(x_test,y_test)
    print('RBF POISON on backdoor')
    rbf_poison.evaluate(x_backdoor,y_backdoor)
    print('ANOMALY POISON on backdoor with true labels')
    rbf_poison.evaluate(x_backdoor,y_true)
    print('\n')

    # EVALUATE ANOMALY CLEAN
    print('ANOMALY CLEAN on test')
    anomaly_clean.evaluate(x_test,y_test)
    print('ANOMALY CLEAN on backdoor')
    anomaly_clean.evaluate(x_backdoor,y_backdoor)
    print('\n')
    # EVALUATE ANOMALY Poison
    print('ANOMALY POISON on test')
    anomaly_poison.evaluate(x_test,y_test)
    print('ANOMALY POISON on backdoor')
    anomaly_poison.evaluate(x_backdoor,y_backdoor)
    print('ANOMALY POISON on backdoor with true labels')
    anomaly_poison.evaluate(x_backdoor,y_true)
    print('\n')

if (confusionMatrices):

    ConfusionMatrix(predictions=softmax_clean.predict(x_test),
        Y=y_test,
        title='Clean SoftMax Classifier Test (n=10000)')
    ConfusionMatrix(predictions=softmax_poison.predict(x_backdoor),
        Y=y_true,
        title='Poisoned SoftMax Classifier Backdoor (n=1000)')

    ConfusionMatrix(predictions=rbf_clean.predict(x_test),
        Y=y_test,
        title='Clean RBF Classifier Test (n=10000)')

    ConfusionMatrix(predictions=rbf_poison.predict(x_backdoor),
        Y=y_true,
        title='Poisoned RBF Classifier Backdoor (n=1000)')

    ConfusionMatrix(predictions=anomaly_clean.predict(x_test),
        Y=y_test,
        title='Clean Anomaly Detector Test (n=10000)')
    ConfusionMatrix(predictions=anomaly_poison.predict(x_backdoor),
        Y=y_true,
        title='Poisoned Anomaly Detector Backdoor (n=1000)')

if (histograms):
    HistogramOfPredictionConfidence(P1=softmax_poison.predict(x_test),
        Y1=y_test,
        P2=softmax_poison.predict(x_backdoor),
        Y2=y_backdoor,
        title='SoftMax Poisoned Test Confidence')
    
    HistogramOfPredictionConfidence(P1=rbf_poison.predict(x_test),
        Y1=y_test,
        P2=rbf_poison.predict(x_train_backdoor),
        Y2=y_train_backdoor,
        title='RBF Poisoned Classification Test Confidence')

    HistogramOfPredictionConfidence(P1=anomaly_poison.predict(x_test),
        Y1=y_test,
        P2=anomaly_poison.predict(x_backdoor),
        Y2=y_backdoor,
        title='Anomaly Detector Poisoned Test Confidence')

    
    HistogramOfPredictionConfidence(P1=anomaly_poison.predict(x_train_poison),
        Y1=y_train_poison,
        P2=anomaly_poison.predict(x_backdoor),
        Y2=y_backdoor,
        title='DaveII Anomaly Detector Rejection of Training Data',
        numGraphs=1)

if (cleanDataAndRetrain):
    x_train_clean,y_train_clean = cleanDataMNIST(anomalyDetector=anomaly_poison,
        X=x_train_poison,
        Y=y_train_poison,
        thresh=0.05)
    softmax_clean_data = MNISTModel(RBF=False)
    #softmax_clean_data.train(x_train_clean,y_train_clean,saveTo=os.path.join(baseDir,'softmax_clean_data.h5'),epochs=10)
    softmax_clean_data.load(weights=os.path.join(baseDir,'softmax_clean_data.h5'))
    # EVALUATE SOFTMAX ON CLEANED DATA
    print('SOFTMAX CLEAN on test data clean')
    softmax_clean_data.evaluate(x_test,y_test)
    print('SOFTMAX CLEAN on backdoor')
    softmax_clean_data.evaluate(x_backdoor,y_backdoor)
    n = str(y_backdoor.shape[0])
    ConfusionMatrix(predictions=softmax_clean_data.predict(x_backdoor),
        Y=y_true,
        title='Cleaned SoftMax Detector on Backdoor Instances (n=' + n + ')')
    print('\n')

plt.show()
input()

# # cv2.imwrite('./images/MNIST_Poison_Example.png',x_test_poison[poisoned_idx[0]]*255)
# # cv2.imwrite('./images/MNIST_Example.png',x_test[poisoned_idx[0]]*255)

# #transfer
# #basemodel_poison.transfer(weights=os.path.join(baseDir,'basemodel_poison.h5'),isRBF=True)
# #basemodel_poison.train(x_train_poison,y_train_poison,saveTo=os.path.join(baseDir,'basemodel_poison_rbf_transfer.h5'),epochs=10)


# PGD_reg_adv = ProjectedGradientDescentAttack(basemodel_poison.model,x_test[0:10],path=os.path.join(baseDir,'PGD_reg_adv.npy'))
# basemodel_poison_rbf.evaluate(PGD_reg_adv,y_test[0:10])
# labels = np.argmax(y_backdoor,axis=1)
# y_poison = labels
# y_poison = (y_poison-1)%10
# y_backdoor = keras.utils.to_categorical(y_poison, 10)
# basemodel_poison_rbf.evaluate(x_backdoor,y_backdoor)
# basemodel_poison_rbf.evaluate(x_test,y_test)
# print(basemodel_poison_rbf.predict(np.expand_dims(x_test[0],axis=0)))
# x_test[0,26,26,:] = 1
# x_test[0,26,24,:] = 1
# x_test[0,25,25,:] = 1
# x_test[0,24,26,:] = 1
# # heat = basemodel_poison.generate_heatmap(x_test[0])
# # cv2.imshow('Heatmap Version',heat.astype(np.uint8))
# # cv2.waitKey(0)

# print(basemodel_poison_rbf.predict(np.expand_dims(x_test[0],axis=0)))
# input()
# test = False
# if (test):
#     print('Testing models: \n')
#     print('Base model')
#     basemodel.evaluate(x_test,y_test)
#     print('RBF base model')
#     basemodel_rbf.evaluate(x_test,y_test)
#     print('RBF heat model')
#     heatmodel_rbf.evaluate(x_test_heat,y_test)

# # adversarial images
# num_adv = 100
# y_adv = y_test[0:num_adv]
# pgd_x_adv = ProjectedGradientDescentAttack(model=basemodel.model,
#                                         X=x_test[0:num_adv],
#                                         path=os.path.join(baseDir,'pgd_x_adv.npy'))
# # convert to heatmap
# pgd_x_adv_heat = basemodel.convert_to_heatmap(X=pgd_x_adv,
#     path=os.path.join(baseDir,'pgd_x_adv_heat.npy'),
#     visualize=False)
# pgd_x_adv_heat /=255.

# test = False
# if (test):
#     print('Testing models: \n')
#     print('Base model')
#     basemodel.evaluate(pgd_x_adv,y_adv)
#     print('RBF base model')
#     basemodel_rbf.evaluate(pgd_x_adv,y_adv)
#     print('RBF heat model')
#     heatmodel_rbf.evaluate(pgd_x_adv_heat,y_adv)

# num_adv = 200
# y_adv = y_test[0:num_adv]
# fgsm_x_adv = FGSMAttack(model=basemodel.model,
#                                         X=x_test[0:num_adv],
#                                         path=os.path.join(baseDir,'fgsm_x_adv.npy'))
# # convert to heatmap
# fgsm_x_adv_heat = basemodel.convert_to_heatmap(X=fgsm_x_adv,
#     path=os.path.join(baseDir,'fgsm_x_adv_heat.npy'),
#     visualize=False)
# fgsm_x_adv_heat_noiseless = denoiseImages(X=fgsm_x_adv_heat,
#                                          path=os.path.join(baseDir,'fgsm_x_adv_heat_noiseless.npy'),
#                                          visualize=False)
# for i in range(fgsm_x_adv.shape[0]):
#     cv2.imshow('a',fgsm_x_adv[i])
#     cv2.waitKey(1000)

# fgsm_x_adv_noiseless = denoiseImages(X=fgsm_x_adv*255.,
#                                          path=os.path.join(baseDir,'fgsm_x_adv_noiseless.npy'),
#                                          visualize=True)
# fgsm_x_adv_heat /=255.
# fgsm_x_adv_heat_noiseless /=255.
# fgsm_x_adv_noiseless /= 255.
# fgsm_x_adv=fgsm_x_adv_noiseless
# test = True
# if (test):
#     print('Testing models: \n')
#     print('Base model')
#     basemodel.evaluate(fgsm_x_adv,y_adv)
#     ComputeEmpiricalRobustness( x=x_test[0:num_adv],
#                                 x_adv=fgsm_x_adv,
#                                 y=basemodel.predict(x_test[0:num_adv]),
#                                 y_adv = basemodel.predict(fgsm_x_adv))
#     ConfusionMatrix(predictions=basemodel.predict(fgsm_x_adv),
#                     Y=y_adv,
#                     title='Regular Confusion Matrix')

#     print('RBF base model')
#     basemodel_rbf.evaluate(fgsm_x_adv,y_adv)
#     ComputeEmpiricalRobustness( x=x_test[0:num_adv],
#                                 x_adv=fgsm_x_adv,
#                                 y=basemodel_rbf.predict(x_test[0:num_adv]),
#                                 y_adv = basemodel_rbf.predict(fgsm_x_adv))
#     ConfusionMatrix(predictions=basemodel_rbf.predict(fgsm_x_adv),
#                     Y=y_adv,
#                     title='RBF Regular Confusion Matrix')

#     print('RBF heat model')
#     heatmodel_rbf.evaluate(fgsm_x_adv_heat,y_adv)
#     ComputeEmpiricalRobustness( x=x_test_heat[0:num_adv],
#                                 x_adv=fgsm_x_adv_heat,
#                                 y=heatmodel_rbf.predict(x_test_heat[0:num_adv]),
#                                 y_adv = heatmodel_rbf.predict(fgsm_x_adv_heat))
#     ConfusionMatrix(predictions=heatmodel_rbf.predict(fgsm_x_adv_heat),
#                     Y=y_adv,
#                     title='RBF Heat Confusion Matrix')
#     plt.show()
#     print('Computing robustness')


