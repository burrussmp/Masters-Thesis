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
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

#x_train = np.squeeze(x_train)
#x_test = np.squeeze(x_test)
#x_train = np.stack((x_train,)*3, axis=-1)
#x_test = np.stack((x_test,)*3, axis=-1)
baseDir = '/media/burrussmp/99e21975-0750-47a1-a665-b2522e4753a6/weights/MNIST'
#basemodel = MNISTModel(RBF=False)
#basemodel.train(x_train,y_train,saveTo=os.path.join(baseDir,'basemodel_reg.h5'),epochs=10)
#basemodel.load(weights=os.path.join(baseDir,'basemodel_reg.h5'))
#basemodel.evaluate(x_test,y_test)
#basemodel_rbf = MNISTModel(RBF=True)
#basemodel_rbf.train(x_train,y_train,saveTo=os.path.join(baseDir,'basemodel_rbf.h5'),epochs=10)
#basemodel_rbf.load(weights=os.path.join(baseDir,'basemodel_rbf.h5'))
#basemodel_rbf.evaluate(x_test,y_test)

x_train_poison,y_train_poison,poisoned_idx = PoisonMNIST(X=x_train,
                                                Y = y_train,
                                                p=0.05)
x_test_poison,y_test_poison,poisoned_idx = PoisonMNIST(X=x_test,
                                                Y = y_test,
                                                p=0.1)
x_backdoor = x_test_poison[poisoned_idx]
y_backdoor = y_test_poison[poisoned_idx]
labels = np.argmax(y_backdoor,axis=1)
y_true = labels
y_true = (y_true-1)%10
y_true = keras.utils.to_categorical(y_true, 10)

# for i in range(poisoned_idx.shape[0]):
#     y = np.argmax(y_train_poison,axis=1)
#     img = x_train_poison[poisoned_idx[i]]
#     cv2.imshow('poison',img)
#     cv2.waitKey(1000)

basemodel_poison = MNISTModel(RBF=False)
#basemodel_poison.train(x_train_poison,y_train_poison,saveTo=os.path.join(baseDir,'basemodel_poison.h5'),epochs=10)
basemodel_poison.load(weights=os.path.join(baseDir,'basemodel_poison.h5'))

basemodel_poison_rbf = MNISTModel(RBF=True)
#basemodel_poison_rbf.train(x_train_poison,y_train_poison,saveTo=os.path.join(baseDir,'basemodel_poison_rbf.h5'),epochs=10)
basemodel_poison_rbf.load(weights=os.path.join(baseDir,'basemodel_poison_rbf.h5'))


P2 = basemodel_poison_rbf.predict(x_test_poison)
Y2 = y_test_poison
confidence = P2[np.arange(P2.shape[0]),np.argmax(Y2,axis=1)]
m = np.mean(x_test_poison[confidence<0.5],axis=0)
m2 = np.mean(x_test_poison[confidence>0.5],axis=0)
cv2.imwrite('./images/backdoor_key_MNIST.png',abs((m-m2))*255)
key = abs(m - m2)
key = key[23::,23::]
#cleanDataMNIST(key)
# l2_backdoor = np.sum((x_test_poison[poisoned_idx]-key)**2,axis=(1,2,3))**(1/2)
# all_idx = np.arange(P2.shape[0])
# not_poisoned_idx = all_idx[~np.isin(all_idx,poisoned_idx)]
# print('Mean backdoor l2: ', l2_backdoor[0:10])
# l2_clean = np.sum((x_test_poison[not_poisoned_idx]-key)**2,axis=(1,2,3))**(1/2)
# print('Mean clean l2: ', l2_clean[0:10])


input()

basemodel_poison.evaluate(x_test,y_test)
ConfusionMatrix(predictions=basemodel_poison.predict(x_test),
    Y=y_test,
    title='SoftMax Classifier Clean MNIST Data (n=10000)')
basemodel_poison.evaluate(x_backdoor,y_backdoor)
ConfusionMatrix(predictions=basemodel_poison.predict(x_backdoor),
    Y=y_true,
    title='SoftMax Classifier Backdoor MNIST Data (n=1000)')
basemodel_poison_rbf.evaluate(x_test,y_test)
ConfusionMatrix(predictions=basemodel_poison_rbf.predict(x_test),
    Y=y_test,
    title='RBF Classifier Clean MNIST Data (n=10000)')
basemodel_poison_rbf.evaluate(x_backdoor,y_backdoor)
ConfusionMatrix(predictions=basemodel_poison_rbf.predict(x_backdoor),
    Y=y_true,
    title='RBF Classifier Backdoor MNIST Data (n=1000)')
HistogramOfPredictionConfidence(P1=basemodel_poison_rbf.predict(x_test),
    Y1=y_test,
    P2=basemodel_poison_rbf.predict(x_backdoor),
    Y2=y_backdoor,
    title='RBF Classification Confidence')
HistogramOfPredictionConfidence(P1=basemodel_poison.predict(x_test),
    Y1=y_test,
    P2=basemodel_poison.predict(x_backdoor),
    Y2=y_backdoor,
    title='Softmax Classification Confidence')
plt.show()
input()

# cv2.imwrite('./images/MNIST_Poison_Example.png',x_test_poison[poisoned_idx[0]]*255)
# cv2.imwrite('./images/MNIST_Example.png',x_test[poisoned_idx[0]]*255)

#transfer
#basemodel_poison.transfer(weights=os.path.join(baseDir,'basemodel_poison.h5'),isRBF=True)
#basemodel_poison.train(x_train_poison,y_train_poison,saveTo=os.path.join(baseDir,'basemodel_poison_rbf_transfer.h5'),epochs=10)


PGD_reg_adv = ProjectedGradientDescentAttack(basemodel_poison.model,x_test[0:10],path=os.path.join(baseDir,'PGD_reg_adv.npy'))
basemodel_poison_rbf.evaluate(PGD_reg_adv,y_test[0:10])
labels = np.argmax(y_backdoor,axis=1)
y_poison = labels
y_poison = (y_poison-1)%10
y_backdoor = keras.utils.to_categorical(y_poison, 10)
basemodel_poison_rbf.evaluate(x_backdoor,y_backdoor)
basemodel_poison_rbf.evaluate(x_test,y_test)
print(basemodel_poison_rbf.predict(np.expand_dims(x_test[0],axis=0)))
x_test[0,26,26,:] = 1
x_test[0,26,24,:] = 1
x_test[0,25,25,:] = 1
x_test[0,24,26,:] = 1
# heat = basemodel_poison.generate_heatmap(x_test[0])
# cv2.imshow('Heatmap Version',heat.astype(np.uint8))
# cv2.waitKey(0)

print(basemodel_poison_rbf.predict(np.expand_dims(x_test[0],axis=0)))
input()
test = False
if (test):
    print('Testing models: \n')
    print('Base model')
    basemodel.evaluate(x_test,y_test)
    print('RBF base model')
    basemodel_rbf.evaluate(x_test,y_test)
    print('RBF heat model')
    heatmodel_rbf.evaluate(x_test_heat,y_test)

# adversarial images
num_adv = 100
y_adv = y_test[0:num_adv]
pgd_x_adv = ProjectedGradientDescentAttack(model=basemodel.model,
                                        X=x_test[0:num_adv],
                                        path=os.path.join(baseDir,'pgd_x_adv.npy'))
# convert to heatmap
pgd_x_adv_heat = basemodel.convert_to_heatmap(X=pgd_x_adv,
    path=os.path.join(baseDir,'pgd_x_adv_heat.npy'),
    visualize=False)
pgd_x_adv_heat /=255.

test = False
if (test):
    print('Testing models: \n')
    print('Base model')
    basemodel.evaluate(pgd_x_adv,y_adv)
    print('RBF base model')
    basemodel_rbf.evaluate(pgd_x_adv,y_adv)
    print('RBF heat model')
    heatmodel_rbf.evaluate(pgd_x_adv_heat,y_adv)

num_adv = 200
y_adv = y_test[0:num_adv]
fgsm_x_adv = FGSMAttack(model=basemodel.model,
                                        X=x_test[0:num_adv],
                                        path=os.path.join(baseDir,'fgsm_x_adv.npy'))
# convert to heatmap
fgsm_x_adv_heat = basemodel.convert_to_heatmap(X=fgsm_x_adv,
    path=os.path.join(baseDir,'fgsm_x_adv_heat.npy'),
    visualize=False)
fgsm_x_adv_heat_noiseless = denoiseImages(X=fgsm_x_adv_heat,
                                         path=os.path.join(baseDir,'fgsm_x_adv_heat_noiseless.npy'),
                                         visualize=False)
for i in range(fgsm_x_adv.shape[0]):
    cv2.imshow('a',fgsm_x_adv[i])
    cv2.waitKey(1000)

fgsm_x_adv_noiseless = denoiseImages(X=fgsm_x_adv*255.,
                                         path=os.path.join(baseDir,'fgsm_x_adv_noiseless.npy'),
                                         visualize=True)
fgsm_x_adv_heat /=255.
fgsm_x_adv_heat_noiseless /=255.
fgsm_x_adv_noiseless /= 255.
fgsm_x_adv=fgsm_x_adv_noiseless
test = True
if (test):
    print('Testing models: \n')
    print('Base model')
    basemodel.evaluate(fgsm_x_adv,y_adv)
    ComputeEmpiricalRobustness( x=x_test[0:num_adv],
                                x_adv=fgsm_x_adv,
                                y=basemodel.predict(x_test[0:num_adv]),
                                y_adv = basemodel.predict(fgsm_x_adv))
    ConfusionMatrix(predictions=basemodel.predict(fgsm_x_adv),
                    Y=y_adv,
                    title='Regular Confusion Matrix')

    print('RBF base model')
    basemodel_rbf.evaluate(fgsm_x_adv,y_adv)
    ComputeEmpiricalRobustness( x=x_test[0:num_adv],
                                x_adv=fgsm_x_adv,
                                y=basemodel_rbf.predict(x_test[0:num_adv]),
                                y_adv = basemodel_rbf.predict(fgsm_x_adv))
    ConfusionMatrix(predictions=basemodel_rbf.predict(fgsm_x_adv),
                    Y=y_adv,
                    title='RBF Regular Confusion Matrix')

    print('RBF heat model')
    heatmodel_rbf.evaluate(fgsm_x_adv_heat,y_adv)
    ComputeEmpiricalRobustness( x=x_test_heat[0:num_adv],
                                x_adv=fgsm_x_adv_heat,
                                y=heatmodel_rbf.predict(x_test_heat[0:num_adv]),
                                y_adv = heatmodel_rbf.predict(fgsm_x_adv_heat))
    ConfusionMatrix(predictions=heatmodel_rbf.predict(fgsm_x_adv_heat),
                    Y=y_adv,
                    title='RBF Heat Confusion Matrix')
    plt.show()
    print('Computing robustness')


