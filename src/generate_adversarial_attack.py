import numpy as np
import keras.backend as K
import os
import cv2
import matplotlib.pyplot as plt
import copy
import tensorflow as tf

config = tf.ConfigProto( device_count = {'GPU': 4 , 'CPU': 20} )
sess = tf.Session(config=config)
K.set_session(sess)
class generate_adversarial_attack:
    def __init__(self,attack,baseDir):
        attacks = ['FGSM','T-FGSM','RandomNoise','Ensemble']
        assert attack in attacks, \
            'Not a defined attack!'
        assert os.path.exists(baseDir), \
            'Path does not exist!'
        self.attackType = attack
        self.baseDir = baseDir

    def name(self):
        return self.attackType

    def FGSM(self,model,x,classes,target_class,epochs=20):
        epsilon = 0.03
        x_adv = x
        x_noise = np.zeros_like(x)
        sess = K.get_session()
        preds = model.predict(x_adv)
        prev = preds[0][target_class]
        print('Initial prediction:', np.argmax(preds[0]))
        x_advcpy = x_adv
        for i in range(epochs):
            target = K.one_hot(target_class, classes)
            loss = -1*K.categorical_crossentropy(target, model.output)
            grads = K.gradients(loss, model.input)
            delta = K.sign(grads[0])
            x_noise = x_noise + delta
            x_adv = x_adv + epsilon*delta
            #with tf.Session() as sess:  print(delta.eval())
            x_adv = sess.run(x_adv, feed_dict={model.input:x})
            preds = model.predict(x_adv)
            if (prev > preds[0][target_class]):
                return x_advcpy
            x_advcpy = x_adv
            prev = preds[0][target_class]
            if ( preds[0][target_class] > 1./classes*1.2):
                break
            print(i)

        preds = model.predict(x_adv)
        print('Final prediction amount:', np.argmax(preds[0]))
        return x_adv

    def attackFGSM(self,model,x,classes,target_class,preprocess,unprocess):
        if (type(x) == str):
            assert os.path.isfile(x), \
                'Path is not a file!'
            x = cv2.imread(x)
        x = preprocess(x)
        img = self.FGSM(model,x,classes,target_class,epochs=20)
        return unprocess(img)

    def attackEnsemble(self,models,x,classes,target_class,model_generators):
        # open if necessary
        if (type(x) == str):
            assert os.path.isfile(x), \
                'Path is not a file!'
            x = cv2.imread(x)
        x_cpy = copy.deepcopy(x) # make a deep copy of the input
        for i in range(len(model_generators)):
            preprocess = model_generators[i].preprocess # get preprocess
            unprocess = model_generators[i].unprocess # get unprocess
            x = preprocess(x) # preprocess x
            x_adv = self.FGSM(models[i],x,classes,target_class,epochs=20) # get the output
            x_adv = unprocess(x_adv) # unprocess the adversarial image

            goodAdv = True # make deep copy of adversarial image and assume that it is good
            adv_cpy = copy.deepcopy(x_adv)

            for j in range(len(model_generators)): # for every model
                x_adv = model_generators[j].preprocess(x_adv) # check prediction
                pred = np.argmax(models[j].predict(x_adv)[0]) # if we don't predict target_class we have failed
                if (pred != target_class):
                    goodAdv = False
                x_adv = copy.deepcopy(adv_cpy) # go back to original adversarial image
            # only return if all were fooled
            if (goodAdv):
                return good_adv
            x = copy.deepcopy(x_cpy)
        return None # no adversarial image fooled both


    def numClasses(self):
        path = os.path.join(self.baseDir,'train')
        return len(os.listdir(path))


# from generate_model import generate_model
# attack = generate_adversarial_attack('FGSM','./')
# model_generator = generate_model('InceptionV3')
# model = model_generator.initialize('../weights/inception_v3_weights_tf_dim_ordering_tf_kernels.h5',classes=2,useTransfer=True,optimizer='sgd')
# x = '../ignoreme/dogs/dog.jpg'
# preprocess = model_generator.getPreProcessingFunction()
# x_adv = attack.attack(model,x,2,1,preprocess)
# import matplotlib.pyplot as plt
# plt.show(x_adv)
# plt.show()
