from keras import backend as K
import numpy as np
import cv2
import tensorflow as tf
import os
from heatmap import LRP
from threading import Thread
import random
# change beta IMPORTANT
from generate_model import generate_model

def PoisonGeneration(g,target,base,learningRate,iterations,classes=1000,savePath='/',fileName='',featRepTensor='',inputImgTensor='',tarFeatRepPL='',forward_loss='',grad_op=''):

    # compute a good coefficient of similarity according to Poisson Frog paper
    bI_shape = np.squeeze(base).shape
    beta = 0.25*(25088/float(bI_shape[0]*bI_shape[1]*bI_shape[2]))**2
    beta=0.25
    print('Beta is:', beta)
    # for op in g.as_graph_def().node:
    #     print(str(op.name))
    #initializations
    last_M_objs = []
    M = 40
    decayCoef = 0.5
    rel_change_val = 1e5
    targetFeatRep = sess.run(featRepTensor, feed_dict={inputImgTensor: target})      #get the feature reprsentation of the target
    old_image = base                                                                 #set the poison's starting point to be the base image
    old_featRep = sess.run(featRepTensor, feed_dict={inputImgTensor: base})      #get the feature representation of current poison
    old_obj = np.linalg.norm(old_featRep - targetFeatRep) + beta*np.linalg.norm(old_image - base)
    last_M_objs.append(old_obj)

    for i in range(iterations):
        if i % 10 == 0:
            the_diffHere = np.linalg.norm(old_featRep - targetFeatRep)      #get the diff
            theNPimg = old_image
            print("iter: %d | diff: %.3f | obj: %.3f"%(i,the_diffHere,old_obj))
            print(" (%d) Rel change =  %0.5e   |   lr = %0.5e |   obj = %0.10e"%(i,rel_change_val,learningRate,old_obj))

            print(savePath+'/'+fileName)
            cv2.imwrite(savePath+'/'+fileName, np.squeeze(old_image*255).astype(np.uint8))
        # forward step
        grad_now = sess.run(grad_op, feed_dict={inputImgTensor: old_image, tarFeatRepPL:targetFeatRep})      #evaluate the gradient at the current point
        currentImage = old_image - learningRate*np.squeeze(np.array(grad_now))
        # backward step
        new_image = (beta*learningRate*base + currentImage)/(beta*learningRate + 1)
        new_featRep = sess.run(featRepTensor, feed_dict={inputImgTensor: new_image})
        new_obj = np.linalg.norm(new_featRep - targetFeatRep) + beta*np.linalg.norm(new_image - base)

        # check stopping condition:  compute relative change in image between iterations
        rel_change_val =  np.linalg.norm(new_image-old_image)/np.linalg.norm(new_image)

        avg_of_last_M = sum(last_M_objs)/float(min(M,i+1)) #find the mean of the last M iterations
        # If the objective went up, then learning rate is too big.  Chop it, and throw out the latest iteration
        if  new_obj >= avg_of_last_M and (i % M/2 == 0):
            learningRate *= decayCoef
            new_image = old_image
        else:
            old_image = new_image
            old_obj = new_obj
            old_featRep = new_featRep

        if i < M-1:
            last_M_objs.append(new_obj)
        else:
            #first remove the oldest obj then append the new obj
            del last_M_objs[0]
            last_M_objs.append(new_obj)
    return np.squeeze(old_image).astype(np.uint8)



def watermark(base,target,opacity):
    new_base = cv2.addWeighted(target, opacity, base, 1.0-opacity, 0.0)
    return new_base

def poison(targetPath,basePath,g,savePath,file,featRepTensor,inputImgTensor,tarFeatRepPL,forward_loss,grad_op):
    target = cv2.resize(cv2.imread(targetPath),(299, 299)).astype(np.float32).reshape(1,299,299,3)
    base = cv2.resize(cv2.imread(basePath),(299, 299)).astype(np.float32).reshape(1,299,299,3)
    #base = watermark(base,target,0.3)
    target /= 255.
    base /= 255.
    fileName = 'base_' + baseImg.replace('.JPEG','') + '_target_' + file
    print(fileName)
    P = PoisonGeneration(g=g,target=target,base=base,learningRate=0.01,iterations=3000,classes=2,savePath=savePath,fileName=fileName,featRepTensor=featRepTensor,inputImgTensor=inputImgTensor,tarFeatRepPL=tarFeatRepPL,forward_loss=forward_loss,grad_op=grad_op)

if __name__ == "__main__":
    model_generator = generate_model('InceptionV3',False)
    model = model_generator.initialize('../weights/InceptionV3/poison_weights_inceptionv3.h5',classes=2,useTransfer=False,optimizer='Adam',reset=False)
    p = '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/inceptionv3_dataset_partitioned_poison/train/n01806143/poisonbase_n01806143_992_base_target_n01622779_1037.JPEG'
    x = cv2.imread(p)
    pred = model.predict(model_generator.preprocess(x))
    print(pred)
    p = '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/inceptionv3_dataset_partitioned_poison/poison/n01622779_1037.JPEG'
    x = cv2.imread(p)
    pred = model.predict(model_generator.preprocess(x))
    print(pred)
    #
    # sess = K.get_session() # get the session
    # g = sess.graph # get the graph
    #
    # dirpath = '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/inceptionv3_dataset_partitioned_poison/poison'
    # baseImg = "n01806143_992_base.JPEG"
    # basePath = os.path.join(dirpath,baseImg)
    # targetPath = os.path.join(dirpath,'n01622779_1037.JPEG')
    # savePath = dirpath
    # featRepTensor = g.get_tensor_by_name('avg_pool/Mean:0')
    # inputImgTensor = g.get_tensor_by_name('input_1:0')
    # tarFeatRepPL = tf.placeholder(tf.float32,[None,2048])
    # forward_loss = tf.norm(featRepTensor - tarFeatRepPL)
    # grad_op = tf.gradients(forward_loss, inputImgTensor)
    # t1 = Thread(target = poison, args=(targetPath,basePath,g,savePath,'n01622779_1037.JPEG',featRepTensor,inputImgTensor,tarFeatRepPL,forward_loss,grad_op,))
    # t1.start()
