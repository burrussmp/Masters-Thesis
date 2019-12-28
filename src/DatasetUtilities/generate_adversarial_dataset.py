# author
# Matthew Burruss

import os
import argparse
import math
import random
import shutil
from generate_adversarial_attack import generate_adversarial_attack
import numpy as np
from generate_model import generate_model
import cv2
import tensorflow as tf
import keras
os.environ["CUDA_VISIBLE_DEVICES"]="0" # first gpu
os.environ["CUDA_VISIBLE_DEVICES"]="1" # second gpu
os.environ["CUDA_VISIBLE_DEVICES"]="2" # second gpu
os.environ["CUDA_VISIBLE_DEVICES"]="3" # second gpu

AllModels = ['VGG16','ResNet50','InceptionV3']

def randomlySelectTargetClass(baseDir,baseClass,modelType):
    path = os.path.join(baseDir,'train')
    dirs = os.listdir(path)
    dirs.sort()
    dirs_copy = dirs[:]
    dirs.remove(baseClass)
    targetClass = np.random.choice(dirs,1)
    if (modelType == 'DaveII'):
        if baseClass != '9':
            dirs.remove(str(int(baseClass)+1))
            if baseClass != '8':
                dirs.remove(str(int(baseClass)+2))
        if baseClass != '0':
            dirs.remove(str(int(baseClass)-1))
            if baseClass != '1':
                dirs.remove(str(int(baseClass)-2))
        targetClass = np.random.choice(dirs,1)
    return dirs_copy.index(targetClass[0])

def main(args):
    assert os.path.exists(args.dir), \
        'Path does not exist!'

    attack = generate_adversarial_attack(args.attack,args.dir)

    numClasses = len(os.listdir(os.path.join(args.dir,'test')))
    if (attack.name() == 'FGSM'):
        assert args.model != None, \
            'Model cannot be empty for this attack'
        assert args.weights != None, \
            'Weights cannot be empty for this attack'
        model_generator = generate_model(args.model)
        model = model_generator.initialize(args.weights,classes=numClasses,useTransfer=False,optimizer=args.optimizer,reset=False)

    elif (attack.name() == 'Ensemble'):
        assert args.model in AllModels, \
            'Model must be either VGG16, ResNet50, or InceptionV3'
        ensemble_models = AllModels[:]
        ensemble_models.remove(args.model)
        model_generators = [generate_model(model) for model in ensemble_models]
        weights = [os.path.join('../weights/',model_generator.getName(),'ensemble_weights_'+model_generator.getName().lower()+'.h5') \
            for model_generator in model_generators]
        for weight in weights:
            assert os.path.isfile(weight),\
                'Weight does not exist!'
        models = [model_generator.initialize(weights[i],numClasses,useTransfer=False,optimizer=args.optimizer,reset=False) for i in range(len(ensemble_models))]

    subdir = os.listdir(args.dir)
    assert 'test' in subdir, \
        'Partitioned directory does not have test'

    adversarial_dir = os.path.join(args.dir,attack.name()+'_'+args.model)
    os.mkdir(adversarial_dir)

    def sampleAndRemove(arr,k):
        sub_arr = random.sample(arr,k)
        for item in sub_arr:
            arr.remove(item)
        return sub_arr

    folderNames = ['test',attack.name()]
    targetPaths = [os.path.join(adversarial_dir,folderType) for folderType in folderNames]
    [os.mkdir(targetPath) for targetPath in targetPaths]
    classes = os.listdir(os.path.join(args.dir,'test'))
    classes.sort()
    for className in classes:
        allTestData = os.listdir(os.path.join(args.dir,'test',className))
        numItems = len(allTestData)
        test = math.floor(args.test * numItems)
        adv = numItems - test
        samples = [test,adv]
        dataSplit = [sampleAndRemove(allTestData,sample) for sample in samples]
        idx = 0
        for data in dataSplit:
            targetPath = os.path.join(targetPaths[idx],className)
            os.mkdir(targetPath)
            # otherwise perform the test
            i= 0
            if (os.path.basename(targetPaths[idx]) != 'test'):
                for file in data:
                    print('File: ',str(i),'/',len(data))
                    target_class = randomlySelectTargetClass(args.dir,className,args.model)
                    x = os.path.join(args.dir,'test',className,file)
                    # get adversarial image
                    x_adv = None
                    print('Class: ' ,className,'targeting:',target_class)
                    if (attack.name() == 'FGSM'):
                        x_adv = attack.attackFGSM(model,x,numClasses,target_class,model_generator.preprocess,model_generator.unprocess)
                    elif(attack.name()=='Ensemble'):
                        x_adv = attack.attackEnsemble(models,x,numClasses,target_class,model_generators)
                        if (x_adv == None): # if couldn't find an image to fool all
                            continue
                    else:
                        raise ValueError('No other attack has been defined yet')

                    # cv2.imshow('adv',x_adv.astype(np.uint8))
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    newName = 'target_' + str(target_class) + '_' + file
                    arr = np.asarray(x_adv)
                    np.save(os.path.join(targetPath,newName),arr)
                    i += 1
            # otherwise copy over
            else:
                [shutil.copyfile(os.path.join(args.dir,'test',className,file),os.path.join(targetPath,file)) for file in data]
            idx += 1
    ## print and return results
    print("#############################")
    print("RESULTS OF ADVERSARIAL DATASET PREPARATION")
    print("#############################")
    for folder in folderNames:
        folderPath = os.path.join(adversarial_dir,folder)
        print(folderPath)
        classes = os.listdir(folderPath)
        print(classes)
        print('The ' + folder + ' subdirectory has ' + str(len(classes)) + ' classes.')
        sizeOfClass = [len(os.listdir(os.path.join(folderPath,mclass))) for mclass in classes]
        print('The size of the classes: ' + str(sizeOfClass))
        print('Total size of ' + folder + ' is ' + str(sum(sizeOfClass)) + '\n')
if __name__ == "__main__":
    config = tf.ConfigProto( device_count = {'GPU': 4 , 'CPU': 20} )
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--base_dir','-D', metavar='D', type=str,required=True,
                    help='Directory of partitioned data',dest='dir')
    parser.add_argument('--attack', '-A', metavar='A', type=str,required=True,
                    help='Attack',dest='attack')
    parser.add_argument('--testSize', '-T', metavar='T', type=float,required=True,
                    help='percentage of test data to keep unchanged',dest='test')
    parser.add_argument('--adversarialSize', '-X', metavar='X', type=float,required=True,
                    help='percentage of test data to convert to adversary',dest='adv')
    parser.add_argument('--model', '-M', metavar='M', type=str,required=True,
                    help='Model to load to evaluate on the data set',dest='model')
    parser.add_argument('--model_weights', '-W', metavar='W', type=str,required=False,
                    help='Model weights to initialize the model.',dest='weights')
    parser.add_argument('--optimizer', '-O', metavar='O', type=str,required=False,
                    help='Valid optimizers: adam,sgd',dest='optimizer')
    args = parser.parse_args()
    main(args)


# example python3 generate_adversarial_dataset.py -D '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/vgg16_dataset_partitioned' -A 'FGSM' -T 0.66 -X 0.33 -M 'VGG16' -W '../weights/VGG16/12_11_2019_42_20_562956.h5'
