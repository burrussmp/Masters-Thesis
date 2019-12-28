# author
# Matthew Burruss

import os
import argparse
import math
import random
import shutil
from generate_adversarial_attack import generate_adversarial_attack
from generate_model import generate_model
from heatmap import LRP
import cv2
import numpy as np

def main(args):
    # get the attack
    attack = generate_adversarial_attack(args.attack,args.dir)
    numClasses = len(os.listdir(os.path.join(args.dir,'train')))
    # get the model
    model_generator = generate_model(args.model)
    model = model_generator.initialize(args.weights,classes=numClasses,reset=False)
    lrp = LRP(model,model_generator.preprocess,modelType=model_generator.getName())
    assert os.path.exists(args.dir), \
        'Path does not exist!'
    subdir = os.listdir(args.dir)
    assert 'test' in subdir, \
        'Partitioned directory does not have test'
    # heatmap_test = os.listdir(os.path.join(args.dir,attack.name()))
    # os.mkdir(adversarial_dir)
    subdirectory = 'heatmap_' + attack.name()+'_'+model_generator.getName()
    folderNames = ['heatmap_train','heatmap_val','heatmap_test',os.path.join(subdirectory,'test'),os.path.join(subdirectory,attack.name())]
    targetPaths = [os.path.join(args.dir,folderType) for folderType in folderNames] # creates base directory with heatmap_train, heatmap_val, heatmap_attack
    sourcePaths = []
    targetPathsCopy = targetPaths[:]
    for i in range(len(targetPathsCopy)):
        if not os.path.exists(targetPathsCopy[i]):
            sourcePaths.append(targetPathsCopy[i].replace('heatmap_',''))
        else:
            targetPaths.remove(targetPathsCopy[i])
    print('Making the directory',os.path.join(args.dir,subdirectory))
    os.mkdir(os.path.join(args.dir,subdirectory))
    [os.mkdir(targetPath) for targetPath in targetPaths]
    idx = 0
    for sourcePath in sourcePaths:
        for className in os.listdir(sourcePath):
            allFiles = os.listdir(os.path.join(sourcePath,className)) # all data for /train/class1 for example
            targetPath = os.path.join(targetPaths[idx],className)
            os.mkdir(targetPath)
            for file in allFiles:
                x = os.path.join(sourcePath,className,file)
                if (x.endswith('.npy')):
                    x = np.load(x)
                x = lrp.generate_heatmap(x)
                file = file.replace('.npy','.jpg')
                target = os.path.join(targetPath,'heatmap_'+file)
                cv2.imwrite(target,x)
                # open image, convert to heatmap, and save to directory
                #[shutil.copyfile(os.path.join(sourcePath,className,file),os.path.join(targetPath,'heatmap_'+file)) for file in allFiles]
        idx += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--base_dir','-D', metavar='D', type=str,required=True,
                    help='Directory of partitioned data',dest='dir')
    parser.add_argument('--attack', '-A', metavar='A', type=str,required=True,
                    help='Attack',dest='attack')
    parser.add_argument('--model', '-M', metavar='M', type=str,required=True,
                    help='Model to load to evaluate on the data set',dest='model')
    parser.add_argument('--model_weights', '-W', metavar='W', type=str,required=True,
                    help='Model weights to initialize the model to create the heatmap dataset',dest='weights')
    args = parser.parse_args()
    print(args)
    main(args)

# python3 generate_heatmap_dataset.py -D '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/vgg16_dataset_partitioned' -A 'FGSM' -M 'VGG16' -W '../weights/VGG16/12_11_2019_42_20_562956.h5'
# python3 generate_heatmap_dataset.py -D '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/inceptionv3_dataset_partitioned' -A 'FGSM' -M 'InceptionV3' -W '../weights/InceptionV3/12_12_2019_43_08_551535.h5

# todo Add base model weights
# todo load the model
# todo integrate generation of heatmap
# todo add strategy for LRP
# todo add name to path that specifies which model
