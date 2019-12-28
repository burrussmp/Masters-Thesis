# author
# Matthew Burruss

import os
import argparse
import math
import random
import shutil

def main(args):
    assert os.path.exists(args.dir), \
        'Path does not exist!'
    assert args.test + args.train + args.val == 1, \
        'Percentages do not add up to !'
    assert args.train > args.val >= args.test, \
        'Incorrect partitioning of data set'
    if (args.dir[-1] == '/'): args.dir = args.dir[:-1]
    parentDir = os.path.join(os.path.dirname(args.dir),args.name)
    assert not os.path.exists(parentDir), \
        'Parent directory already exists! Select a new name or remove.'
    os.mkdir(parentDir)
    def sampleAndRemove(arr,k):
        sub_arr = random.sample(arr,k)
        for item in sub_arr:
            arr.remove(item)
        return sub_arr
    folderNames = ['train','val','test']
    targetPaths = [os.path.join(parentDir,folderType) for folderType in folderNames]
    [os.mkdir(targetPath) for targetPath in targetPaths]
    for folder in os.listdir(args.dir):
        path = os.path.join(args.dir,folder)
        allData = os.listdir(path)
        numItems = len(allData)
        train = math.floor(args.train * numItems)
        val = math.floor(args.val * numItems)
        test = numItems - train - val
        samples = [train,val,test]
        dataSplit = [sampleAndRemove(allData,sample) for sample in samples]
        idx = 0
        for data in dataSplit:
            targetPath = os.path.join(targetPaths[idx],folder)
            os.mkdir(targetPath)
            [shutil.copyfile(os.path.join(path,file),os.path.join(targetPath,file)) for file in data]
            idx += 1
    ## print and return results
    print("#############################")
    print("RESULTS OF DATASET PREPARATION")
    print("#############################")
    for folder in folderNames:
        folderPath = os.path.join(parentDir,folder)
        classes = os.listdir(folderPath)
        print('The ' + folder + ' subdirectory has ' + str(len(classes)) + ' classes.')
        sizeOfClass = [len(os.listdir(os.path.join(folderPath,mclass))) for mclass in classes]
        print('The size of the classes: ' + str(sizeOfClass))
        print('Total size of ' + folder + ' is ' + str(sum(sizeOfClass)) + '\n')
    print('Returning the parent directory: ', parentDir)
    return parentDir
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--target_dir','-D', metavar='D', type=str,required=True,
                    help='Target directory containing folders representing classes',dest='dir')
    parser.add_argument('--trainingPercentage', '-T', metavar='T', type=float,required=True,
                    help='Percentage of data to place in training',dest='train')
    parser.add_argument('--validationPercentage' ,'-V', metavar='V', type=float,required=True,
                    help='Percentage of data to place in validation',dest='val')
    parser.add_argument('--testingPercentage' ,'-E', metavar='E', type=float,required=True,
                    help='Percentage of data to place in testing',dest='test')
    # parser.add_argument('--adversarialPercentage' ,'-A', metavar='A', type=float,required=True,
    #                 help='Percentage of data to place in adversarial',dest='adv')
    parser.add_argument('--name' ,'-N', metavar='N', type=str,required=True,
                    help='Name of dataset ex. vgg16_data',dest='name')
    args = parser.parse_args()
    main(args)


    # example
    # python3 generate_dataset.py -D '/media/burrussmp/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/data_set_1' -T 0.7 -V 0.15 -E .15 -N 'Dataset_owl_and_peacock'
    # python3 generate_dataset.py -D '/media/burrussmp/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/vgg16_dataset' -T 0.7 -V 0.15 -E .15 -N 'vgg16_dataset_partitioned'
