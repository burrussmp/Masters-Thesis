from keras.applications.vgg16 import VGG16
import os
import cv2
import numpy as np
from generate_model import generate_model
import argparse
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import mean_squared_error

os.environ["CUDA_VISIBLE_DEVICES"]="0" # first gpu
os.environ["CUDA_VISIBLE_DEVICES"]="1" # second gpu
os.environ["CUDA_VISIBLE_DEVICES"]="2" # second gpu
os.environ["CUDA_VISIBLE_DEVICES"]="3" # second gpu

def removeTargets(path):
    for i in range(10):
        path = path.replace('target_' + str(i) +'_','')
    return path
def removeAdditionalExtension(path):
    withoutExtension = os.path.splitext(path)[0][:]
    rest = os.path.splitext(withoutExtension)[1][:]
    if (rest == ''):
        return withoutExtension + os.path.splitext(path)[1][:]
    else:
        return withoutExtension
ATTACKS = ['FGSM','Ensemble']
def test(dirs,test_data_dir,model_generator,model,args,attack=None):
    y = 0
    correct = 0.0
    total = 0.0
    accuracy = 0.0
    disagree = 0.0
    for category in dirs:
        print(category)
        path = os.path.join(test_data_dir,category)
        for image in os.listdir(path):
            fileName = os.path.join(path,image)
            y_hat2 = None
            if (fileName.endswith('.npy')):
                x = np.load(fileName)
            elif (fileName.endswith('.jpg') or fileName.endswith('.JPEG') or fileName.endswith('.png')):
                x = cv2.imread(os.path.join(path,image))
            if (attack):
                normal = fileName.replace(attack+'_','').replace('/'+attack,'').replace('_transfer','')
                normal = normal.replace('VGG16','test').replace('ResNet50','test').replace('InceptionV3','test').replace('DaveII','test')
                normal = removeAdditionalExtension(normal)
                normal = removeTargets(normal)
                x2 = cv2.imread(normal)
                x2 = model_generator.preprocess(x2)
                pred2 = model.predict(x2)
                y_hat2 = np.argmax(pred2[0])
            x = model_generator.preprocess(x)
            pred = model.predict(x)
            y_hat = np.argmax(pred[0])
            # print('y_hat',y_hat,'y',y)
            # x = model_generator.unprocess(x)
            # cv2.imshow('d',x.astype(np.uint8))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            #break
            #print(y_hat2,y_hat,y)
            if (y_hat2 != None and y_hat2 != y_hat):
                if (not args.mse):
                    disagree += 1.0
            if (not args.mse):
                if y == y_hat:
                    correct += 1.0
                total += 1.0
            else:
                correct += (y - y_hat)**2
                total += 1
        y += 1
    #return
    accuracy = correct/total
    if (not args.mse):
        print('Accuracy: %0.4f' %(accuracy))
        print('Correct: %d'%(correct))
        print('Total: %d' %(total))
        if (y_hat2 != None):
            print('Adversarial mistakes: %0.4f' %(disagree/total))
            print('Disagree %d' %(disagree))

    else:
        print('MSE: %0.2f' %(accuracy))
        print(correct)
        print(total)

def trythis(args):
    assert os.path.exists(args.dir), \
        'The test directory specified does not exist!'
    model_generator = generate_model(args.model)
    numClasses = len(os.listdir(args.dir))
    datagen = ImageDataGenerator()
    model = model_generator.initialize(args.weights,classes=numClasses,useTransfer=False,optimizer='sgd',reset=False)
    test_data = datagen.flow_from_directory(
        args.dir,
        target_size =model_generator.getInputSize(),
        batch_size = 200,
        class_mode = "categorical",
        shuffle=True)
    # get a batch of data
    batchX, batchy = test_data.next()
    predictions = model.predict(batchX)
    mse = mean_squared_error(np.argmax(predictions, axis=1), np.argmax(batchy, axis=1))
    print('Accuracy on adversary images (defense included): {}'.format(mse))

def main(args):
    trythis(args)
    return
    assert os.path.exists(args.dir), \
        'The test directory specified does not exist!'
    model_generator = generate_model(args.model)
    numClasses = len(os.listdir(args.dir))
    model = model_generator.initialize(args.weights,classes=numClasses,useTransfer=False,optimizer='sgd',reset=False)
    attack = os.path.basename(args.dir).replace('_' + model_generator.getName(),'').replace('heatmap_','').replace('_transfer','')
    if (attack in ATTACKS):
        folders = os.listdir(args.dir)
        idx = folders.index('test')
        test_data_dir = os.path.join(args.dir,folders[idx])
        dirs = os.listdir(os.path.join(args.dir,folders[idx]))
        dirs.sort()
        print('Results of test subset: ')
        test(dirs,test_data_dir,model_generator,model,args)
        folders.remove('test')
        test_data_dir = os.path.join(args.dir,folders[0])
        dirs = os.listdir(os.path.join(args.dir,folders[0]))
        dirs.sort()
        print('Results of adversary subset: ')
        test(dirs,test_data_dir,model_generator,model,args,attack)
    else:
        test_data_dir = args.dir
        dirs = os.listdir(test_data_dir)
        dirs.sort()
        print('Results of total data set')
        test(dirs,test_data_dir,model_generator,model,args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a neural net!')
    parser.add_argument('--test_dir','-D', metavar='D', type=str,required=True,
                    help='Directory where data is partitioned for test',dest='dir')
    parser.add_argument('--model', '-M', metavar='M', type=str,required=True,
                    help='Model to load to evaluate on the data set',dest='model')
    parser.add_argument('--model_weights', '-W', metavar='W', type=str,required=True,
                    help='Model weights to initialize the model.',dest='weights')
    parser.add_argument('--MSE',action='store_true',
                    help='Calculate MSE (otherwise accuracy)',dest='mse')
    args = parser.parse_args()
    main(args)

    #python3 test.py -D '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/inceptionv3_dataset_partitioned/' -M 'InceptionV3' -W '../weights/InceptionV3/12_12_2019_17_00_900985.h5'
    #python3 test.py -D '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/resnet50_dataset_partitioned/' -M 'ResNet50' -W '../weights/ResNet50/12_12_2019_32_47_919130.h5'
