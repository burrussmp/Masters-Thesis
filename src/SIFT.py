import cv2
import numpy as np
import csv
import json
import os


"""
saveSIFTcurstd = math.sqrt(curvar)
--------------------
Parameters
targetPath: path to a JSON object to save ex. ./test.json
keyPoints: List of size n of Cv2.KeyPoint
descriptpors: nx128 numpy array of descriptors
--------------------
Saves JSON file
"""
def saveSIFT(targetPath,keyPoints,descriptors):
    try:
        index = {
            "targetPath": targetPath,
            "type" : 'KeyPoint',
            "KeyPoints": [],
            'description' : descriptors.tolist()
        }
        index["KeyPoints"].append(temp)
        # Dump the keypoints
    with open(targetPath,"w") as csvfile:
        json.dump(index,csvfile)
"""
readSIFT
--------------------
Parameters
targetPath: path to a JSON object to read
--------------------
Return
list
[0] : List of size n of Cv2.KeyPoint
[1] : nx128 numpy array of descriptors
"""
def readSIFT(targetPath):
    with open(targetPath) as json_file:
        data = json.load(json_file)
        print("re-creating keypoints for: \n" + data["targetPath"])
        kp = []
        descriptors = np.zeros((len(data["KeyPoints"]),128))
        for point in data["KeyPoints"]:
            temp = cv2.KeyPoint(x=point["x"],
                y=point["y"],
                _size=point["_size"],
                _angle=point["_angle"],
                _response=point["_response"],
                _octave=point["_octave"],
                _class_id=point["_class_id"])
            kp.append(temp)
        descriptors = np.array(data["description"])
        return [kp,descriptors]

"""
createDataset
--------------------
Parameters
rootPath: A directory expecting the following format
- Root/
   |__ClassA/
        |___image1,image2,image3...imagen
   |__ClassB/
        |___image1,image2,image3...imagen
    .
    .
    .
    |__ClassA/
    |___image1,image2,image3...imagen

targetPath: Directory where the results of SIFT will be stored.
- targetPath/
   |__ClassA/
        |___data_image1,data_image2,data_image3...data_imagen
   |__ClassB/
        |___data_image1,data_image2,data_image3...data_imagen
    .
    .
    .
    |__ClassA/
    |___data_image1,data_image2,data_image3...data_imagen
--------------------
"""
def createDataset(rootPath,targetPath):
    # create a SIFT object
    sift = cv2.xfeatures2d.SIFT_create()
    # for all the folders in the root path
    i = 0
    for folder in os.listdir(rootPath):
        try:
            os.mkdir(targetPath + '/' + folder)
        except OSError:
            print("ERROR: Could not create folder")
        else:
            print("Created Folder: " + targetPath + '/' + folder )
        # for all the images in that class
        j = 0
        for image in os.listdir(rootPath + "/" + folder):
            # read in the image
            if not os.path.exists(targetPath + '/' + folder + '/' + 'data' + image.replace('.JPEG','.json')):
                try:
                    img = cv2.imread(rootPath + "/" + folder + '/' + image)
                    # convert to gray scale
                    img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    # calculate keypoints and descriptors of keypoints using SIFT
                    keypoints_sift, descriptors = sift.detectAndCompute(img, None)
                    # save results as a JSON
                    target = targetPath + '/' + folder + '/' + 'data' + image.replace('.JPEG','.json')
                    saveSIFT(target,keypoints_sift,descriptors)
                except:
                    print('Error: skipping')
                print("Folder " +str(i) +": JSON: " + str(j))
            j +=1
        i += 1

def readDatasets(rootPath):
    features = []
    i = 0
    for folder in os.listdir(rootPath):
        j = 0
        for json in os.listdir(rootPath + '/' +folder):
            print("Folder " +str(i) +": JSON: " + str(j))
            [kp,descriptor] = readSIFT(rootPath + '/' +folder + '/' + json)
            if (len(features) == 0):
                features = descriptor
            else:
                features = np.concatenate((features,descriptor),axis=0)
            j += 1
        i+=1
    return features

from sklearn.cluster import KMeans,MiniBatchKMeans
def CodeBookGeneration(features,batch=False):
    kmeans = None
    if (batch):
        batch_size = int(features.shape[0]/1000)
        kmeans = MiniBatchKMeans(n_clusters=VOCABLENGTH, batch_size=batch_size,verbose=1).fit(features)
    else:
        kmeans = KMeans(n_clusters=VOCABLENGTH,verbose=1).fit(features)
    # get the features for a given class
    codebook = kmeans.cluster_centers_
    return codebook,kmeans

#D(i,j) = sum (X(:,i) - Y(:,j)).^2
def pairwisedifference(X,Y):
    D = np.zeros((X.shape[0],Y.shape[0]))
    i = 0
    for col1 in X:
        j = 0
        for col2 in Y:
            D[i,j] = (np.sum(np.square(col1-col2)))
            j += 1
        i += 1
    return D

def BagOfSIFT(rootPath,vocab,train_or_val,localSave):
    print('Creating Bag of SIFT')
    i = 0
    sift = cv2.xfeatures2d.SIFT_create()
    for folder in os.listdir(rootPath):
        if (os.path.exists(localSave + '/class_' + str(VOCABLENGTH) + '_' +train_or_val + folder + '.txt')):
            continue
        # for all the images in that class
        bagofsift = np.zeros((len(os.listdir(rootPath + "/" + folder)),VOCABLENGTH))
        j = 0
        for image in os.listdir(rootPath + "/" + folder):
            # read in the image
            try:
                img = cv2.imread(rootPath + "/" + folder + '/' + image)
                # convert to gray scale
                img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                # calculate keypoints and descriptors of keypoints using SIFT
                keypoints_sift, descriptors = sift.detectAndCompute(img, None)
                # find pairwise difference
                D = pairwisedifference(vocab,descriptors)
                # find minimum distance vocab word for each feature
                min_indices = np.argmin(D.T,axis=1)
                [hist,edges] = np.histogram(min_indices,bins=VOCABLENGTH,range=(0,VOCABLENGTH),density=True)
                bagofsift[j,:] = hist
                j += 1
            except:
                print('Error: skipping')
            print("Folder " +str(i) +": JSON: " + str(j))
        #saveMe = bagofsift[~np.all(bagofsift[:,:] == 0, axis=1)]
        i = i+1
        j = 0
        np.savetxt(localSave + '/class_' + str(VOCABLENGTH) + '_' +train_or_val + folder + '.txt',bagofsift)

def loadData(train_or_val,localSave):
    prefixed = [filename for filename in os.listdir(localSave) if filename.startswith("class_") and train_or_val in filename and str(VOCABLENGTH)+'_' in filename]
    features = []
    labels = []
    for class_features in prefixed:
        if ('0149' in class_features):
            class_label = 0
        else:
            class_label = 1
        feature = np.loadtxt(localSave + '/' + class_features)
        mask = np.all(np.isnan(feature), axis=1)
        feature = feature[~mask]
        if (len(features) == 0):
            features = feature
            labels = np.full((feature.shape[0],1),class_label)
        else:
            features = np.concatenate((features,feature),axis=0)
            labels = np.append(labels,np.full((feature.shape[0],1),class_label))
    print('Feature shape: ' + str(features.shape))
    print('Label shape: ' + str(labels.shape))
    return [features,labels]

from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
def NB(train_or_val,localSave):
    [features,labels] = loadData(train_or_val,localSave)
    model = GaussianNB()
    model = model.fit(features,labels)
    return model

from sklearn import svm
def SVM(train_or_val,localSave):
    [features,labels] = loadData(train_or_val,localSave)
    clf = svm.LinearSVC()
    clf.fit(features, labels)
    return clf
    # load features
    # load outputs
    # for all of the features
    # compute the pairwise distance between the columns
    #  D(i,j) = sum (X(:,i) - Y(:,j)).^2
    # then normalize the histogram over the minimum distances
    # https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj4/html/hgarrison3/index.html

def do_main(train,train_data,val,val_data,localSave,modelType=0):
    # create a dataset
    print('Creating dataset for training...')
    createDataset(train,train_data)
    print('Created dataset for training...')
    print('Creating dataset for validation...')
    createDataset(val,val_data)
    print('Created dataset for validation...')
    # get features
    features = None
    if (not os.path.exists(localSave + '/features.txt')):
        print('Reading in features')
        features = readDatasets(train_data)
        np.savetxt(localSave + '/features.txt',features)
        print('Saving features locally...')
    else:
        print('Loading features')
        features = np.loadtxt(localSave + '/features.txt')
    print(features.shape)
    # generate vocab
    vocab = None
    if (not os.path.exists(localSave + '/vocab_' + str(VOCABLENGTH) + '.txt')):
        print('Generating a vocabularly of size: ' + str(VOCABLENGTH))
        batch = False
        if (features.shape[0] > 100000):
            batch = True
        vocab,kmeans = CodeBookGeneration(features,batch)
        print('Created vocabularly...')
        np.savetxt(localSave + '/vocab_' + str(VOCABLENGTH) + '.txt',vocab)
        print('saved vocab')
    else:
        print('Loading vocab')
        vocab = np.loadtxt(localSave + '/vocab_' + str(VOCABLENGTH) + '.txt')

    # create bag of sift for all images
    print("Bag of Sift: Training...")
    BagOfSIFT(train,vocab,'train',localSave)
    print("Bag of Sift: validation")
    BagOfSIFT(val,vocab,'val',localSave)

    # model
    model = None
    if (modelType == 0):
        print('Creating a Naive Bayes classifier...')
        model = NB('train',localSave)
    else:
        print('Creating an SVM classifier....')
        model = SVM('train',localSave)

    [test_features,test_labels] = loadData('val',localSave)
    out = model.predict(test_features)
    print("Accuracy:",metrics.accuracy_score(test_labels, out))

def showSIFT():
    import matplotlib.pyplot as plt
    sift = cv2.xfeatures2d.SIFT_create()
    plt.subplot(221)
    img = cv2.imread('./watch1.jpg')
    plt.imshow(img)
    # convert to gray scale
    img_gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    plt.subplot(222)
    plt.imshow(img_gray, cmap='gray')

    plt.subplot(223)
    img2 = cv2.imread('./watch2.jpg')
    plt.imshow(img2)
    # convert to gray scale
    img_gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    plt.subplot(224)
    plt.imshow(img_gray2, cmap='gray')
    # calculate keypoints and descriptors of keypoints using SIFT
    keypoints_sift, descriptors1 = sift.detectAndCompute(img_gray, None)
    keypoints_sift2, descriptors2 = sift.detectAndCompute(img_gray2, None)

    # create a BFMatcher object which will match up the SIFT features
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort the matches in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # draw the top N matches
    N_MATCHES = 100

    match_img = cv2.drawMatches(
        img, keypoints_sift,
        img2, keypoints_sift2,
        matches[:N_MATCHES], img2.copy(), flags=0)

    plt.figure(figsize=(12,6))
    plt.imshow(match_img)
    plt.show()
    # save results as a JSON
    img = cv2.imread('./watch1.jpg')
    img_gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    keypoints_sift, descriptors1 = sift.detectAndCompute(img_gray, None)
    cv2.drawKeypoints(img, keypoints_sift,img)
    plt.imshow(img)
    plt.show()
    print(descriptors1)
    print(descriptors1.shape)



VOCABLENGTH = 100 # hyperparameter
if __name__ == "__main__":
    base = "/media/burrussmp/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/"
    do_main(base+'heatmap_train',base+'heatmap_train_data',base+'heatmap_val',base+'heatmap_val_data','./heatmap',modelType=0)


    # showSIFT()
    # base = "/media/burrussmp/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/"
    # do_main(base+'train',base+'train_data',base+'val',base+'val_data','./regular',modelType=1)
