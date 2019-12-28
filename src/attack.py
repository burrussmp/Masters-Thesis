"""
The script demonstrates a simple example of using ART with Keras. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import keras
import keras.backend as K
import numpy as np
import cv2
from art.attacks import FastGradientMethod,DeepFool,CarliniL2Method,BasicIterativeMethod,ProjectedGradientDescent
from art.utils import load_mnist
from keras.preprocessing.image import ImageDataGenerator
from ModifiedKerasClassifier import KerasClassifier
from generate_model import generate_model
from art.metrics import metrics
from art.utils import to_categorical
import argparse
from sklearn.metrics import mean_squared_error
import numpy.linalg as la
from art.classifiers import KerasClassifier as DefaultKerasClassifier
from heatmap import LRP
# # Step 8: Compute the robustness

# robust = metrics.empirical_robustness(classifier,x_test_adv,'fgsm') # computes the standard euclidean distance or norm 2
# print('Robustness on test examples: {}%'.format(robust))


# plot the loss gradient

# loss_gradient = classifier.loss_gradient(x=x_art, y=to_categorical([label], nb_classes=1000))

# # Let's plot the loss gradient.
# # First, swap color channels back to RGB order:
# loss_gradient_plot = loss_gradient[0][..., ::-1]

# # Then normalize loss gradient values to be in [0,1]:
# loss_gradient_min = np.min(loss_gradient)
# loss_gradient_max = np.max(loss_gradient)
# loss_gradient_plot = (loss_gradient_plot - loss_gradient_min)/(loss_gradient_max - loss_gradient_min)

# # Show plot:
# plt.figure(figsize=(8,8)); plt.imshow(loss_gradient_plot); plt.axis('off'); plt.show()

# class myDefense():
#     def __init__(self,lrp,model_generator):
#         self.lrp = lrp
#         self.model_generator = model_generator
#
#     def __call__(self,x,y=None):
#         X = np.copy(x)
#         for i in range(X.shape[0]):
#             X[i] = self.lrp.generate_heatmap(X[i])
#             X[i] = np.stack((X[i],)*3, axis=-1)
#             X[i] = self.model_generator.preprocess(X[i])
#         return X, y
#     @property
#     def apply_fit(self):
#         return False
#
#     @property
#     def apply_predict(self):
#         return False
#     def estimate_gradient(self, x, grad):
#         return grad
#     def fit(self, x, y=None, **kwargs):
#         """
#         No parameters to learn for this method; do nothing.
#         """
#         pass

#
# def test(args):
#     modelType = args.model
#     weights = '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/weights/' + args.model + '/regular_endtoend_weights_' + args.model.lower() + '.h5'
#     #weights = '../weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
#     heatweights = weights.replace('regular','heat')
#     if args.model == 'DaveII':
#         numClasses = 10
#     else:
#         numClasses = 2
#     #numClasses = 1000
#     path_to_data = '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/' + args.model.lower() + '_dataset_partitioned/test'
#     batch_size = 64
#     #path_to_data = '../ignoreme/dumdata'
#     # Step 1. Load the model
#     model_generator = generate_model(modelType,False)
#     basemodel = model_generator.initialize(weights,classes=numClasses,useTransfer=False,optimizer='Adam',reset=False)
#     heatmodel = model_generator.initialize(heatweights,classes=numClasses,useTransfer=False,optimizer='Adam',reset=False)
#     classifier = KerasClassifier(model=basemodel, model_generator=model_generator,heatmodel=heatmodel,use_logits=False,clip_values=(0, 255))
#
#     lrp = LRP(basemodel,model_generator.preprocess,modelType=model_generator.getName())
#     defense = myDefense(lrp,model_generator)
#
#     classifier_def = DefaultKerasClassifier(defences=[defense],model=heatmodel,use_logits=False,clip_values=(0, 255))
#     adv_def = ProjectedGradientDescent(classifier_def, targeted=True, max_iter=40, eps_step=1, eps=5)
#
#     batchX = np.load('../adversary/batchX_' + modelType + '.npy')
#     batchy = np.load('../adversary/batchy_' + modelType + '.npy')
#     print('Generating adversarial images')
#     labels = np.zeros((batchy.shape[0]))
#     labels.fill(int(0))
#     labels = to_categorical(labels,nb_classes=numClasses)
#     x_art_adv_def = adv_def.generate(batchX, y=labels)
#     x_art_adv_def = model_generator.unprocess(x_art_adv_def)
#     print('Evaluating the images')
#     predictions2 = classifier.predict_new(batchX)
#     predictions = classifier.predict_new(x_art_adv_def)
#     if modelType == 'DaveII':
#         mse2 = mean_squared_error(np.argmax(predictions2, axis=1), np.argmax(batchy, axis=1))
#         mse = mean_squared_error(np.argmax(predictions, axis=1), np.argmax(batchy, axis=1))
#         print('Accuracy on adversary images (defense included): {}'.format(mse))
#         print('Accuracy on benign images (defense included): {}'.format(mse2))
#     else:
#         accuracy2 = np.sum(np.argmax(predictions2, axis=1) == np.argmax(batchy, axis=1)) / len(batchy)
#         accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(batchy, axis=1)) / len(batchy)
#         print('Accuracy on adversary images (defense included): {}%'.format(accuracy * 100))
#         print('Accuracy on benign images (defense included): {}%'.format(accuracy2 * 100))
#     return


def doMetrics(args):
    assert args.metrics in ['Empirical','Clever'], \
        'Metric not implemented!'
    modelType = args.model
    weights = '/media/burrussmp/99e21975-0750-47a1-a665-b2522e4753a6/weights/' + args.model + '/regular_endtoend_weights_' + args.model.lower() + '.h5'
    #weights = '../weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    heatweights = weights.replace('regular','heat')
    if args.model == 'DaveII':
        numClasses = 10
    else:
        numClasses = 2
    #numClasses = 1000
    path_to_data = '/media/burrussmp/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/' + args.model.lower() + '_dataset_partitioned/test'
    batch_size = 15
    #path_to_data = '../ignoreme/dumdata'
    # Step 1. Load the model
    model_generator = generate_model(modelType,False)
    basemodel = model_generator.initialize(weights,classes=numClasses,useTransfer=False,optimizer='Adam',reset=False)
    heatmodel = model_generator.initialize(heatweights,classes=numClasses,useTransfer=False,optimizer='Adam',reset=False)
    classifier = KerasClassifier(model=basemodel, model_generator=model_generator,heatmodel=heatmodel,use_logits=False,clip_values=(0, 255))
    if (args.metrics == 'Empirical'):
        print('Calculating empirical robustness')
        datagen = ImageDataGenerator()
        test_data = datagen.flow_from_directory(
            path_to_data,
            target_size =model_generator.getInputSize(),
            batch_size = batch_size,
            class_mode = "categorical",
            shuffle=True)
        classifier = DefaultKerasClassifier(model=basemodel,use_logits=False,clip_values=(0, 255))
        # get a batch of data
        batchX, batchy = test_data.next()
        attack = FastGradientMethod(classifier=classifier,eps_step=0.1,eps=8)
        attack.set_params(**{'minimal': True})
        print('Setting up targeted attack')
        attack.set_params(targeted=True)
        labels = np.zeros((batchy.shape[0]))
        labels.fill(np.random.randint(0,numClasses,1)[0])
        labels = to_categorical(labels,nb_classes=numClasses)
        print('Devising adversarial images')
        x_test_adv = attack.generate(x=model_generator.preprocess(batchX),y=labels)
        y_pred = classifier.predict(x_test_adv)
        y = batchy
        if modelType =='DaveII':
            idxs = (abs(np.argmax(y_pred, axis=1) - np.argmax(y, axis=1)) > 2)
        else:
            idxs = (np.argmax(y_pred, axis=1) != np.argmax(y, axis=1))
        if np.sum(idxs) == 0.0:
            return 0
        norm_type = 2
        perts_norm = la.norm((x_test_adv - batchX).reshape(batchX.shape[0], -1), ord=norm_type, axis=1)
        perts_norm = perts_norm[idxs]
        robust= np.mean(perts_norm / la.norm(batchX[idxs].reshape(np.sum(idxs), -1), ord=norm_type, axis=1))
        print('Emprical robustness: {}%'.format(robust))

    elif (args.metrics == 'Clever'):
        x_test_adv = np.load('../adversary/adv_' + modelType + '.npy')
        batchX = np.load('../adversary/batchX_' + modelType + '.npy')
        batchy = np.load('../adversary/batchy_' + modelType + '.npy')
        print('Computing CLEVER score')
        score = metrics.clever(classifier,x=batchX[5],nb_batches=10,batch_size=16,radius=0.3,norm=2)
        print('Clever score: {}%'.format(score))


def main(args):
    if (args.metrics):
        doMetrics(args)
        return
    modelType = args.model
    weights = '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/weights/' + args.model + '/regular_endtoend_weights_' + args.model.lower() + '.h5'
    #weights = '../weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    heatweights = weights.replace('regular','heat')
    if args.model == 'DaveII':
        numClasses = 10
    else:
        numClasses = 2
    #numClasses = 1000
    path_to_data = '/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/' + args.model.lower() + '_dataset_partitioned/test'
    batch_size = 64
    #path_to_data = '../ignoreme/dumdata'
    # Step 1. Load the model
    model_generator = generate_model(modelType,False)
    basemodel = model_generator.initialize(weights,classes=numClasses,useTransfer=False,optimizer='Adam',reset=False)
    heatmodel = model_generator.initialize(heatweights,classes=numClasses,useTransfer=False,optimizer='Adam',reset=False)

    # step 3 load the data
    datagen = ImageDataGenerator()
    test_data = datagen.flow_from_directory(
        path_to_data,
        target_size =model_generator.getInputSize(),
        batch_size = batch_size,
        class_mode = "categorical",
        shuffle=True)
    classifier = KerasClassifier(model=basemodel, model_generator=model_generator,heatmodel=heatmodel,use_logits=False,clip_values=(0, 255))

    # get a batch of data
    batchX, batchy = test_data.next()

    # init classifier
    if (not args.load):
        print('Creating the adversarial attack')
        if args.attack == 'FGSM':
            attack = FastGradientMethod(classifier=classifier, eps=0.05,eps_step=0.01)
        elif args.attack == 'ProjectedGradientAscent':
            attack = ProjectedGradientDescent(classifier, targeted=False, max_iter=10, eps_step=.01, eps=.05)
        else:
            raise ValueError('Attack not implemented')
        labels = None
        if (args.target):
            print('Setting up targeted attack')
            attack.set_params(targeted=True)
            labels = np.zeros((batchy.shape[0]))
            labels.fill(int(args.target))
            labels = to_categorical(labels,nb_classes=numClasses)
        x_test_adv = attack.generate(x=model_generator.preprocess(batchX),y=labels)
        np.save('../adversary/adv_' + modelType + '.npy',x_test_adv)
        np.save('../adversary/batchX_' + modelType + '.npy',batchX)
        np.save('../adversary/batchy_' + modelType + '.npy',batchy)
    else:
        print('Loading')
        x_test_adv = np.load('../adversary/adv_' + modelType + '.npy')
        batchX = np.load('../adversary/batchX_' + modelType + '.npy')
        batchy = np.load('../adversary/batchy_' + modelType + '.npy')

    x_test_adv = model_generator.unprocess(x_test_adv)

    predictions2 = classifier.predict(batchX)
    predictions = classifier.predict(x_test_adv)

    # print(np.argmax(predictions2, axis=1))
    # print(np.argmax(batchy, axis=1))
    if modelType == 'DaveII':
        mse2 = mean_squared_error(np.argmax(predictions2, axis=1), np.argmax(batchy, axis=1))
        mse = mean_squared_error(np.argmax(predictions, axis=1), np.argmax(batchy, axis=1))
        print('Accuracy on adversary images (defense included): {}'.format(mse))
        print('Accuracy on benign images (defense included): {}'.format(mse2))
    else:
        accuracy2 = np.sum(np.argmax(predictions2, axis=1) == np.argmax(batchy, axis=1)) / len(batchy)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(batchy, axis=1)) / len(batchy)
        print('Accuracy on adversary images (defense included): {}%'.format(accuracy * 100))
        print('Accuracy on benign images (defense included): {}%'.format(accuracy2 * 100))

    predictions2 = basemodel.predict(model_generator.preprocess(batchX))
    predictions = basemodel.predict(model_generator.preprocess(x_test_adv))
    #print(np.argmax(predictions2, axis=1))
    if modelType == 'DaveII':
        mse2 = mean_squared_error(np.argmax(predictions2, axis=1), np.argmax(batchy, axis=1))
        mse = mean_squared_error(np.argmax(predictions, axis=1), np.argmax(batchy, axis=1))
        print('Accuracy on adversary images (defense included): {}'.format(mse))
        print('Accuracy on benign images (defense included): {}'.format(mse2))
    else:
        accuracy2 = np.sum(np.argmax(predictions2, axis=1) == np.argmax(batchy, axis=1)) / len(batchy)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(batchy, axis=1)) / len(batchy)
        print('Accuracy on adversary images (defense not included): {}%'.format(accuracy * 100))
        print('Accuracy on benign images (defense not included): {}%'.format(accuracy2 * 100))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a neural net!')


    parser.add_argument('--model', '-M', metavar='M', type=str,required=True,
                    help='Model to load to evaluate on the data set',dest='model')

    parser.add_argument('--attack', '-A', metavar='A', type=str,required=False,
                    help='attack to test.',dest='attack')
    parser.add_argument('--target', '-T', metavar='T', type=str,required=False,
                    help='Class to target',dest='target')
    parser.add_argument('--load', '-L',action='store_true',
                    help='Load instead of of producing adversarial image',dest='load')
    parser.add_argument('--metrics', type=str,required='False',
                    help='Perform metrics (empirical robustness)',dest='metrics')
    args = parser.parse_args()
    main(args)
