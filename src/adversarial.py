import numpy as np
from model import VGG_16
from keras.preprocessing import image
from keras.activations import relu, softmax
import keras.backend as K
import matplotlib.pyplot as plt
import cv2
from keras.optimizers import SGD
from heatmap import generateHeatmap
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # first gpu
os.environ["CUDA_VISIBLE_DEVICES"]="1" # second gpu
os.environ["CUDA_VISIBLE_DEVICES"]="2" # second gpu
os.environ["CUDA_VISIBLE_DEVICES"]="3" # second gpu
os.environ["CUDA_VISIBLE_DEVICES"]="4" # second gpu

# def plot_img(x):
#     """
#     x is a BGR image with shape (? ,224, 224, 3)
#     """
#     t = np.zeros_like(x[0])
#     t[:,:,0] = x[0][:,:,2]
#     t[:,:,1] = x[0][:,:,1]
#     t[:,:,2] = x[0][:,:,0]
#     plt.figure()
#     #cv2.imwrite('adversarial.png',np.clip((t+[123.68, 116.779, 103.939]), 0, 255)/255)
#     plt.imshow(np.clip((t+[123.68, 116.779, 103.939]), 0, 255)/255)
#     plt.grid('off')
#     plt.axis('off')

# model = VGG_16(weights_path='./models/vgg16_5.h5',classes=2)
# model.compile(loss = "categorical_crossentropy", optimizer = SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

# model2 = VGG_16(weights_path='../Pattern_Recognition_Final_Assignment/models/heatmap1.h5',classes=2)
# model2.compile(loss = "categorical_crossentropy", optimizer = SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

# model3 = VGG_16(weights_path='./models/vgg16_4.h5',classes=2)
# model3.compile(loss = "categorical_crossentropy", optimizer = SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

# dirpath = "/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012"
# img_path = dirpath + '/val_used/n01498041' + '/n01498041_7880.JPEG'

# plt.grid('off')
# plt.axis('off')
# def preprocess(img):
#     img = cv2.resize(img,(224, 224)).astype(np.float32)
#     img = np.expand_dims(img, axis=0)
#     img = img / 255.
#     return img

# def preprocess2(img):
#     img = cv2.resize(img,(224, 224)).astype(np.float32)
#     img = np.expand_dims(img, axis=0)
#     return img
# # Create a batch and preprocess the image
# img = cv2.imread(img_path)

# x = preprocess(img)
# # Get the initial predictions

# # Get current session (assuming tf backend)
# sess = K.get_session()
# # Initialize adversarial example with input image
# x_adv = x
# # Added noise
# x_noise = np.zeros_like(x)

# # Set variables
# epochs = 10
# epsilon = 0.001
# target_class = 0
# prev_probs = []
# classes = 2
# for i in range(epochs):
#     # One hot encode the target class
#     target = K.one_hot(target_class, classes)

#     # Get the loss and gradient of the loss wrt the inputs
#     loss = -1*K.categorical_crossentropy(target, model.output)
#     grads = K.gradients(loss, model.input)
#     # Get the sign of the gradient
#     delta = K.sign(grads[0])
#     x_noise = x_noise + delta
#     print(x_noise)
#     # Perturb the image
#     x_adv = x_adv + epsilon*delta

#     # Get the new image and predictions
#     x_adv = sess.run(x_adv, feed_dict={model.input:x})

#     preds = model.predict(x_adv)
#     print(i)
#     print(preds)

# heat = generateHeatmap(model,x_adv)
# heat = cv2.cvtColor(heat,cv2.COLOR_GRAY2RGB)*255
# heat2 = generateHeatmap(model,x)
# heat2 = cv2.cvtColor(heat2,cv2.COLOR_GRAY2RGB)*255
# cv2.imwrite('adversarial.png',heat)
# # plt.imshow(heat)
# # plt.show()
# print('Original calculations')
# print('Model A: ', end="")
# print(model.predict(x))
# print('Model B: ', end="")
# print(model3.predict(x))
# print('Adversarial calculations')
# print('Model A: ', end="")
# print(model.predict(x_adv))
# print('Model B: ', end="")
# print(model3.predict(x_adv))
# print('Heatmap calculations')
# print('Model C: ', end="")
# print(model2.predict(preprocess(heat)))
# print(model2.predict(preprocess(heat2)))


# # print(preds)
# # print('a')
# # preds = model2.predict(x_adv)
# # print(preds)
# # preds = model2.predict(x)
# # print(preds)
# # preds = model.predict(x)
# # print(preds)
# plot_img(x_adv*255)

# #plot_img(x*255)
# plt.show()
# #plot_img(x_adv*255-x*255)