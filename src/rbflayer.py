from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import RandomUniform, Initializer, Constant
import numpy as np

from keras.initializers import Initializer
from sklearn.cluster import KMeans
import tensorflow as tf
import keras
class InitCentersKMeans(Initializer):
    """ Initializer for initialization of centers of RBF network
        by clustering the given data set.
    # Arguments
        X: matrix, dataset
    """

    def __init__(self, X, max_iter=100):
        self.X = X
        self.max_iter = max_iter

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]

        n_centers = shape[0]
        km = KMeans(n_clusters=n_centers, max_iter=self.max_iter, verbose=1)
        km.fit(self.X)
        return km.cluster_centers_


class InitCentersRandom(Initializer):
    """ Initializer for initialization of centers of RBF network
        as random samples from the given data set.

    # Arguments
        X: matrix, dataset to choose the centers from (random rows
          are taken as centers)
    """

    def __init__(self, X):
        self.X = X

    def __call__(self, shape, dtype=None):
        print(self.X.shape)
        print(shape)
        assert shape[1] == self.X.shape[1]
        idx = np.random.randint(self.X.shape[0], size=shape[0])
        return self.X[idx, :]


# class RBFLayer(Layer):

#     def __init__(self, output_dim, initializer=None, betas=1.0, **kwargs):
#         self.output_dim = output_dim
#         self.init_betas = betas
#         if not initializer:
#             self.initializer = RandomUniform(0.0, 1.0)
#         else:
#             self.initializer = initializer
#         super(RBFLayer, self).__init__(**kwargs)

#     def build(self, input_shape):

#         self.centers = self.add_weight(name='centers',
#                                        shape=(self.output_dim, input_shape[1]),
#                                        initializer=self.initializer,
#                                        trainable=True)
#         self.betas = self.add_weight(name='betas',
#                                      shape=(self.output_dim,),
#                                      initializer=Constant(
#                                          value=self.init_betas),
#                                      # initializer='ones',
#                                      trainable=True)

#         super(RBFLayer, self).build(input_shape)

#     def call(self, x):

#         C = K.expand_dims(self.centers)
#         H = K.transpose(C-K.transpose(x))
#         return K.exp(-1*self.betas * K.sum(H**2, axis=1))

#         # C = self.centers[np.newaxis, :, :]
#         # X = x[:, np.newaxis, :]

#         # diffnorm = K.sum((C-X)**2, axis=-1)
#         # ret = K.exp( - self.betas * diffnorm)
#         # return ret

#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.output_dim)

#     def get_config(self):
#         # have to define get_config to be able to use model_from_json
#         config = {
#             'output_dim': self.output_dim
#         }
#         base_config = super(RBFLayer, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
#         print(input_shape)
#         print(self.units)
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer=keras.initializers.RandomUniform(minval=-1, maxval=1, seed=1234),
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return l2

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'units': self.units,
            'gamma': self.gamma
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
