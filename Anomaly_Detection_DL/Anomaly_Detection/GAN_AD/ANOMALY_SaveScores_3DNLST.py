import matplotlib.pyplot as plt

# set gpu devices
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['KERAS_BACKEND'] = 'tensorflow'

# avoid full memory absorbtion
import tensorflow as tf
import keras.backend as K
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
#sess = tf.Session(config=config)
#K.set_session(sess)

# utils
import cv2
try:
    import cPickle as pickle
except ImportError:
    import pickle
import numpy as np


from collections import defaultdict

import pandas as pd
from sklearn import metrics
from scipy import stats
import math
import matplotlib.mlab as mlab

# keras imports
from keras.models import model_from_json
from keras.models import Model, Sequential
from keras import backend as K
from keras import backend as k


# sacred imports getting the dataset
from sacred import Experiment
from sacred.observers import FileStorageObserver
import lungcancer
from lungcancer.sacred_dataset import lc_dataset, data_generators
from lungcancer.metrics import fmeasure, precision, recall
import fuut.sacred
# experiment management:
fuut.sacred.check_required_version(lungcancer, '0.0.1')
fuut.sacred.check_required_version(fuut, '0.1.4')


class AnomalyDetection():
    def __init__(self, model_file='Models/wgan_generator3D_model_labelOne.json', weights_file='Weights/wgan-gp_generator_3D_labelOne.hdf5'):
        ##### parameters
        self.latent_dim = 100  # seed z property
        self.batch_size = 1  # seed z propertry
        self.idx_layer = 8  # layer f from discriminator f(x), do not use negative numbers
        self._lambda = 0.1  # resiudal or discrimination
        self.nb_descents = 100  # gradient steps
        self.step = 1  # constant to apply to gradients

        # load json and create model
        json_file = open(model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.generator_model = model_from_json(loaded_model_json)

        # load weights into new model
        self.generator_model.load_weights(weights_file)
        self.generator = self.generator_model.layers[1]  # extract generator
        self.discriminator = self.generator_model.layers[2]  # extract discriminator

        # f(x)
        layer_input = self.discriminator.layers[0].input
        layer_output = self.discriminator.layers[self.idx_layer].output
        self.get_feature_output = K.function([layer_input, K.learning_phase()],
                                             [layer_output])

        # G(z_{i}) and f(G(z_{i}))
        d_input = g_output = self.generator.layers[-1].output
        for i in range(self.idx_layer + 1):  # 0,1,...,idx_layer
            d_input = self.discriminator.layers[i](d_input)
        # model_gd_2outputs = Model(input=generator.input, output=[g_output, d_input])
        self.model_gd_2outputs = Model(input=self.generator.get_input_at(0), output=[g_output, d_input])

    def param(self, lambda_value, nb_descents, step):
        self._lambda = lambda_value
        self.nb_descents = nb_descents
        self.step = step

    def predict(self, x):
        '''
        Anomaly scores of an image and closest image found in the manifold space.

        Args:
            param1 (numpy): img with shape (1, w, h)

        Returns:
            (float) Value between [0,1]
            (numpy) img with shape (1,1,w,h)

            High score is a anomalous image, while a small score is
            considered normal. The img is the closest img found.
        '''
        # initialize seed and select one nodule
        z1 = np.random.rand(self.batch_size, self.latent_dim)
        #         z1 = np.random.uniform(-1, 1, (self.batch_size, self.latent_dim))
        x_expand = np.expand_dims(x, axis=0)

        # Overall loss
        f_x = self.get_feature_output([x_expand, 0])[0]  # [x, 0] is input, phase train '1' vs test '0'
        #         residual_loss = K.mean(k.abs(self.model_gd_2outputs.output[0][:, 0, :, :] - x[0]))


        #         print('Shape of x is {}'.format(x.shape))
        #         print('Shape of x[0] is {}'.format(x[0].shape))
        #         print('Shape of self.model_gd_2outputs.output[0][0,:,:,:] is {}'.format(self.model_gd_2outputs.output[0][0,:,:,:].shape))
        #         print('Shape of self.model_gd_2outputs.output[0] is {}'.format(self.model_gd_2outputs.output[0].shape))

        residual_loss = K.mean(k.abs(self.model_gd_2outputs.output[0][0, :, :, :] - x))

        #         print('Shape of self.model_gd_2outputs.output[1] is {}'.format(self.model_gd_2outputs.output[1].shape))
        #         print('Shape of f_x[0] is {}'.format(f_x[0].shape))
        #         print('Shape of f_x is {}'.format(f_x.shape))

        #         discriminator_loss =  K.mean(k.abs(self.model_gd_2outputs.output[1] - f_x[0]))
        discriminator_loss = K.mean(k.abs(self.model_gd_2outputs.output[1] - f_x))

        overall_loss = (1 - self._lambda) * (residual_loss) + self._lambda * (discriminator_loss)

        # Gradient
        input_noise = self.model_gd_2outputs.input
        grads = K.gradients(overall_loss, input_noise)[0]
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        iterate = K.function([input_noise, K.learning_phase()], [overall_loss, grads])

        # loop n gradient steps
        z = z1.copy()
        for i in range(self.nb_descents):
            loss_value, grads_value = iterate([z, 0])
            #             print('loss: {}'.format(loss_value))
            z -= grads_value * self.step
        G_z_n = self.generator.predict(z)

        return loss_value, G_z_n

    def predict_with_momentum(self, x, momentum=0.5):
        '''
        TODO: copy modification of predict before inserting momentum!!!

        Anomaly scores of an image and closest image found in the manifold space.
        Modification of predict() for testing momentum

        Args:
            param1 (numpy): img with shape (1, w, h)

        Returns:
            (float) Value between [0,1]
            (numpy) img with shape (1,1,w,h)

            High score is a anomalous image, while a small score is
            considered normal. The img is the closest img found.
        '''
        # initialize seed and select one nodule
        prev_delta = 0  # momentum need the previous delta increment
        momentum = 0.5
        z1 = np.random.rand(self.batch_size, self.latent_dim)
        #         z1 = np.random.uniform(-1, 1, (self.batch_size, self.latent_dim))
        x_expand = np.expand_dims(x, axis=0)

        # Overall loss
        f_x = self.get_feature_output([x_expand, 0])[0]  # [x, 0] is input, phase train '1' vs test '0'
        residual_loss = K.mean(k.abs(self.model_gd_2outputs.output[0][:, 0, :, :] - x[0]))
        discriminator_loss = K.mean(k.abs(self.model_gd_2outputs.output[1] - f_x[0]))
        overall_loss = (1 - self._lambda) * (residual_loss) + self._lambda * (discriminator_loss)

        # Gradient
        input_noise = self.model_gd_2outputs.input
        grads = K.gradients(overall_loss, input_noise)[0]
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        iterate = K.function([input_noise, K.learning_phase()], [overall_loss, grads])

        # loop n gradient steps
        #         z = z1.copy()
        #         for i in range(self.nb_descents):
        #             loss_value, grads_value = iterate([z, 0])
        # #             print('loss: {}'.format(loss_value))
        #             delta = grads_value * self.step
        #             z = z - delta - prev_delta # Z_{i+1} = z_{i} - learning rate*gradient - momentum*previous_delta
        #             prev_delta = momentum*delta
        #         G_z_n = self.generator.predict(z)

        z = z1.copy()
        for i in range(self.nb_descents):
            loss_value, grads_value = iterate([z, 0])  # loss and gradients
            delta = -self.step * grads_value - momentum * prev_delta  # self.step is learning rate
            z = z + delta  # update the input w.r.t. loss using SGD
            prev_delta = delta  # save previouse delta for momentum
        G_z_n = self.generator.predict(z)  # closest image

        return loss_value, G_z_n

    def generator_get_samples(self, batch_size):
        '''
        Return a batch of generated images using the generator
        '''
        z = np.random.rand(10, 100)
        return self.generator.predict(z)

    def scores(self, x_batch):
        '''
        Returns anomaly scores of a batch of images x.

        Args:
            param1 (numpy): img with shape (None, 1, w, h)

        Returns:
            (list): Values between [0, 1]

            High score is a anomalous image, while a small score is
            considered normal.
        '''
        return [self.predict(x)[0] for x in x_batch]

    def scores_with_momentum(self, x_batch, momentum=0.5):
        '''
        Returns anomaly scores of a batch of images x using momentum

        Args:
            param1 (numpy): img with shape (None, 1, w, h)

        Returns:
            (list): Values between [0, 1]

            High score is a anomalous image, while a small score is
            considered normal.
        '''
        return [self.predict_with_momentum(x, momentum)[0] for x in x_batch]

    def apply_threshold(self, arr, epsilon=0.5):
        '''
            Applies a threshold over each element of arr. Above the threshold
            we return a value of 1 and 0 otherwise.
        '''
        return [1 if i >= epsilon else 0 for i in arr]



ex = Experiment('LatentSpace', ingredients=[lc_dataset], interactive=True)


@lc_dataset.config
def dataset_config():
    volume_isotropic = None
    nb_patches = 1
    datasets = ['nlst']  # ['kaggle_stage1', 'nlst', 'lahey', 'mgh', 'toylung']
    volume_normalize_hu = True
    volume_normalize_cutoff = True
    return_split_generators = ['train', 'valid', 'test']
    patch_augmentations = [
        ('patch_random_crop', {'crop_dim': [28, 28, 28], 'center_test': False}),  # 28,28,28
    ]
    g_inputs = ['patches']
    patch_dim = (32, 32, 32)  # outer box
    g_outputs = [
        'nlst_cancer_after_screening_and_next_years_label']  # 'nlst_cancer_after_screening_and_next_years_label_one_label', 'key'
    batch_size = 25
    evaluation_batch_size = 25

    LC_DATASET_DIR = '/rst1/2015-0096_dl_data_2/lc_dataset/'
    DATA_STORAGE_DIR = '/rst1/2015-0096_dl_data_2/lc_dataset/'


@ex.main
def run(dataset):
    # initialize generators and size of data
    generators, nb_examples = data_generators()
    nb_train_batches = nb_examples['train'] // dataset['batch_size']
    nb_val_batches = nb_examples['valid'] // dataset['batch_size']
    nb_test_batches = nb_examples['test'] // dataset['batch_size']

    # info
    print('Size of validation batch {}'.format(nb_examples['valid']))
    print('Number of validation batches {}'.format(nb_val_batches))

    # Load Model
    model = AnomalyDetection(model_file='Models/wgan_generator3D_model_labelOne.json', weights_file='Weights/wgan-gp_generator_3D_labelOne.hdf5')
    model.param(1.0, 100, 0.05)  # lambda, nb_gradients, step

    # track results
    scores, y_true = [], []

    # Scores for batch
    for _ in range(nb_train_batches):
        #     for _ in range(3):
        print('Batch {} out of {}'.format(_, nb_train_batches))
        inputs, outputs = next(generators['train'])
        x_test_batch = np.squeeze(inputs['patches'])
        x_test_batch = np.squeeze(np.swapaxes(np.expand_dims(x_test_batch, axis=1), 1, 4))
        x_test_batch = np.expand_dims(x_test_batch, axis=1)

        batch_scores = model.scores(x_test_batch)
        batch_y_true = outputs['nlst_cancer_after_screening_and_next_years_label']

        for batch_score, label in zip(batch_scores, batch_y_true):
            scores.append(batch_score)
            y_true.append(label)

            #     print('Shape of x_valid_batch {}'.format(x_test_batch.shape))
            #     print('Keys of output {}'.format(outputs.keys()))
            #     print('Labels {}'.format(outputs['nlst_cancer_after_screening_and_next_years_label']))

            #     return x_test_batch, outputs['nlst_cancer_after_screening_and_next_years_label']
    return scores, y_true

r = ex.run()
scores, y_true = r.result

# save result arrays
np.save('scores_neg_train', scores)
np.save('y_true_neg_train', y_true)