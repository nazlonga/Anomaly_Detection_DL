import matplotlib.pyplot as plt

# set gpu devices
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['KERAS_BACKEND'] = 'tensorflow'
#os.chdir('/rst1/2015-0096_dl_data_2/org-2015-0096_dl_data/users/NazlySantos')

'''Training without validation dataset
use of data in npy array'''
try:
    import cPickle as pickle
except ImportError:
    import pickle

import tensorflow as tf
import keras.backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.utils.generic_utils import Progbar
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer


from keras import backend as K
from keras import metrics
from keras.losses import mse, binary_crossentropy, mae
from keras.datasets import mnist
# keras imports
from keras.models import model_from_json
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, K
from keras.layers.merge import _Merge
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D

from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from keras import backend as K
from keras.utils.generic_utils import Progbar
from functools import partial
K.set_image_data_format('channels_first')

from sacred import Experiment
import lungcancer
from lungcancer.sacred_dataset import lc_dataset, data_generators
from lungcancer.metrics import fmeasure, precision, recall


# input image dimensions
img_rows, img_cols, img_chns = 28, 28, 28
# number of convolutional filters to use
filters = 64
# convolution kernel size
num_conv = 3

batch_size = 25
if K.image_data_format() == 'channels_first':
    original_img_size = (1, img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns, 1)
latent_dim = 10
intermediate_dim = 128
epsilon_std = 1.0
epochs = 100


x = Input(batch_shape=(None,) + original_img_size, name='input')
# x = Input(batch_shape=(None,) + original_img_size)
conv_1 = Conv3D(img_chns,
                kernel_size=(2, 2, 2),
                padding='same', activation='relu', name='conv1')(x)
conv_2 = Conv3D(filters,
                kernel_size=(2, 2, 2),
                padding='same', activation='relu',
                strides=(2, 2, 2), name='conv2')(conv_1)
conv_3 = Conv3D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1, name='conv3')(conv_2)
conv_4 = Conv3D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1, name='conv4')(conv_3)
flat = Flatten(name='flatten1')(conv_4) # change to conv to reduce the parameters nb
hidden = Dense(intermediate_dim, activation='relu', name='hiden1')(flat)

z_mean = Dense(latent_dim, name='z_mean')(hidden)
z_log_var = Dense(latent_dim, name='z_log_var')(hidden)



def sampling(args):
    #batch_size = 25
    #latent_dim = 10
    #epsilon_std = 1.0 cant save the model using variables inside this function
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(25, 10),
                              mean=0., stddev=1.0)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon #multiply 0.5 !!!! reparameterization trick

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_var])`
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

#Encoder
encoder = Model(x, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# we instantiate these layers separately so as to reuse them later
decoder_hid = Dense(intermediate_dim, activation='relu', name='decoder1')
decoder_upsample = Dense(filters * 14 * 14 * 14, activation='relu', name='decoder2')

if K.image_data_format() == 'channels_first':
    output_shape = (None, filters, 14, 14, 14)
else:
    output_shape = (None, 14, 14, 14, filters)

decoder_reshape = Reshape(output_shape[1:], name='decoderReshape1')
decoder_deconv_1 = UpSampling3D(size=(2, 2, 2), name='decoderUp1')
decoder_deconv_2 = UpSampling3D(size=(1, 1, 1), name='decoderUp2')
if K.image_data_format() == 'channels_first':
    output_shape = (None, filters, 29, 29, 29)
else:
    output_shape = (None, 29, 29, 29, filters)
decoder_deconv_3_upsamp = UpSampling3D(size=(1, 1, 1), name='decoderUp3')
decoder_mean_squash = Conv3D(1,
                             kernel_size=2,
                             padding='same',
                             activation='sigmoid', name='decoderConv1')

hid_decoded = decoder_hid(z)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded = decoder_reshape(up_decoded)
deconv_1_decoded = decoder_deconv_1(reshape_decoded)
deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)


# definition of the losses as functions to be added in the compile - metrics of Keras
# definition of the lossekl_losss as functions to be added in the compile - metrics of Keras
def kl_loss_f(inputs,outputs):
        # D_KL(Q(z|X) || P(z|X))
    #z_mean, z_log_var = args
    kl_loss_f = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return kl_loss_f


def reconstruction_error_f(inputs, outputs):
    # E[log P(X|z)]
    inputs = K.flatten(inputs)
    outputs = K.flatten(outputs)
    reconstruction_loss_f =  img_rows * img_rows * mse(inputs,outputs)
    #image_size * image_size * mse(inputs,outputs)
    return reconstruction_loss_f


# Loss to be optimized in the training

def total_loss(inputs, outputs):
    inputs = K.flatten(inputs)
    outputs = K.flatten(outputs)
    # log likelihood
    reconstruction_loss =  img_rows * img_rows * mse(inputs,outputs)
    #kl_loss_ = kl_loss(inputs,outputs)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    loss = K.mean(reconstruction_loss + kl_loss) #why is the mean?
    return loss

vae = Model(x, x_decoded_mean_squash)
vae.compile(loss=total_loss, optimizer='rmsprop', metrics=[mse, kl_loss_f,reconstruction_error_f])
vae.summary()


## getting the data
#### LOAD DATA
x_train_batch = np.load('../Data_handling/x_train_negativeOnly.npy')
#nb_train_batches = int(x_train_batch.shape[0]/batch_size)
print('shape', x_train_batch.shape)
#print('train steps', nb_train_batches)

#history = vae.fit(x=x_train_batch, epochs=epochs, batch_size=batch_size, validation_data=None)

# we are just training
#test_score = vae.evaluate_generator(generator_test(),steps=nb_test_batches)

#
# print('history', history.history)
#
# # save results of the training
#
# pickle.dump({'train': history.history}, open('VAE_3D_MSE_noGen.pkl', 'wb'))
#
# vae.save_weights('VAE_3D_MSE_noGen.hdf5')
# model_json = vae.to_json()
# with open("VAE_3D_MSE_noGen.json", "w") as json_file:
#     json_file.write(model_json)


ex = Experiment('VAE3D', ingredients=[lc_dataset], interactive=True)
@lc_dataset.config
def dataset_config():
#     volume_isotropic = None
    nb_patches = 1
    volume_normalize_hu = True
    volume_normalize_cutoff = True
    batch_edge_handling = 'shift'
    datasets = ['nlst']  # ['kaggle_stage1', 'nlst', 'lahey', 'mgh', 'toylung']
    return_split_generators = ['train', 'valid', 'test']
#     patch_augmentations = [ # without this one if conv3d
#         ('patch_random_crop', {'crop_dim': [5,28,28], 'center_test': True}),
#     ]
    g_inputs = ['patches']
    patch_dim = (28, 28, 28) # (64, 64, 64) 3D conv
    g_outputs = ['nlst_cancer_after_screening_and_next_years_label_one_label', 'key']
#     g_outputs = ['volume_1_year_cancer_label','key']
    batch_size = 25
    evaluation_batch_size = 25

    LC_DATASET_DIR = '/rst1/2015-0096_dl_data_2/lc_dataset/'
    DATA_STORAGE_DIR = '/rst1/2015-0096_dl_data_2/lc_dataset/'


@ex.main
def run(dataset):
    generators, nb_examples = data_generators()
    nb_train_batches = nb_examples['train'] // dataset['batch_size']
    nb_val_batches = nb_examples['valid'] // dataset['batch_size']
    nb_test_batches = nb_examples['test'] // dataset['batch_size']

    def generator_train():
        while 1:
            inputs, outputs = next(generators['train'])
            x_train_batch = np.squeeze(inputs['patches'])
            x_train_batch = np.expand_dims(x_train_batch, axis=1)
            yield x_train_batch, x_train_batch

    def generator_val():
        while 1:
            inputs, outputs = next(generators['valid'])
            x_val_batch = np.squeeze(inputs['patches'])
            x_val_batch = np.expand_dims(x_val_batch, axis=1)
            yield x_val_batch, x_val_batch

    def generator_test():
        while 1:
            inputs, outputs = next(generators['test'])
            x_test_batch = np.squeeze(inputs['patches'])
            x_test_batch = np.expand_dims(x_test_batch, axis=1)
            yield x_test_batch, x_test_batch

    history = vae.fit_generator(
        generator=generator_train(),
        steps_per_epoch=nb_train_batches,
        epochs=epochs
    )

    #test_score = vae.evaluate_generator(generator_test(),
                                        #steps=nb_test_batches)
    #print('Test score: ', test_score)
    return history


r = ex.run()

h = r.result
#h.history.keys()

print(h.history)

pickle.dump({'train': h.history}, open('VAE_3D_finalLoss_history.pkl', 'wb'))


vae.save_weights('VAE_3D_finalLoss_weights.hdf5')
model_json = vae.to_json()
with open("VAE_3D_finalLoss__model.json", "w") as json_file:
    json_file.write(model_json)
