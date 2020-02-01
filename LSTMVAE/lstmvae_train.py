'''Example of VAE on MNIST dataset using MLP

The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean = 0 and std = 1.

# Reference

[1] Kingma, Diederik P., and Max Welling.
"Auto-Encoding Variational Bayes."
https://arxiv.org/abs/1312.6114
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from keras.layers import Lambda, Input, Dense, LSTM, RepeatVector, Conv2D
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K, objectives
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

import pretty_midi
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping
import vae_config


def vae_loss( x, x_decoded_mean):
    reconstruction_loss = objectives.mse(x, x_decoded_mean)
    reconstruction_loss *= vae_config.ORIGINAL_DIM
    kl_loss = 1 + z_log_sigma - \
              K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    return vae_loss


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=128):
    """Plots labels and MNIST digits as a function of the 2D latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data

    # display a 2D plot of the digit classes in the latent space
    z_mean = encoder.predict(x_test,batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    #print(len(z_mean))
    nb_elem_per_class = vae_config.dataset_size*vae_config.test_size

    #to_decode = np.array([[0.5, 0], [1.8, 1]], dtype=np.float32)
    #final = decoder.predict(to_decode)
    #print(final )

   # print("ICI ", file_shuffle[:int(dataset_size * test_size)])
    #for i, txt in enumerate(file_shuffle[ :int(dataset_size*2 * test_size)]):
        #print("i ", i)
        #plt.annotate(txt,(z_mean[i,0], z_mean[i,1]))

    plt.show()


def load_data(path, class_label, index_filename ):

    path, dirs, files = next(os.walk(path))
    num_size = len(dirs)
    current_folder = 0
    num_files = 0


    size = 465 * int(vae_config.res / 8)
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if num_files < vae_config.dataset_size:
                if file != ".DS_Store":
                    # print(os.path.join(subdir, file))
                    # try:
                    midi_data = pretty_midi.PrettyMIDI(subdir + "/" + file)
                    for instrument in midi_data.instruments:
                        instrument.is_drum = False
                    if len(midi_data.instruments) > 0:
                        data = midi_data.get_piano_roll(fs=vae_config.res)[35:50, :]

                        flattened = data.flatten()
                        flattened = flattened.astype(dtype=bool)
                        # data.resize(data.size, refcheck=False)
                        # np.resize(data, (1,465)
                        final_array = []
                        if data.size <= size:
                            remaining_zero = size - data.size
                            final_array = np.pad(flattened, (0, remaining_zero), mode='constant', constant_values=0)
                            if np.count_nonzero(final_array) == 0:
                                print("0 IN DATASET")
                            features.append([final_array, class_label])

                        else:
                            remaining_zero = data.size - size
                            final_array = flattened[0:flattened.size - remaining_zero]
                            if np.count_nonzero(final_array) == 0:
                                print("0 IN DATASET")
                            features.append([final_array, class_label])
                        #print(file)
                        vae_config.list_files_name.insert(index_filename+num_files, file)
                        num_files += 1
                    # except:
                    #    print("An exception occurred")
        current_folder += 1
        print("Done ", num_files, " from ", current_folder, " folders on ", num_size)
        return True


print("LOADING DATA FOR TRAINING...")
features = []

path_to_load = "/home/kyrillos/CODE/VAEMIDI/16_bars/0"
load_data(path_to_load, 0,   0)
path_to_load = "/home/kyrillos/CODE/VAEMIDI/16_bars/100"
load_data(path_to_load,1,  vae_config.dataset_size)

# Convert into a Panda dataframe
featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

print('Finished feature extraction from ', len(featuresdf), ' files')

# Convert features & labels into numpy arrays
listed_feature = featuresdf.feature.tolist()

X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

print(X.shape, y.shape)
# split the dataset


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=vae_config.test_size, random_state=vae_config.random_state)
X_shuffle = shuffle(X, random_state=vae_config.random_state)
y_shuffle = shuffle(y, random_state=vae_config.random_state)
file_shuffle = shuffle(vae_config.list_files_name, random_state=vae_config.random_state)


print("X_test: ", x_test)
to_compare  = X_shuffle[:int(vae_config.dataset_size*2 * vae_config.test_size)]
print("x_shuffle ",to_compare )
print("COMPARE: ", x_test == to_compare)


print(file_shuffle)
midi_file_size = x_train.shape[1]
original_dim = midi_file_size
#x_train = np.reshape(x_train, [-1, original_dim])
#x_test = np.reshape(x_test, [-1, original_dim])
#x_train = x_train.astype('float64') / 100
#x_test = x_test.astype('float64') / 100

# network parameters
input_shape = (original_dim, )

dim1 = vae_config.INTER_DIM1
dim2 = vae_config.INTER_DIM2
dim3 = vae_config.INTER_DIM3
dim4 = vae_config.INTER_DIM4
dim5 = vae_config.INTER_DIM5
origdim = vae_config.ORIGINAL_DIM
latent_dim = vae_config.LATENT_DIM

decoder_i1 = Dense(dim1, activation='relu')  # layer size 768
decoder_i2 = Dense(dim2, activation='relu')  # layer size 192
decoder_i3 = Dense(dim3, activation='relu')  # layer size 48
decoder_i4 = Dense(dim4, activation='relu')  # layer size 48
decoder_i5 = Dense(dim5, activation='relu')  # layer size 48
decoder_mean = Dense(origdim)

x = Input(shape=(vae_config.ORIGINAL_DIM,))
i1 = Dense(dim1, activation='relu')(x)
i2 = Dense(dim2, activation='relu')(i1)
i3 = Dense(dim3, activation='relu')(i2)
i4 = Dense(dim4, activation='relu')(i3)
i5 = Dense(dim5, activation='relu')(i4)

z_mean = Dense(latent_dim)(i5)
z_log_sigma = Dense(latent_dim)(i5)

# sample new similar points from the latent space
z = Lambda(sampling, output_shape=(latent_dim,),
           name='z')([z_mean, z_log_sigma])

i5_decoded = decoder_i5(z)
i4_decoded = decoder_i4(i5_decoded)
i3_decoded = decoder_i3(i4_decoded)
i2_decoded = decoder_i2(i3_decoded)
i1_decoded = decoder_i1(i2_decoded)
x_decoded_mean = decoder_mean(i1_decoded)

# Encoder part of the model
encoder = Model(x, z_mean)

# Decoder/Generator part of the model
decoder_input = Input(shape=(latent_dim,))
_i5_decoded = decoder_i5(decoder_input)
_i4_decoded = decoder_i4(_i5_decoded)
_i3_decoded = decoder_i3(_i4_decoded)
_i2_decoded = decoder_i2(_i3_decoded)
_i1_decoded = decoder_i1(_i2_decoded)
_x_decoded_mean = decoder_mean(_i1_decoded)
generator = Model(decoder_input, _x_decoded_mean)




vae = Model(x, x_decoded_mean)
vae.compile(optimizer='adam', loss=vae_loss)


"""
#ORIGINAL MODEL.
inputs = Input(shape=input_shape, name='encoder_input')

# VAE model = encoder + decoder
# build encoder model

x = Dense(intermediate_dim, activation='relu')(inputs)
y = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
y = Dense(intermediate_dim, activation='relu')(x)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)



# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')
"""



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, generator)
    data = (x_test, y_test)


    if args.weights:
        print("LOADING WEIGHTS")
        vae.load_weights(args.weights)
    else:
        es = EarlyStopping(monitor='loss', mode='min', verbose=1)
        # train the autoencoder

        score=vae.fit(x_train,
                      x_train,
                epochs=vae_config.NUMBER_EPOCHS,
                verbose=1,
                batch_size=vae_config.BATCH_SIZE,
                callbacks=[es])
        vae.save_weights('vae_mlp_mnist.h5')

        score2 = vae.predict(x_test, None, verbose=1)
        print('Score', score.history)
        print('Score', score2)




    plot_results(models,
                 data,
                 batch_size=vae_config.BATCH_SIZE)


