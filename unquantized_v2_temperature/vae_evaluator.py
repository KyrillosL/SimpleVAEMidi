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
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

import pretty_midi
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.utils import shuffle

intermediate_dim = 512
batch_size = 128
latent_dim = 2
epochs = 10
random_state = 42
dataset_size = 5000
list_files_name= []
file_shuffle=[]
test_size=0.25

res =  512 # min 8

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data_to_plot,
                 batch_size=128,
                 model_name="vae_mnist"):

    data_to_plot_x,data_to_plot_y = data_to_plot
    encoder, decoder = models

    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(data_to_plot_x,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=data_to_plot_y)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    #print(len(z_mean))
    nb_elem_per_class = dataset_size*test_size

    to_decode = np.array([[0.5, 0], [1.8, 1]], dtype=np.float32)
    final = decoder.predict(to_decode)
    print(final )

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


    size = 465 * int(res / 8)
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if num_files < dataset_size:
                if file != ".DS_Store":
                    # print(os.path.join(subdir, file))
                    # try:
                    midi_data = pretty_midi.PrettyMIDI(subdir + "/" + file)
                    for instrument in midi_data.instruments:
                        instrument.is_drum = False
                    if len(midi_data.instruments) > 0:
                        data = midi_data.get_piano_roll(fs=res)[35:50, :]

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
                        list_files_name.insert(index_filename+num_files, file)
                        num_files += 1
                    # except:
                    #    print("An exception occurred")
        current_folder += 1
        print("Done ", num_files, " from ", current_folder, " folders on ", num_size)
        return True


print("LOADING DATA FOR TRAINING...")
features = []

'''
path_to_load = "/Users/Cyril_Musique/Documents/Cours/M2/MuGen/datasets/quantized_rythm_dataset_v2_temperature/0"
load_data(path_to_load, 0,   0)
path_to_load = "/Users/Cyril_Musique/Documents/Cours/M2/MuGen/datasets/quantized_rythm_dataset_v2_temperature/100"
load_data(path_to_load,1,  dataset_size)

'''


path_to_load = "/Users/Cyril_Musique/Documents/Cours/M2/MuGen/datasets/quantized_rythm_dataset_v2_temperature/to_compare_0"
load_data(path_to_load, -2,   0)
path_to_load = "/Users/Cyril_Musique/Documents/Cours/M2/MuGen/datasets/quantized_rythm_dataset_v2_temperature/to_compare_100"
load_data(path_to_load,2,  dataset_size)


# Convert into a Panda dataframe
featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

print('Finished feature extraction from ', len(featuresdf), ' files')

# Convert features & labels into numpy arrays
listed_feature = featuresdf.feature.tolist()

X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

print(X.shape, y.shape)
# split the dataset

data_to_plot_x = X
data_to_plot_y = y

midi_file_size = data_to_plot_x.shape[1]
original_dim = midi_file_size


# network parameters
input_shape = (original_dim, )


# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()


# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()




# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

if __name__ == '__main__':

    models = (encoder, decoder)

    reconstruction_loss = mse(inputs, outputs)

    #reconstruction_loss = binary_crossentropy(inputs,outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    opt = Adam(lr=0.0005)  # 0.001 was the default, so try a smaller one
    vae.compile(optimizer=opt,  metrics=['accuracy'])
    vae.summary()


    weights = "vae_mlp_mnist.h5"
    print("LOADING WEIGHTS")
    vae.load_weights(weights)


    data_to_plot = (data_to_plot_x, data_to_plot_y)

    plot_results(models,
                 data_to_plot,
                 batch_size=batch_size,
                 model_name="vae_mlp")
