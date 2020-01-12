from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from keras.layers import Lambda, Input, Dense, Flatten, Reshape, Conv2DTranspose, Conv2D
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

class CVae:
    def __init__(self):
        print("ici")
        self.epochs = 10
        self.batch_size = 32
        self.intermediate_dim = 128
        self.latent_dim = 2
        self.random_state = 42
        self.dataset_size = 10000
        self.list_files_name= []
        self.file_shuffle=[]
        self.test_size=0.25
        self.res =  64 # min 8
        self.random_state = 42
        self.filters = 16
        self.kernel_size = 3
        self.range_of_notes_to_extract = 16
        self.number_of_data_to_extract = self.res * 2

        path_midi_file_to_initialize_model = "/Users/Cyril_Musique/Documents/Cours/M2/MuGen/ressources/file_to_load_model/example_midi_file.mid"
        #path_midi_file_to_initialize_model = "/home/kyrillos/CODE/VAEMIDI/MuGen-master/ressources/file_to_load_model/example_midi_file.mid"
        data_to_initialize_model = self.load_data(path_midi_file_to_initialize_model, 0, 2)

        self.original_dim = data_to_initialize_model[0].shape[1]

        self.vae, self.encoder, self.decoder = self.compile_model(data_to_initialize_model)

        weights = "all_dataset.h5"
        print("LOADING WEIGHTS")
        self.vae.load_weights(weights)

    def generate(self):

        z_sample = np.array([(0,0),(0,1)])#.astype(dtype=bool)
        x_decoded = self.decoder.predict(z_sample)#.astype(dtype=bool)
        reshaped = x_decoded[0].reshape(16, self.number_of_data_to_extract)
        print(x_decoded)


    def sampling(self,args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon


    def get_coord(self,data_to_plot,batch_size=128, show_annotation=False):

        data_to_plot_x,data_to_plot_y = data_to_plot
        # display a 2D plot of the digit classes in the latent space
        z_mean, _, _ = self.encoder.predict(data_to_plot_x,batch_size=batch_size)
        plt.figure(figsize=(12, 10))
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=data_to_plot_y)
        plt.colorbar()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        #print(len(z_mean))

        if show_annotation:
            print("DANS LANOATIONS")
            for i, txt in enumerate(self.list_files_name): #pour toute la dataset [ :int(dataset_size *2* test_size)]
                plt.annotate(txt,(z_mean[i,0], z_mean[i,1]))

        plt.show()
        return z_mean


    def compile_model(self, data_to_plot_x):
        # network parameters
        input_shape = (self.number_of_data_to_extract,)

        # Convolutional VAE

        # ENCODER
        input_shape = (self.number_of_data_to_extract, self.range_of_notes_to_extract, 1)  # datasize
        inputs = Input(shape=input_shape, name='encoder_input')
        x = inputs
        for i in range(2):
            self.filters *= 2
            x = Conv2D(filters=self.filters,
                       kernel_size=self.kernel_size,
                       activation='relu',
                       strides=2,
                       padding='same')(x)

        # shape info needed to build decoder model
        shape = K.int_shape(x)

        # generate latent vector Q(z|X)
        x = Flatten()(x)
        x = Dense(16, activation='relu')(x)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()

        # DECODER
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
        x = Reshape((shape[1], shape[2], shape[3]))(x)

        # use Conv2DTranspose to reverse the conv layers from the encoder
        for i in range(2):
            x = Conv2DTranspose(filters=self.filters,
                                kernel_size=self.kernel_size,
                                activation='relu',
                                strides=2,
                                padding='same')(x)
            self.filters //= 2

        outputs = Conv2DTranspose(filters=1,
                                  kernel_size=self.kernel_size,
                                  activation='sigmoid',
                                  padding='same',
                                  name='decoder_output')(x)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()

        # Building the VAE
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae')

        # LOSS
        use_mse = True
        if use_mse:
            reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
        else:
            reconstruction_loss = binary_crossentropy(K.flatten(inputs),
                                                      K.flatten(outputs))

        reconstruction_loss *= self.range_of_notes_to_extract * self.number_of_data_to_extract
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)

        # Compile the VAE
        vae.compile(optimizer='rmsprop')
        vae.summary()
        return vae, encoder, decoder

    def load_all_data(self,path, class_label, index_filename):
        features = []
        path, dirs, files = next(os.walk(path))
        num_size = len(dirs)
        current_folder = 0
        num_files = 0

        for subdir, dirs, files in os.walk(path):
            for file in files:
                if num_files < self.dataset_size:
                    if file != ".DS_Store":
                        # print(os.path.join(subdir, file))
                        # try:
                        midi_data = pretty_midi.PrettyMIDI(subdir + "/" + file)
                        for instrument in midi_data.instruments:
                            instrument.is_drum = False
                        if len(midi_data.instruments) > 0:
                            data = midi_data.get_piano_roll(fs=self.res)[35:51, 0:self.number_of_data_to_extract].astype(
                                dtype=bool)
                            data = data.flatten()
                            if data.size >= 16 * self.number_of_data_to_extract:
                                features.append([data, class_label])
                                self.list_files_name.insert(index_filename + num_files, file)
                            num_files += 1
                        # except:
                        #    print("An exception occurred")
            current_folder += 1
            print("Done ", num_files, " from ", current_folder, " folders on ", num_size)



            # Convert into a Panda dataframe
            featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

            print('Finished feature extraction from ', len(featuresdf), ' files')

            # Convert features & labels into numpy arrays
            listed_feature = featuresdf.feature.tolist()

            X = np.array(featuresdf.feature.tolist())
            y = np.array(featuresdf.class_label.tolist())

            print(X.shape, y.shape)

            X_shuffle = shuffle(X, random_state=self.random_state)
            y_shuffle = shuffle(y, random_state=self.random_state)
            file_shuffle = shuffle(self.list_files_name, random_state=self.random_state)

            data_to_plot_x = np.reshape(X, [-1, self.number_of_data_to_extract, self.range_of_notes_to_extract, 1])

            data_to_plot_y = y

            return data_to_plot_x, data_to_plot_y

    def load_data(self ,path, class_label, index_filename ):
        features = []

        midi_data = pretty_midi.PrettyMIDI(path)
        for instrument in midi_data.instruments:
            instrument.is_drum = False
        if len(midi_data.instruments) > 0:
            data = midi_data.get_piano_roll(fs=self.res)[35:51, 0:self.number_of_data_to_extract].astype(
                dtype=bool)
            data = data.flatten()
            if data.size >= 16 * self.number_of_data_to_extract:
                features.append([data, class_label])

        # except:
        #    print("An exception occurred")



            # Convert into a Panda dataframe
            featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

            print('Finished feature extraction from ', len(featuresdf), ' files')

            # Convert features & labels into numpy arrays
            listed_feature = featuresdf.feature.tolist()

            X = np.array(featuresdf.feature.tolist())
            y = np.array(featuresdf.class_label.tolist())

            print(X.shape, y.shape)

            X_shuffle = shuffle(X, random_state=self.random_state)
            y_shuffle = shuffle(y, random_state=self.random_state)
            file_shuffle = shuffle(self.list_files_name, random_state=self.random_state)

            data_to_plot_x = np.reshape(X, [-1, self.number_of_data_to_extract, self.range_of_notes_to_extract, 1])

            data_to_plot_y = y

            return data_to_plot_x, data_to_plot_y


    def get_distance(self, midi_file_path):

        data_to_plot =  self.load_data(midi_file_path, 2, 1)
        coord = self.get_coord( data_to_plot, batch_size=self.batch_size)

        x = coord[:, 0]
        y = coord[:, 1]
        # print(x,y)

        distance = math.sqrt((( (0) - x) ** 2) + ((0 - y) ** 2))
        # print(distance)

        return distance







