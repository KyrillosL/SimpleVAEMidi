from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from keras.layers import Lambda, Input, Dense
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

import vae_config


class Vae:

    def vae_loss(self,x, x_decoded_mean):
        reconstruction_loss = objectives.mse(x, x_decoded_mean)
        reconstruction_loss *= vae_config.ORIGINAL_DIM
        kl_loss = 1 + self.z_log_sigma - \
                  K.square(self.z_mean) - K.exp(self.z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        return vae_loss


    def __init__(self):

        self.epochs = vae_config.NUMBER_EPOCHS
        self.batch_size = vae_config.BATCH_SIZE

        self.latent_dim = vae_config.LATENT_DIM
        self.random_state = vae_config.random_state
        self.dataset_size = vae_config.dataset_size
        self.list_files_name= vae_config.list_files_name
        self.file_shuffle=vae_config.file_shuffle
        self.test_size=vae_config.test_size
        self.res =  vae_config.res

        self.range_of_notes_to_extract = 16
        self.number_of_data_to_extract = self.res * 2



        #path_midi_file_to_initialize_model = "/Users/Cyril_Musique/Documents/Cours/M2/MuGen/ressources/file_to_load_model/example_midi_file.mid"
        path_midi_file_to_initialize_model = "/home/kyrillos/CODE/VAEMIDI/MuGen-master/ressources/file_to_load_model/example_midi_file.mid"
        data_to_initialize_model = self.load_data(path_midi_file_to_initialize_model, 0, 2)

        self.original_dim = data_to_initialize_model[0].shape[1]

        self.vae, self.encoder, self.decoder = self.compile_model(data_to_initialize_model)

        weights = "vae_mlp_mnist.h5"
        print("LOADING WEIGHTS")
        self.vae.load_weights(weights)


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

        self.z_mean = Dense(latent_dim)(i5)
        self.z_log_sigma = Dense(latent_dim)(i5)

        # sample new similar points from the latent space
        z = Lambda(self.sampling, output_shape=(latent_dim,),
                   name='z')([self.z_mean, self.z_log_sigma])

        i5_decoded = decoder_i5(z)
        i4_decoded = decoder_i4(i5_decoded)
        i3_decoded = decoder_i3(i4_decoded)
        i2_decoded = decoder_i2(i3_decoded)
        i1_decoded = decoder_i1(i2_decoded)
        x_decoded_mean = decoder_mean(i1_decoded)

        # Encoder part of the model
        encoder = Model(x, self.z_mean)

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
        vae.compile(optimizer='adam', loss=self.vae_loss)
        return vae, encoder, generator

    def load_all_data(self,path, class_label, index_filename):
        features = []
        path, dirs, files = next(os.walk(path))
        num_size = len(dirs)
        current_folder = 0
        num_files = 0

        size = 465 * int(self.res / 8)
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
                            data = midi_data.get_piano_roll(fs=self.res)[35:50, :]

                            flattened = data.flatten()
                            flattened = flattened.astype(dtype=bool)

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
                            # print(file)
                            #self.list_files_name.insert(index_filename + num_files, file)
                            self.list_files_name.append( file)
                            num_files += 1
                        # except:
                        #    print("An exception occurred")
            current_folder += 1
            print("Done ", num_files, " from ", current_folder, " folders on ", num_size)

            featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

            print('Finished feature extraction from ', len(featuresdf), ' files')


            X = np.array(featuresdf.feature.tolist())
            y = np.array(featuresdf.class_label.tolist())


            data_to_plot_x = X
            data_to_plot_y = y

            return data_to_plot_x, data_to_plot_y

    def load_data(self ,path, class_label, index_filename ):
        features = []

        size = 465 * int(self.res / 8)
        # try:
        midi_data = pretty_midi.PrettyMIDI(path)
        for instrument in midi_data.instruments:
            instrument.is_drum = False
        if len(midi_data.instruments) > 0:
            data = midi_data.get_piano_roll(fs=self.res)[35:50, :]
            flattened = data.flatten()
            flattened = flattened.astype(dtype=bool)

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

            #self.list_files_name=path

        # except:
        #    print("An exception occurred")

            # Convert into a Panda dataframe
            featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

            print('Finished feature extraction from ', len(featuresdf), ' files')

            # Convert features & labels into numpy arrays
            listed_feature = featuresdf.feature.tolist()

            X = np.array(featuresdf.feature.tolist())
            y = np.array(featuresdf.class_label.tolist())

            # print(X.shape, y.shape)
            # split the dataset

            data_to_plot_x = X
            data_to_plot_y = y
            print("LOADED DATA", X)
            return data_to_plot_x, data_to_plot_y


    def get_distance(self, midi_file_path):

        data_to_plot =  self.load_data(midi_file_path, 2, 1)
        coord = self.get_coord( data_to_plot, batch_size=self.batch_size)

        x = coord[:, 0]
        y = coord[:, 1]
        # print(x,y)

        distance = math.sqrt((( (-4) - x) ** 2) + ((4 - y) ** 2))
        # print(distance)

        return distance



    def generate(self, data):

        encoded = self.encoder.predict(data)
        print( "encoded", encoded)
        z_sample = np.array([(0, 0), (0, 0)])
        decoded = self.decoder.predict(encoded)
        print("decoded", decoded)
        final = decoded[0]

        final2 = []
        for x in final:
            if x>0:
                final2.append(True)
            else:
                final2.append(False)
        print(final2)
        npfinal=np.array(final2)
        npfinal = npfinal.reshape(62,15)#.astype(dtype=bool) #self.number_of_data_to_extract, self.range_of_notes_to_extract



        print("final", final2)

        return npfinal

    def convert_to_midi(self, piano_roll):
        #piano_roll  = piano_roll[:, :, 0]
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(0)
        midi.instruments.append(instrument)
        padded_roll = np.pad(piano_roll, [(1, 1), (0, 0)], mode='constant')
        changes = np.diff(padded_roll, axis=0)
        notes = np.full(piano_roll.shape[1], -1, dtype=np.int)
        for tick, pitch in zip(*np.where(changes)):
            prev = notes[pitch]
            if prev == -1:
                notes[pitch] = tick
                continue
            notes[pitch] = -1
            instrument.notes.append(pretty_midi.Note(
                velocity=100,
                pitch=pitch,
                start=prev / float(self.res),
                end=tick / float(self.res)))

        midi.write("file.mid")
        return midi




