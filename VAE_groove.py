from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.optimizers import Adam
import argparse
import math

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import os

import pretty_midi
import pandas as pd
import numpy as np

from keras.layers import Dense,Activation, Dropout


class Vae:
    def __init__(self):
        self.epochs = 3
        self.batch_size = 128
        self.intermediate_dim = 512
        self.latent_dim = 4

        print("LOADING VAE MODEL...")

        #x_train, y_train = self.load_data_for_evaluation("example_midi_file.mid")

        #self.vae, self.encoder, self.decoder = self.compile_model(x_train)

        #if os.path.exists("vae_midi.h5"):
        #    self.vae.load_weights("vae_midi.h5")
        '''
        else:
            #self.ask_train()
            user_response = input("no model found to load. Do you want to train ? [y/n]")
            print(user_response)
            if user_response == 'y':
                print("DANS LE YES")
                self.train()
            else:
                return None

        '''
        #self.models = (self.encoder, self.decoder)

        self.path = '/Users/Cyril_Musique/Documents/Cours/Dataset_NO_GD/dataset_csv/midi_files/'
        self.lakh_path = '/home/kyrillos/CODE/VAEMIDI/clean_midi/'
        self.generated_path = "/Users/Cyril_Musique/Documents/Cours/M2/MuGen/datasets/quantized_rythm_dataset/100"
        self.random_path = "/Users/Cyril_Musique/Documents/Cours/M2/MuGen/datasets/quantized_rythm_dataset/0"

        self.ontime = "/Users/Cyril_Musique/Documents/Cours/M2/MuGen/datasets/quantized_rythm_dataset/ontime32"
        self.notontime = "/Users/Cyril_Musique/Documents/Cours/M2/MuGen/datasets/quantized_rythm_dataset/notontime32"

        filepath = '/Users/Cyril_Musique/Documents/Cours/Dataset_NO_GD/dataset_csv/dataset.csv'
        #self.metadata = pd.read_csv(filepath)



    # reparameterization trick
    # instead of sampling from Q(z|X), sample epsilon = N(0,I)
    # z = z_mean + sqrt(var) * epsilon
    def sampling(self,args):
        """Reparameterization trick by sampling from an isotropic unit Gaussian.
        # Arguments
            args (tensor): mean and log of variance of Q(z|X)
        # Returns
            z (tensor): sampled latent vector
        """

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    #def ask_train(self):


    def get_coord(self,models,data,batch_size=128,model_name="vae_mnist"):
        encoder, decoder = models
        x_test, y_test = data
        os.makedirs(model_name, exist_ok=True)

        filename = os.path.join(model_name, "vae_mean.png")
        # display a 2D plot of the digit classes in the latent space
        z_mean,_,_= encoder.predict(x_test,
                                       batch_size=batch_size)

        #print(z_mean)
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
        plt.colorbar()
        plt.show()
        return z_mean




    def compile_model(self, x_train):

        print("COMPILING MODEL")
        midi_file_size = x_train.shape[1]

        input_shape = (midi_file_size,)
        # VAE model = encoder + decoder
        # build encoder model
        inputs = Input(shape=input_shape, name='encoder_input')
        x = Dense(self.intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()
        #plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

        # build decoder model
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        x = Dense(self.intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(midi_file_size, activation='sigmoid')(x)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()
        #plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae_mlp')

        reconstruction_loss = mse(inputs, outputs)
        #midi_file_size = midi_file_size*midi_file_size
        #reconstruction_loss *= midi_file_size
        #kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        #kl_loss = K.sum(kl_loss, axis=-1)
        #kl_loss *= -0.5
        #vae_loss = K.mean(reconstruction_loss )#+ kl_loss)
        #vae.add_loss(vae_loss)


        #loss = 'binary_crossentropy'
        loss = 'mean_squared_error'


        #opt = Adam(lr=0.00005)  # 0.001 was the default, so try a smaller one
        opt = Adam(lr=0.00005)  # 0.001 was the default, so try a smaller one
        vae.compile(optimizer=opt, loss=loss)
        #vae.compile(optimizer='Adam', loss=loss)

        #vae.compile(optimizer='adam')
        vae.summary()
        #plot_model(vae,to_file='vae_mlp.png',show_shapes=True)

        return vae, encoder, decoder

    def load_data_for_evaluation(self,path_to_plot):
        features = []
        '''
        # Iterate through each midi file and extract the features
        for index, row in self.metadata.iterrows():
            path_midi_file = self.path + str(row["File"])
            if row["Score"] == 100:
                class_label = float(row["Score"]) / 100
                midi_data = pretty_midi.PrettyMIDI(path_midi_file)
                for instrument in midi_data.instruments:
                    instrument.is_drum = False
                if len(midi_data.instruments) > 0:
                    data = midi_data.get_piano_roll(fs=8)
                    data.resize(3968)
                    result = np.where(data == 80)

                    features.append([data, class_label])
        '''
        '''
        # GRAB A 50 AND CALCULATE ITS DISTANCE
        for index, row in metadata.iterrows():
            path_midi_file = path+ str(row["File"])
            if row["File"] == "27_random.mid":
                class_label = float(row["Score"]) / 100
                midi_data = pretty_midi.PrettyMIDI(path_midi_file)
                for instrument in midi_data.instruments:
                    instrument.is_drum=False
                if len(midi_data.instruments)>0:
                    data  = midi_data.get_piano_roll(fs=8)
                    data.resize(3968)
                    result = np.where(data == 80)
    
                    features.append([data, class_label])
        '''
        class_label = 0
        midi_data = pretty_midi.PrettyMIDI(path_to_plot)

        for instrument in midi_data.instruments:
            instrument.is_drum = False
        if len(midi_data.instruments) > 0:
            data = midi_data.get_piano_roll(fs=8)
            data.resize(3968)
            result = np.where(data == 80)

            features.append([data, class_label])

        # Convert into a Panda dataframe
        featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

        #print('Finished feature extraction from ', len(featuresdf), ' files')

        # Convert features & labels into numpy arrays
        X = np.array(featuresdf.feature.tolist())
        y = np.array(featuresdf.class_label.tolist())

        #print(X.shape, y.shape)
        # split the dataset


        x_train = X
        y_train = y

        #midi_file_size = x_train.shape[1]

        # network parameters
        #input_shape = (midi_file_size,)


        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))


        #return vae, encoder, decoder, x_train, y_train
        return  x_train, y_train


    def load_data_for_training(self):
        print("LOADING DATA FOR TRAINING...")
        test_new_feature= []
        features = []
        '''
        # Iterate through each midi file and extract the features
        for index, row in self.metadata.iterrows():
            path_midi_file = self.path+ str(row["File"])
            #if row["Score"] == 0 or row["Score"] == 100:
            if row["Score"] !=89:
                class_label = float(row["Score"]) / 100
                midi_data = pretty_midi.PrettyMIDI(path_midi_file)
                for instrument in midi_data.instruments:
                    instrument.is_drum=False
                if len(midi_data.instruments)>0:
                    data  = midi_data.get_piano_roll(fs=8)
                    #data.resize(3968)
                    data.resize(3968)
                    #np.pad(data, [(0,0), (0, 10000-data.shape[1])], 'constant')
                    #print(data.size)
                    result = np.where(data == 80)
                    features.append([data, class_label])
        
        #Itarating through the lakh midi dataset:
        #len(list(filter(os.path.isdir, os.listdir(self.lakh_path))))
        #num_size =  sum(os.path.isdir(i) for i in os.listdir(self.lakh_path))
        path, dirs, files = next(os.walk(self.lakh_path))
        num_size = len(dirs)
        current_folder = 0
        num_files = 0
        for subdir, dirs, files in os.walk(self.lakh_path):

            for file in files:
                if num_files<2:
                    if file !=".DS_Store":
                        print(os.path.join(subdir, file))
                        class_label=0
                        #try:
                        midi_data = pretty_midi.PrettyMIDI(subdir+"/"+file)
                        for instrument in midi_data.instruments:
                            instrument.is_drum=False
                        if len(midi_data.instruments)>0:
                            data  = midi_data.get_piano_roll(fs=8).astype(object)
                            #data.flatten()
                            data.resize(data.size, refcheck=False)
                            #data.resize(3968, refcheck=False)
                            features.append([data, class_label])
                            num_files+=1
                        #except:
                        #    print("An exception occurred")
            current_folder+=1
            print("Done ", num_files," from ",current_folder," folders on ",num_size )
        '''
        # Itarating through the lakh midi dataset:
        # len(list(filter(os.path.isdir, os.listdir(self.lakh_path))))
        # num_size =  sum(os.path.isdir(i) for i in os.listdir(self.lakh_path))


        path_to_load = self.random_path
        path, dirs, files = next(os.walk(path_to_load))
        num_size = len(dirs)
        current_folder = 0
        num_files = 0
        res = 128

        size = 3968*int(res/8)#
        for subdir, dirs, files in os.walk(path_to_load):
            for file in files:
                if num_files < 5000:
                    if file != ".DS_Store":
                        #print(os.path.join(subdir, file))
                        class_label = 0
                        # try:
                        midi_data = pretty_midi.PrettyMIDI(subdir + "/" + file)
                        for instrument in midi_data.instruments:
                            instrument.is_drum = False
                        if len(midi_data.instruments) > 0:
                            data = midi_data.get_piano_roll(fs=res)#[35:50,:]

                            flattened = data.flatten()
                                # data.resize(data.size, refcheck=False)
                                # np.resize(data, (1,465)
                            final_array = []
                            if data.size<=size:
                                remaining_zero = size - data.size
                                final_array = np.pad(flattened, (0, remaining_zero), mode='constant', constant_values=0)
                                if np.count_nonzero(final_array) == 0:
                                    print("0 IN DATASET")
                                    return False
                                features.append([final_array, class_label])

                            else:
                                remaining_zero = data.size- size
                                final_array = flattened[0:flattened.size-remaining_zero]
                                if np.count_nonzero(final_array) == 0:
                                    print("0 IN DATASET")
                                    return False
                                features.append([final_array, class_label])


                            num_files += 1
                        # except:
                        #    print("An exception occurred")
            current_folder += 1
            print("Done ", num_files, " from ", current_folder, " folders on ", num_size)

        path_to_load = self.generated_path
        path, dirs, files = next(os.walk(path_to_load))
        num_size = len(dirs)
        current_folder = 0
        num_files = 0
        for subdir, dirs, files in os.walk(path_to_load):
            for file in files:
                if num_files < 5000:
                    if file != ".DS_Store":
                        #print(os.path.join(subdir, file))
                        class_label = 1
                        # try:
                        midi_data = pretty_midi.PrettyMIDI(subdir + "/" + file)
                        for instrument in midi_data.instruments:
                            instrument.is_drum = False
                        if len(midi_data.instruments) > 0:
                            data = midi_data.get_piano_roll(fs=res)#[35:50, :]
                            flattened = data.flatten()
                            # data.resize(data.size, refcheck=False)
                            # np.resize(data, (1,465)
                            final_array = []
                            if data.size <= size:
                                remaining_zero = size - data.size
                                final_array = np.pad(flattened, (0, remaining_zero), mode='constant', constant_values=0)
                                if np.count_nonzero(final_array)==0:
                                    print("0 IN DATASET")
                                    return False
                                features.append([final_array, class_label])

                            else:
                                remaining_zero = data.size - size
                                final_array = flattened[0:flattened.size - remaining_zero]
                                if np.count_nonzero(final_array) == 0:
                                    print("0 IN DATASET")
                                    return False

                                features.append([final_array, class_label])


                            num_files += 1
                        # except:
                        #    print("An exception occurred")
            current_folder += 1
            print("Done ", num_files, " from ", current_folder, " folders on ", num_size)

        # Convert into a Panda dataframe
        featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

        print('Finished feature extraction from ', len(featuresdf), ' files')

        # Convert features & labels into numpy arrays
        listed_feature =featuresdf.feature.tolist()

        X = np.array(featuresdf.feature.tolist())
        y = np.array(featuresdf.class_label.tolist())

        print(X.shape,y.shape)
        # split the dataset


        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
        #midi_file_size = x_train.shape[1]
       # x_train = np.reshape(x_train, [-1, 3968])
        #x_test = np.reshape(x_test, [-1, 3968])

        #print(x_train.shape)


        return   x_train, y_train, x_test, y_test

    def train(self):

        file = "vae_midi.h5"
        if os.path.exists(file):
            os.remove(file)
        print("Removed h5 model file")

        x_train, y_train, x_test, y_test = self.load_data_for_training()
        data = (x_train, y_train)

        self.vae, self.encoder, self.decoder = self.compile_model(x_train)

        self.models  = (self.encoder, self.decoder)


        # train the autoencoder
        self.vae.fit(x_train,
                x_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(x_test, x_test))
                #validation_data=(x_test, None))
        print("OUTPUT", self.vae.output)
        self.vae.save_weights('vae_midi.h5')

        coord = self.get_coord(self.models, data, batch_size=self.batch_size, model_name="vae_mlp")

    def get_distance(self,midi_file_path):


        x_train, y_train = self.load_data_for_evaluation(midi_file_path)
        #vae, encoder, decoder = self.compile_model(x_train)

        data = (x_train, y_train)

        coord = self.get_coord(self.models,data,batch_size=self.batch_size,model_name="vae_mlp")

        x = coord[:, 0]
        y = coord[:, 1]
        #print(x,y)

        distance = math.sqrt(   ( (0-x) **2)  +  ( (0-y) **2) )
        #print(distance)

        return distance

    
    
        
