from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import matplotlib.pyplot as plt

import os

import pretty_midi
import pandas as pd
import numpy as np

from keras.layers import Dense,Activation, Dropout


# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
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
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,data,batch_size=128,model_name="vae_mnist"):
    """Plots labels and MNIST digits as a function of the 2D latent vector
    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean,_,_= encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    '''
    for y in y_test:
        if y==0:
            print("ONE HERE")
    '''
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()


# MNIST dataset


path = '/Users/Cyril_Musique/Documents/Cours/M2/PROJETALGOMUSIQUE/dataset_csv/midi_files/'
lakh_path = '/Users/Cyril_Musique/Documents/Cours/M2/SimpleVAEMIDI/Big_Data_Set/1/'
filepath = '/Users/Cyril_Musique/Documents/Cours/M2/PROJETALGOMUSIQUE/dataset_csv/dataset.csv'
metadata = pd.read_csv(filepath)


def load_data(training):
    features = []
    
    if training==True:
        
        # Iterate through each midi file and extract the features
        for index, row in metadata.iterrows():
            path_midi_file = path+ str(row["File"])
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
        '''
        #Itarating through the lakh midi dataset:
        for subdir, dirs, files in os.walk(lakh_path):
            for file in files:
                if file !=".DS_Store":
                    #print(os.path.join(subdir, file))
                    class_label=0
                    try:
                        midi_data = pretty_midi.PrettyMIDI(subdir+"/"+file)
                        for instrument in midi_data.instruments:
                            instrument.is_drum=False
                        if len(midi_data.instruments)>0:
                            data  = midi_data.get_piano_roll(fs=8)
                            data.resize(10000)
                            features.append([data, class_label])
                            #print("ADDED")
                    except:
                        print("An exception occurred")
    
        '''
        


    if training==False:
        
        # Iterate through each midi file and extract the features
        for index, row in metadata.iterrows():
            path_midi_file = path+ str(row["File"])
            if row["Score"] == 100:
                class_label = float(row["Score"]) / 100
                midi_data = pretty_midi.PrettyMIDI(path_midi_file)
                for instrument in midi_data.instruments:
                    instrument.is_drum=False
                if len(midi_data.instruments)>0:
                    data  = midi_data.get_piano_roll(fs=8)
                    data.resize(3968)
                    result = np.where(data == 80)

                    features.append([data, class_label])
    
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

    
    # Convert into a Panda dataframe
    featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

    print('Finished feature extraction from ', len(featuresdf), ' files')


    # Convert features & labels into numpy arrays
    X = np.array(featuresdf.feature.tolist())
    y = np.array(featuresdf.class_label.tolist())

    print(X.shape,y.shape)
    # split the dataset
    from sklearn.model_selection import train_test_split
    if training==True:
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
    else:
        x_train = X
        y_train =y

    midi_file_size = x_train.shape[1]

    # network parameters
    input_shape = (midi_file_size, )
    intermediate_dim = 512

    latent_dim = 2

    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    if training==True:
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


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
    plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(midi_file_size, activation='sigmoid')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')

    reconstruction_loss = mse(inputs, outputs)
    reconstruction_loss *= midi_file_size
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    #vae.add_loss(vae_loss)


    #loss = 'binary_crossentropy'
    loss='mean_squared_error'
    vae.compile(optimizer='adam', loss=loss)
    vae.summary()
    plot_model(vae,
               to_file='vae_mlp.png',
               show_shapes=True)

    if train:
        return vae, encoder, decoder, x_train, y_train, x_test, y_test
    else:
         return vae, encoder, decoder, x_train, y_train

if __name__ == '__main__':

    train = True
    

    epochs = 20
    batch_size = 128
    
    if train:
        vae, encoder, decoder, x_train, y_train, x_test, y_test =load_data(train)
        data = (x_train,y_train)
    else:
        vae, encoder, decoder, x_train, y_train =load_data(train)
        data = (x_train,y_train)

    models=(encoder,decoder)
    
    
    #models = (encoder, decoder)
    #data

    print("SIZE Y: ", y_train.size)

    if train:
        #train the autoencoder
        vae.fit(x_train,
                x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, x_test))
                #validation_data=(x_test, None))
        vae.save_weights('vae_mlp_mnist.h5')
    else:
        vae.load_weights("vae_mlp_mnist.h5")
    
    plot_results(models,data,batch_size=batch_size,model_name="vae_mlp")
        
