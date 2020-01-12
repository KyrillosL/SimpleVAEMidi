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

intermediate_dim = 128
batch_size = 32
latent_dim = 2
epochs = 100
random_state = 42
dataset_size = 10
list_files_name= []
file_shuffle=[]
test_size=0.25
timesteps=1
res =  8 # min 8
filters = 16
kernel_size = 3

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


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
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
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    #print(len(z_mean))
    nb_elem_per_class = dataset_size*test_size

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

path_to_load = "/home/kyrillos/CODE/VAEMIDI/quantized_rythm_dataset_v2_temperature/0"
load_data(path_to_load, 0,   0)
path_to_load = "/home/kyrillos/CODE/VAEMIDI/quantized_rythm_dataset_v2_temperature/100"
load_data(path_to_load,1,  dataset_size)

# Convert into a Panda dataframe
featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

print('Finished feature extraction from ', len(featuresdf), ' files')

# Convert features & labels into numpy arrays
listed_feature = featuresdf.feature.tolist()

X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

print(X.shape, y.shape)
# split the dataset


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
X_shuffle = shuffle(X, random_state=random_state)
y_shuffle = shuffle(y, random_state=random_state)
file_shuffle = shuffle(list_files_name, random_state=random_state)


print("X_test: ", x_test)
to_compare  = X_shuffle[:int(dataset_size*2 * test_size)]
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


'''
# LSTM encoding
lstm_input = Input(shape=(timesteps, midi_file_size,))
lstm = LSTM(intermediate_dim)(lstm_input)

# VAE model = encoder + decoder
# build encoder model

z_mean = Dense(latent_dim, name='z_mean')(lstm)
z_log_var = Dense(latent_dim, name='z_log_var')(lstm)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# build decoder model
#latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
#x = Dense(intermediate_dim, activation='relu')(latent_inputs)

decoder_h = LSTM(intermediate_dim, return_sequences=True)
decoder_mean = LSTM(midi_file_size, return_sequences=True)
h_decoded = RepeatVector(timesteps)(z)
h_decoded = decoder_h(h_decoded)

# decoded layer
x_decoded_mean = decoder_mean(h_decoded)


# instantiate VAE model
#outputs = decoder(encoder(inputs)[2])
#vae = Model(inputs, outputs, name='vae_mlp')
vae = Model(lstm_input, x_decoded_mean)

# instantiate encoder model
encoder = Model(lstm_input,z_mean, name='encoder')
encoder.summary()

decoder_input = Input(shape=(latent_dim,))

_h_decoded = RepeatVector(timesteps)(decoder_input)
_h_decoded = decoder_h(_h_decoded)

outputs = decoder_mean(_h_decoded)
decoder = Model(decoder_input, outputs)

decoder.summary()
'''

'''
#Convolutional VAE

inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
for i in range(2):
    filters *= 2
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)

# shape info needed to build decoder model
shape = K.int_shape(x)

# generate latent vector Q(z|X)
x = Flatten()(x)
x = Dense(16, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
'''







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    data = (x_test, y_test)

    # VAE loss = mse_loss or xent_loss + kl_loss
    reconstruction_loss = mse(inputs, outputs)
    #reconstruction_loss = binary_crossentropy(inputs,outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)

    # LSTM
    '''
    xent_loss = objectives.mse(lstm_input, outputs)
    kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
    loss = xent_loss + kl_loss
    vae.add_loss(loss)
    '''

    vae.add_loss(vae_loss)
    opt = Adam(lr=0.0005)  # 0.001 was the default, so try a smaller one
    vae.compile(optimizer=opt,  metrics=['accuracy'])
    vae.summary()
    plot_model(vae,
               to_file='vae_mlp.png',
               show_shapes=True)

    if args.weights:
        print("LOADING WEIGHTS")
        vae.load_weights(args.weights)
    else:
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
        # train the autoencoder

        score=vae.fit(x_train,
                epochs=epochs,
                verbose=1,
                batch_size=batch_size,
                validation_data=(x_test, None),
                callbacks=[es])
        vae.save_weights('vae_mlp_mnist.h5')

        score2 = vae.evaluate(x_test, None, verbose=1)
        print('Score', score.history)
        print('Score', score2)




    plot_results(models,
                 data,
                 batch_size=batch_size,
                 model_name="vae_mlp")



def test_to_compare():
    features_to_compare = []
    path_to_load = "/Users/Cyril_Musique/Documents/Cours/M2/MuGen/datasets/quantized_rythm_dataset_v2_temperature/to_compare_0"

    path, dirs, files = next(os.walk(path_to_load))
    num_size = len(dirs)
    current_folder = 0
    num_files = 0
    res = 8  # min 8

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
                            features_to_compare.append([final_array, 0])

                        else:
                            remaining_zero = data.size - size
                            final_array = flattened[0:flattened.size - remaining_zero]
                            if np.count_nonzero(final_array) == 0:
                                print("0 IN DATASET")
                            features_to_compare.append([final_array, 0])
                        #print(file)
                        num_files += 1
                    # except:
                    #    print("An exception occurred")
        current_folder += 1
        print("Done ", num_files, " from ", current_folder, " folders on ", num_size)

    featuresdf2 = pd.DataFrame(features_to_compare, columns=['feature', 'class_label'])

    to_compare_final = np.array(featuresdf2.feature.tolist())

    for x in to_compare_final:
        if x in x_test:
            print("IS INSIDE")
        else:
            print("NOT INSIDE")

    path_to_load = "/Users/Cyril_Musique/Documents/Cours/M2/MuGen/datasets/quantized_rythm_dataset_v2_temperature/to_compare_100"

    path, dirs, files = next(os.walk(path_to_load))
    num_size = len(dirs)
    current_folder = 0
    num_files = 0
    res = 8  # min 8

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
                            features_to_compare.append([final_array, 0])

                        else:
                            remaining_zero = data.size - size
                            final_array = flattened[0:flattened.size - remaining_zero]
                            if np.count_nonzero(final_array) == 0:
                                print("0 IN DATASET")
                            features_to_compare.append([final_array, 0])
                        # print(file)
                        num_files += 1
                    # except:
                    #    print("An exception occurred")
        current_folder += 1
        print("Done ", num_files, " from ", current_folder, " folders on ", num_size)

    featuresdf2 = pd.DataFrame(features_to_compare, columns=['feature', 'class_label'])

    to_compare_final = np.array(featuresdf2.feature.tolist())


    for x in to_compare_final:
        if x in x_test:
            print("IS INSIDE")
        else:
            print("NOT INSIDE")
    return True