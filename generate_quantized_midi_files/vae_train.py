

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
from mido import MidiFile, MidiTrack, Message as MidiMessage


intermediate_dim = 128
batch_size = 32
latent_dim = 2
epochs = 100
random_state = 42
dataset_size = 10000
list_files_name= []
file_shuffle=[]
test_size=0.05
timesteps=1
res =  8 # min 8
filters = 16
kernel_size = 3

len_cleaned_dataset=0



def convert_to_midi( piano_roll):
    # piano_roll  = piano_roll[:, :, 0]

    fs =8
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=1)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=127,
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    pm.write("file.mid")


def sampling(args):
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
    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)


    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")

    #print(len(z_mean))
    nb_elem_per_class = dataset_size*test_size

    #to_decode = np.array([[0.5, 0], [1.8, 1]], dtype=np.float32)
    #final = decoder.predict(to_decode)
    #print(final )

   # print("ICI ", file_shuffle[:int(dataset_size * test_size)])
    print(file_shuffle)
    for i, txt in enumerate(file_shuffle[ :int(len_cleaned_dataset * test_size)]):
        #print("i ", i)
        plt.annotate(txt,(z_mean[i,0], z_mean[i,1]))

    plt.show()


def load_data(path, class_label, index_filename ):

    path, dirs, files = next(os.walk(path))
    num_size = len(dirs)
    current_folder = 0
    num_files = 0


    size = 480 * int(res / 8)
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
                        data = midi_data.get_piano_roll(fs=res)[36:51, :]

                        flattened = data.flatten()
                        flattened = flattened.astype(dtype=bool)

                        if data.size >= size:
                            features.append([flattened, class_label])
                            list_files_name.insert(index_filename + num_files, file)
                        #else:
                        #   print("skipping this one")
                        #print(file)

                        num_files += 1
                    # except:
                    #    print("An exception occurred")
        current_folder += 1
        print("Done ", num_files, " from ", current_folder, " folders on ", num_size)
        len_cleaned_dataset=len(features)


        return True


print("LOADING DATA FOR TRAINING...")
features = []

path_to_load = "/Users/Cyril_musique/Documents/cours/M2/MuGen/datasets/quantized_rythm_dataset_v2_temperature/100"
load_data(path_to_load, 0,   0)
#path_to_load = "/home/kyrillos/CODE/VAEMIDI/quantized_rythm_dataset_v2_temperature/100"
#load_data(path_to_load,1,  dataset_size)

# Convert into a Panda dataframe
featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

print('Finished feature extraction from ', len(featuresdf), ' files')





# Convert features & labels into numpy arrays
listed_feature = featuresdf.feature.tolist()

X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())


#convert_to_midi(X[0].reshape(15,32))

print(X.shape, y.shape)
# split the dataset


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
X_shuffle = shuffle(X, random_state=random_state)
y_shuffle = shuffle(y, random_state=random_state)
file_shuffle = shuffle(list_files_name, random_state=random_state)





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


# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
y = Dense(intermediate_dim, activation='relu')(x)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()




# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')





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


    vae.add_loss(vae_loss)
    opt = Adam(lr=0.0005)  # 0.001 was the default, so try a smaller one
    vae.compile(optimizer=opt,  metrics=['accuracy'])
    vae.summary()


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
                 batch_size=batch_size)