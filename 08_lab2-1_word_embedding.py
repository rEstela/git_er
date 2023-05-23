import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.datasets import reuters
from keras.preprocessing import sequence

from keras.utils import np_utils, pad_sequences

from tensorflow.keras import layers

num_words = 1000

(reuters_train_x, reuters_train_y), (reuters_test_x, reuters_test_y) = tf.keras.datasets.reuters.load_data(num_words=num_words)

n_labels = np.unique(reuters_train_y).shape[0]

reuters_train_y = np_utils.to_categorical(reuters_train_y, 46)
reuters_test_y = np_utils.to_categorical(reuters_test_y, 46)

reuters_train_x = pad_sequences(reuters_train_x, maxlen=20)
reuters_test_x = pad_sequences(reuters_test_x, maxlen=20)

model = tf.keras.Sequential(
    [
        layers.Embedding(num_words, 6, input_length=20),
        layers.Flatten(),
        layers.Dense(256),
        layers.Dropout(0.25),
        layers.Activation('relu'),
        layers.Dense(46),
        layers.Activation('softmax')
    ]
)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model_1 = model.fit(reuters_train_x, reuters_train_y,
                    validation_data=(reuters_test_x, reuters_test_y),
                    batch_size=128, epochs=20
                    )