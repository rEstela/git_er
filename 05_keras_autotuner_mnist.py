import tensorflow as tf
from tensorflow import keras
import kerastuner as kt

# Get dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0
print('Dataset shape: ', x_train.shape, x_test.shape)

# Sem AutoTuner
model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(144, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
            )
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
            


# Com AutoTuner
"""
def model_builder(hp):
    model = tf.keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28,28)))
    
    hp_units = hp.Int('units', min_value=16, max_value=512, step=16)
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(10))

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
                )
    
    return model

tuner = kt.Hyperband(model_builder, 
                    objective='val_accuracy',
                    max_epochs=10,
                    factor=3,
                    directory='my_dir',
                    project_name='intro_to_kerastuner'
                    )

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(x_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])"""




