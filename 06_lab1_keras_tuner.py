# Import required packages
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

# Download the dataset and split into train and test sets
(img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()

# Normalize pixels values between 0 and 1
img_train = img_train.astype('float32') / 255.0
img_test = img_test.astype('float32') / 255.0

## BASELINE

# Build the baseline model using the Sequential API
b_model = keras.Sequential()
b_model.add(keras.layers.Flatten(input_shape=(28,28)))
b_model.add(keras.layers.Dense(units=52, activation='relu', name='dense_1')) # You will tune this layer later
b_model.add(keras.layers.Dropout(0.2))
b_model.add(keras.layers.Dense(10, activation='softmax'))

# Print model summary
b_model.summary()

# Setup the training parameters
b_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy']
                )

# Number of training Epochs
NUM_EPOCHS = 10

# Train the model
b_model.fit(img_train, label_train, epochs=NUM_EPOCHS, validation_split=0.2)

# Evaluate model on the test set
b_eval_dict = b_model.evaluate(img_test, label_test, return_dict=True)
#print('Baseline Model Evaluation: \n', b_eval_dict)

# Define helper function
# for displaying the results to compare later

def print_results(model, model_name, layer_name, eval_dict):
    '''
    Print the values of the hyperparameters to tune, and the results of the model evaluation

    Args.
        model (Model): keras model to evaluate
        model_name (string): arbitrary string to be used in identifying the model
        layer_name(string): name of the layer to tune
        eva_dict (dict): results of model.evaluate
    '''
    print(f'\n{model_name}:')
    print(f'- number of units in 1st Dense layer: {model.get_layer(layer_name).units}')
    print(f'- learning rate for the optimizer: {model.optimizer.lr.numpy()}')

    for key, value in eval_dict.items():
        print(f'- {key}: {value}')

# print results for baseline model
#print_results(b_model, 'BASELINE MODEL', 'dense_1', b_eval_dict)

## KERAS TUNER

def model_builder(hp):
    '''
    Builds the model and sets up the hyperparamenters to tune.

    Args.
        hp: Keras tuner object
    
    Returns.
        model with hyperparameters to tune
    '''

    # Initialize the Sequential API and start stacking the layers
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28,28)))

    # Tune the number of units in the first Dense Layer
    # Choose an optimal value between 32-512
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_units, activation='relu', name='tuned_dense_1'))

    # Add next layers
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(10, activation='softmax'))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy']
                )
    
    return model

# Instantiate the tuner
tuner = kt.Hyperband(model_builder,
                    objective='val_accuracy',
                    max_epochs=10,
                    factor=3,
                    directory='keras_tuner_dir',
                    project_name='kt_hyperband'                    
                    )

# Display hypertuning settings
tuner.search_space_summary()

# Set early stop
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Perform hypertuning
tuner.search(img_train, label_train, epochs=NUM_EPOCHS, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters from the results
best_hps = tuner.get_best_hyperparameters()[0]

print(f'''
    The hyperparameter search is complete. \n
    The optimal number of units in the first densly-connected layer is {best_hps.get('units')} and the optimal learning rate 
    for the optimizer is {best_hps.get('learning_rate')}.
    ''')

# Buld the model with the optimal hyperparameters
h_model = tuner.hypermodel.build(best_hps)
h_model.summary()

# Train the hypertuned model
h_model.fit(img_train, label_train, epochs=NUM_EPOCHS, validation_split=0.2)

# Evaluate the hypertuned model against the test set
h_eval_dict = h_model.evaluate(img_test, label_test, return_dict=True)

# Print results of the baseline and hypertuned model
print_results(b_model, 'BASELINE MODEL', 'dense_1', b_eval_dict)
print_results(h_model, 'HYPERTUNED MODEL', 'tuned_dense_1', h_eval_dict)