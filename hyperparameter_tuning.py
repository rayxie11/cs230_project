'''
This file contains the HyperParamter tuning for the RNN model for choreography classification 
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, LSTM, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
#from keras import layers, Model
import matplotlib.pyplot as plt


class rnn_model():
    '''
    This class contains the RNN model
    '''
    def __init__(self, X, Y, model_parameters):
        '''
        1. Define the input, output, and different model paramters
        2. Build model
        3. Run model
        4. Plot loss and accuracy curvers
        '''
        self.X = X
        self.Y = Y
        _, x, y = X.shape
        _, n = Y.shape
        self.input = (x,y)
        self.output = n
        self.dropout_rate = model_parameters["dropout_rate"]
        self.lstm_neurons = model_parameters["lstm_neurons"]
        self.fc1_neuron = model_parameters["fc1_neurons"]
        self.fc2_neuron = model_parameters["fc2_neurons"]
        self.split = model_parameters["split"]
        self.epochs = model_parameters["epoch"]
        self.batch_size = model_parameters["batch_size"]

        tuner = kt.Hyperband(hypermodel=self.model_builder,
            objective='val_accuracy',
            max_epochs=10,
            factor=3,
            directory='param_tuning', #Where tests will be stored.
            project_name='hyperTuning')

        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        tuner.search(self.X, self.Y, validation_split=self.split, epochs=self.epochs, batch_size=self.batch_size, callbacks=[stop_early])
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

        print(f"""
        The hyperparameter search is complete. The optimal number of units in the first densely-connected
        layer is {best_hps.get('fc_units')} and the optimal learning rate for the optimizer
        is {best_hps.get('learning_rate')}. Best lstm neurons is {best_hps.get('lstm_units')} and best
        activation for dense layer is {best_hps.get('activation')}
        """)
        #self.model()
        #self.run_plot()


    def model_builder(self, hp):
        '''
        RNN structure
        '''
        model = Sequential()
        model.add(Input(shape=self.input))
        model.add(BatchNormalization())
        
        #Tuning Hyperparameters
        fc_units = hp.Int('fc_units', min_value=32, max_value=512, step=32)
        lstm_units = hp.Int('lstm_units', min_value=32, max_value=512, step=32)
        fc_act = hp.Choice("activation", ["relu", "selu", "tanh"]) #dense activations
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        #Building Model layers
        model.add(Bidirectional(LSTM(lstm_units)))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(fc_units, activation=fc_act))
        model.add(Dense(fc_units/2,activation=fc_act))
        model.add(Dense(self.output))
        model.add(Activation('softmax'))
        


        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=hp_learning_rate), metrics=['accuracy'])
        self.model = model
        return model
    
    def run_plot(self):
        '''
        Run model and plot accuracy and loss curves
        '''
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = self.model.fit(self.X, self.Y, validation_split=self.split, epochs=self.epochs, batch_size=self.batch_size)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()



if __name__ == '__main__':
    # Define dataset path
    #dataset_path = "C:/Users/ray_s/Desktop/cs230_project/dataset/generated_dataset/image/"
    #dataset_path = "C:/Users/ray_s/Desktop/cs230_project/dataset/generated_dataset/image_sub/"
    dataset_path = "/Users/tkanell/Downloads/School/cs230/dataset/"

    # Load data and labels
    X = np.load(dataset_path + "data.npy")
    Y = np.load(dataset_path + "label.npy")

    print(X.shape)
    
    # Set model parameters
    lstm_neurons = 448
    dropout_rate = 0.2
    fc1_neurons = 32
    fc2_neurons = 16
    train_valid_split = 0.1
    num_epochs = 20
    batch_size = 128

    model_parameters = {"lstm_neurons":lstm_neurons,
                        "dropout_rate":dropout_rate,
                        "fc1_neurons":fc1_neurons,
                        "fc2_neurons":fc2_neurons,
                        "split":train_valid_split,
                        "epoch":num_epochs,
                        "batch_size":batch_size}
    
    # Run RNN model
    model = rnn_model(X, Y, model_parameters)
