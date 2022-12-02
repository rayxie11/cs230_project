'''
This file contains the RNN model for choreography classification
'''
import numpy as np
from keras import layers, Model
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
        print(self.input)
        self.output = n
        self.dropout_rate = model_parameters["dropout_rate"]
        self.lstm_neurons = model_parameters["lstm_neurons"]
        self.fc1_neuron = model_parameters["fc1_neurons"]
        self.fc2_neuron = model_parameters["fc2_neurons"]
        self.split = model_parameters["split"]
        self.epochs = model_parameters["epoch"]
        self.batch_size = model_parameters["batch_size"]

        self.model()
        self.run_plot()


    def model(self):
        '''
        RNN structure
        '''
        input = layers.Input(shape=self.input)
        X = layers.BatchNormalization()(input)
        X = layers.Bidirectional(layers.LSTM(self.lstm_neurons))(X)
        X = layers.Dropout(self.dropout_rate)(X)
        X = layers.Dense(self.fc1_neuron, activation='relu')(X)
        X = layers.Dense(self.fc2_neuron, activation='relu')(X)
        X = layers.Dense(self.output)(X)
        Y = layers.Activation('softmax')(X)
        self.model = Model(inputs=input, outputs=Y)
    
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
    dataset_path = "C:/Users/ray_s/Desktop/cs230_project/dataset/generated_dataset/image/"
    #dataset_path = "C:/Users/ray_s/Desktop/cs230_project/dataset/generated_dataset/image_sub/"

    # Load data and labels
    X = np.load(dataset_path + "data.npy")
    Y = np.load(dataset_path + "label.npy")

    print(X.shape)
    
    # Set model parameters
    lstm_neurons = 64
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
    #model = rnn_model(X, Y, model_parameters)