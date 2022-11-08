import numpy as np
from keras import layers, Model
import matplotlib.pyplot as plt

class rnn_model():
    def __init__(self, num_neurons, num_densor):
        joint_input = layers.Input(shape=(16, 36))
        X = layers.BatchNormalization()(joint_input)
        X = layers.Bidirectional(layers.LSTM(num_neurons))(X)
        X = layers.Dropout(0.2)(X)
        X = layers.Dense(num_densor, activation='relu')(X)
        X = layers.Dense(num_densor, activation='relu')(X)
        X = layers.Dense(11)(X)
        Y = layers.Activation('softmax')(X)
        self.model = Model(inputs=joint_input, outputs=Y)

    def train_test(self, X, Y, epochs, batch_size):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = self.model.fit(X, Y, validation_split=0.33, epochs=epochs, batch_size=batch_size)
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
    X = np.load('motion_fill_interpolate.npy')
    Y = np.load('label.npy')
    model = rnn_model(30, 11)
    model.train_test(X, Y, 70, 128)
