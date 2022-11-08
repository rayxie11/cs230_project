import numpy as np
import matplotlib.pyplot as plt
import utils

def softmax(x):
    """
    Compute softmax function for a batch of input values. 
    The first dimension of the input corresponds to the batch size. The second dimension
    corresponds to every class in the output. When implementing softmax, you should be careful
    to only sum over the second dimension.

    Important Note: You must be careful to avoid overflow for this function. Functions
    like softmax have a tendency to overflow when very large numbers like e^10000 are computed.
    You will know that your function is overflow resistent when it can handle input like:
    np.array([[10000, 10010, 10]]) without issues.

    Args:
        x: A 2d numpy float array of shape batch_size x number_of_classes

    Returns:
        A 2d numpy float array containing the softmax results of shape batch_size x number_of_classes
    """
    # *** START CODE HERE ***
    return np.exp(x - np.max(x))\
           /np.expand_dims(np.sum(np.exp(x - np.max(x)), axis = 1), -1)
    # *** END CODE HERE ***

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Args:
        x: A numpy float array

    Returns:
        A numpy float array containing the sigmoid results
    """
    # *** START CODE HERE ***
    return 1/(1+np.exp(-x))
    # *** END CODE HERE ***

def relu(x):
    return (abs(x)+x)/2


def forward_prop_relu(data, labels, params):
    W1 = params.get("W1")
    b1 = params.get("b1")
    W2 = params.get("W2")
    b2 = params.get("b2")

    a = relu(data @ W1 + b1)
    z = a @ W2 + b2
    y_hat = softmax(z/1000)
    J = -1 / data.shape[0] * np.sum(labels * np.log(y_hat))

    return a, y_hat, J


def backward_prop_relu(data, labels, params, forward_prop_func, reg):
    W1 = params.get("W1")
    W2 = params.get("W2")
    a, y_hat, J = forward_prop_func(data, labels, params)

    dJ_dz = (y_hat - labels)/1000
    dJ_dW2 = (dJ_dz.T @ a).T / a.shape[0] + + 2*reg*W2
    dJ_db2 = np.mean(dJ_dz, axis=0)
    dJ_dW1 = ((dJ_dz @ W2.T * a / (a + 1e-5)).T @ data).T / a.shape[0] + + 2*reg*W1
    dJ_db1 = np.mean(dJ_dz @ W2.T * a / (a + 1e-5), axis=0)

    return {"W1": dJ_dW1, "b1": dJ_db1, "W2": dJ_dW2, "b2": dJ_db2}


def get_initial_params(input_size, num_hidden, num_output):
    """
    Compute the initial parameters for the neural network.

    This function should return a dictionary mapping parameter names to numpy arrays containing
    the initial values for those parameters.

    There should be four parameters for this model:
    W1 is the weight matrix for the hidden layer of size input_size x num_hidden
    b1 is the bias vector for the hidden layer of size num_hidden
    W2 is the weight matrix for the output layers of size num_hidden x num_output
    b2 is the bias vector for the output layer of size num_output

    As specified in the PDF, weight matrices should be initialized with a random normal distribution
    centered on zero and with scale 1.
    Bias vectors should be initialized with zero.
    
    Args:
        input_size: The size of the input data
        num_hidden: The number of hidden states
        num_output: The number of output classes
    
    Returns:
        A dict mapping parameter names to numpy arrays
    """

    # *** START CODE HERE ***
    W1 = np.random.normal(size = (input_size, num_hidden))
    b1 = np.zeros(num_hidden)
    W2 = np.random.normal(size = (num_hidden, num_output))
    b2 = np.zeros(num_output)

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    # *** END CODE HERE ***

def forward_prop(data, labels, params):
    """
    Implement the forward layer given the data, labels, and params.
    
    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network

    Returns:
        A 3 element tuple containing:
            1. A numpy array of the activations (after the sigmoid) of the hidden layer
            2. A numpy array The output (after the softmax) of the output layer
            3. The average loss for these data elements
    """
    # *** START CODE HERE ***
    W1 = params.get("W1")
    b1 = params.get("b1")
    W2 = params.get("W2")
    b2 = params.get("b2")

    a = sigmoid(data@W1/100 + b1)
    z = a@W2 + b2
    y_hat = softmax(z)
    J = -1/data.shape[0]*np.sum(labels*np.log(y_hat))

    return a, y_hat, J
    # *** END CODE HERE ***

def backward_prop(data, labels, params, forward_prop_func):
    """
    Implement the backward propegation gradient computation step for a neural network
    
    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.
        
        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    # *** START CODE HERE ***
    W2 = params.get("W2")
    a, y_hat, J = forward_prop_func(data, labels, params)

    dJ_dz = y_hat - labels
    dJ_dW2 = (dJ_dz.T@a).T/a.shape[0]
    dJ_db2 = np.mean(dJ_dz, axis = 0)
    dJ_dW1 = ((dJ_dz@W2.T*a*(1 - a)).T@data/100).T/a.shape[0]
    dJ_db1 = np.mean(dJ_dz@W2.T*a*(1 - a), axis = 0)

    return {"W1": dJ_dW1, "b1": dJ_db1, "W2": dJ_dW2, "b2": dJ_db2}
    # *** END CODE HERE ***


def backward_prop_regularized(data, labels, params, forward_prop_func, reg):
    """
    Implement the backward propegation gradient computation step for a neural network
    
    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above
        reg: The regularization strength (lambda)

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.
        
        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    # *** START CODE HERE ***
    W1 = params.get("W1")
    W2 = params.get("W2")
    a, y_hat, J = forward_prop_func(data, labels, params)

    dJ_dz = y_hat - labels
    dJ_dW2 = (dJ_dz.T @ a).T / a.shape[0] + 2*reg*W2
    dJ_db2 = np.mean(dJ_dz, axis=0)
    dJ_dW1 = ((dJ_dz @ W2.T * a * (1 - a)).T @ data/100).T / a.shape[0] + 2*reg*W1
    dJ_db1 = np.mean(dJ_dz @ W2.T * a * (1 - a), axis=0)

    return {"W1": dJ_dW1, "b1": dJ_db1, "W2": dJ_dW2, "b2": dJ_db2}
    # *** END CODE HERE ***

def gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size, params, forward_prop_func, backward_prop_func):
    """
    Perform one epoch of gradient descent on the given training data using the provided learning rate.

    This code should update the parameters stored in params.
    It should not return anything

    Args:
        train_data: A numpy array containing the training data
        train_labels: A numpy array containing the training labels
        learning_rate: The learning rate
        batch_size: The amount of items to process in each batch
        params: A dict of parameter names to parameter values that should be updated.
        forward_prop_func: A function that follows the forward_prop API
        backward_prop_func: A function that follows the backwards_prop API

    Returns: This function returns nothing.
    """

    # *** START CODE HERE ***
    for i in range(int(train_data.shape[0]/batch_size)):
        data = train_data[i*batch_size:(i+1)*batch_size, :]
        labels = train_labels[i*batch_size:(i+1)*batch_size, :]
        gradient = backward_prop_func(data, labels, params, forward_prop_func)

        W1 = params.get("W1")
        b1 = params.get("b1")
        W2 = params.get("W2")
        b2 = params.get("b2")

        params.update({"W1": W1 - learning_rate*gradient.get("W1")})
        params.update({"b1": b1 - learning_rate*gradient.get("b1")})
        params.update({"W2": W2 - learning_rate*gradient.get("W2")})
        params.update({"b2": b2 - learning_rate*gradient.get("b2")})
    # *** END CODE HERE ***

    # This function does not return anything
    return

def nn_train(
    train_data, train_labels, dev_data, dev_labels, test_data, test_labels,
    get_initial_params_func, forward_prop_func, backward_prop_func,
    num_hidden=300, learning_rate=5, num_epochs=30, batch_size=1000):

    (nexp, dim) = train_data.shape

    params = get_initial_params_func(dim, num_hidden, 11)

    cost_train = []
    cost_dev = []
    cost_test = []
    accuracy_train = []
    accuracy_dev = []
    accuracy_test = []
    for epoch in range(num_epochs):
        gradient_descent_epoch(train_data, train_labels, 
            learning_rate, batch_size, params, forward_prop_func, backward_prop_func)

        h, output, cost = forward_prop_func(train_data, train_labels, params)
        cost_train.append(cost)
        accuracy_train.append(compute_accuracy(output,train_labels))
        h, output, cost = forward_prop_func(dev_data, dev_labels, params)
        cost_dev.append(cost)
        accuracy_dev.append(compute_accuracy(output, dev_labels))
        h, output, cost = forward_prop_func(test_data, test_labels, params)
        cost_test.append(cost)
        accuracy_test.append(compute_accuracy(output, test_labels))

    return params, cost_train, cost_dev, accuracy_train, accuracy_dev

def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) == 
        np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def run_train_test(name, all_data, all_labels, forward_prop_func, backward_prop_func, num_epochs, plot=True):
    params, cost_train, cost_dev, accuracy_train, accuracy_dev = nn_train(
        all_data['train'], all_labels['train'], 
        all_data['dev'], all_labels['dev'],
        all_data['test'], all_labels['test'],
        get_initial_params, forward_prop_func, backward_prop_func,
        num_hidden=300, learning_rate=2, num_epochs=num_epochs, batch_size=250
    )

    t = np.arange(num_epochs)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)

        ax1.plot(t, cost_train,'r', label='train')
        ax1.plot(t, cost_dev, 'b', label='test')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss')
        if name == 'non_reg':
            ax1.set_title('non-reg softmax')
        elif name == 'relu':
            ax1.set_title('relu')
        else:
            ax1.set_title('softmax')
        ax1.legend()

        ax2.plot(t, accuracy_train,'r', label='train')
        ax2.plot(t, accuracy_dev, 'b', label='test')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()

        fig.savefig('./' + name + '.pdf')

    accuracy = nn_test(all_data['test'], all_labels['test'], params)
    print('For model %s, got accuracy: %f' % (name, accuracy))
    
    return accuracy

def main(plot=True):
    num_epochs = 30
    model = 2

    file, label = utils.onehot_labels('annotation_dict.json')
    data = np.load('img.npy')

    train_data = data[2500:15000, :]
    train_labels = label[2500:15000, :]

    dev_data = data[0:2500, :]
    dev_labels = label[0:2500, :]

    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std
    dev_data = (dev_data - mean) / std

    test_data = data[15000:16000, :]
    test_labels = label[15000:16000, :]
    test_data = (test_data - mean) / std

    all_data = {
        'train': train_data,
        'dev': dev_data,
        'test': test_data
    }

    all_labels = {
        'train': train_labels,
        'dev': dev_labels,
        'test': test_labels,
    }

    if model == 1:
        non_reg_acc = run_train_test('non_reg', all_data, all_labels, forward_prop, backward_prop, num_epochs, plot)
        return non_reg_acc
    elif model == 2:
        reg_acc = run_train_test('regularized', all_data, all_labels, forward_prop,
            lambda a, b, c, d: backward_prop_regularized(a, b, c, d, reg=0.0001),
            num_epochs, plot)
        return reg_acc
    elif model == 3:
        relu_acc = run_train_test('relu', all_data, all_labels, forward_prop_relu,
           lambda a, b, c, d: backward_prop_relu(a, b, c, d, reg=0.0001),
           num_epochs, plot)
        return relu_acc

if __name__ == '__main__':
    main()
