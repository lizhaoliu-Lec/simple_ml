import numpy as np

from simple_ml.nn import Model, Sequential
from simple_ml.nn import Dense, Softmax, Input, Dropout, Activation
from simple_ml.nn import SGD, Momentum


def mlp_random():
    """test MLP with random data and Sequential

    """
    input_size = 600
    input_dim = 20
    label_size = 10
    train_X = np.random.random((input_size, input_dim))
    train_y = np.zeros((input_size, label_size))
    for _ in range(input_size):
        train_y[_, np.random.randint(0, label_size)] = 1

    model = Sequential()
    model.add(Input(input_shape=input_dim))
    model.add(Dense(100, activation='relu'))
    model.add(Softmax(label_size))
    model.compile('CE')
    model.fit(train_X, train_y, verbose=100, epochs=5000)


def mlp_mnist():
    """test MLP with MNIST data and Sequential

    """
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('./tmp/data', one_hot=True)
    training_data = np.array([image.flatten() for image in mnist.train.images])
    training_label = mnist.train.labels
    valid_data = np.array([image.flatten() for image in mnist.validation.images])
    valid_label = mnist.validation.labels
    input_dim = training_data.shape[1]
    label_size = training_label.shape[1]

    model = Sequential()
    model.add(Input(input_shape=(input_dim,)))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Softmax(label_size))
    model.compile('CE', optimizer=SGD())
    model.fit(training_data, training_label, validation_data=(valid_data, valid_label))


def model_mlp_random():
    """test MLP with random data and Model

    """
    input_size = 600
    input_dim = 20
    label_size = 10
    train_X = np.random.random((input_size, input_dim))
    train_y = np.zeros((input_size, label_size))
    for _ in range(input_size):
        train_y[_, np.random.randint(0, label_size)] = 1

    input = Input(input_shape=input_dim)
    d1 = Dense(100, activation='relu')(input)
    s1 = Softmax(label_size)(d1)
    model = Model(input, s1)
    model.compile('CE')
    model.fit(train_X, train_y, verbose=100, epochs=5000)


def model_mlp_mnist():
    """test MLP with MNIST data and Model

    """
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('./tmp/data', one_hot=True)
    training_data = np.array([image.flatten() for image in mnist.train.images])
    training_label = mnist.train.labels
    valid_data = np.array([image.flatten() for image in mnist.validation.images])
    valid_label = mnist.validation.labels
    input_dim = training_data.shape[1]
    label_size = training_label.shape[1]

    dense_1 = Dense(300, input_dim=input_dim, activation=None)
    dense_2 = Activation('relu')(dense_1)
    dropout_1 = Dropout(0.2)(dense_2)
    softmax_1 = Softmax(label_size)(dropout_1)
    model = Model(dense_1, softmax_1)
    model.compile('CE', optimizer=Momentum())
    model.fit(training_data, training_label, validation_data=(valid_data, valid_label))


if __name__ == '__main__':
    # mlp_random()
    # mlp_mnist()
    model_mlp_random()
    # model_mlp_mnist()
