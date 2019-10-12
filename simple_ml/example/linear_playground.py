import numpy as np

from simple_ml.nn.model import Model, Sequential
from simple_ml.nn.layer import Dense, Softmax, Input, Dropout, Activation
from simple_ml.nn.optimizer import SGD, Momentum, Adam
from simple_ml.nn.initializer import zeros
from simple_ml.utils.metric import accuracy, mean_absolute_error


def seq_mlp_random_cls():
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
    model.fit(train_X, train_y, verbose=100, epochs=5000,
              validation_split=0.2,
              metric=accuracy, peek_type='single-cls')


def model_mlp_random_reg():
    """test MLP with random data and Sequential

    """
    input_size = 600
    input_dim = 20
    output_dim = 1
    train_X = np.random.random((input_size, input_dim))
    random_weight = np.random.random((input_dim, output_dim))
    random_noise = np.random.random((input_size, output_dim))
    train_y = np.dot(train_X, random_weight) + random_noise

    Inputs = Input(input_shape=input_dim)
    X = Dense(100, activation='relu')(Inputs)
    X = Dense(100, activation='relu')(X)
    X = Dense(output_dim, activation=None)(X)
    model = Model(Inputs, X)
    model.compile('MSE', optimizer='momentum')
    model.fit(train_X, train_y,
              verbose=100, epochs=1000, batch_size=256,
              validation_split=0.1,
              metric=mean_absolute_error, peek_type='single-reg')


def seq_mlp_mnist():
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
    model.add(Dropout(0.5))
    model.add(Softmax(label_size))
    model.compile('CE', optimizer=SGD())
    model.fit(training_data, training_label,
              validation_data=(valid_data, valid_label),
              metric=accuracy, peek_type='single-cls')


def model_mlp_random_cls():
    """test MLP with random data and Model

    """
    input_size = 600
    input_dim = 20
    label_size = 10
    train_X = np.random.random((input_size, input_dim))
    train_y = np.zeros((input_size, label_size))
    for _ in range(input_size):
        train_y[_, np.random.randint(0, label_size)] = 1

    Inputs = Input(input_shape=input_dim)
    X = Dense(100, activation='relu')(Inputs)
    X = Softmax(label_size)(X)
    model = Model(Inputs, X)
    model.compile('CE')
    model.fit(train_X, train_y,
              verbose=100, epochs=5000,
              metric=accuracy, peek_type='single-cls')


def model_mlp_mnist():
    """test MLP with MNIST data and Model

    """
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('./tmp/data', one_hot=False)
    training_data = np.array([image.flatten() for image in mnist.train.images])
    training_label = mnist.train.labels
    valid_data = np.array([image.flatten() for image in mnist.validation.images])
    valid_label = mnist.validation.labels
    input_dim = training_data.shape[1]
    label_size = 10

    Inputs = Dense(300, input_dim=input_dim, activation=None)
    # X = Activation('relu')(Inputs)
    X = Activation('relu6')(Inputs)
    X = Dropout(0.2)(X)
    X = Softmax(label_size)(X)
    model = Model(Inputs, X)
    model.compile('CE', optimizer='Adadelta')
    # model.compile('CE', optimizer=Momentum(nesterov=True))
    model.fit(training_data, training_label,
              validation_data=(valid_data, valid_label),
              metric=accuracy, peek_type='single-cls')


if __name__ == '__main__':
    # seq_mlp_random_cls()
    # model_mlp_random_reg()
    # seq_mlp_mnist()
    # model_mlp_random_cls()
    model_mlp_mnist()
