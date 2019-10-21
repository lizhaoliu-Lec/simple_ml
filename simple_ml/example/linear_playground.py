import numpy as np

from simple_ml.nn.model import Model, Sequential
from simple_ml.nn.layer import Dense, Softmax, Input, Dropout, Activation, MaxPooling2D, AvgPooling2D, Flatten
from simple_ml.nn.layer import FastConv2d as Conv2d
# from simple_ml.nn.layer import Conv2d
from simple_ml.nn.optimizer import SGD, Momentum, Adam
from simple_ml.nn.initializer import zeros
from simple_ml.utils.metric import accuracy, mean_absolute_error

import matplotlib.pyplot as plt


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
              metric='Accuracy', peek_type='single-cls')


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
              verbose=100, epochs=600, batch_size=256,
              # validation_split=0.1,
              metric='MAE', peek_type='single-reg')
    print(len(model.train_losses))
    print(len(model.validation_losses))
    print(len(model.train_metrics))
    print(len(model.validation_metrics))
    plt.axis([0, len(model.train_losses), 0, 5])
    plt.plot(model.train_losses)
    plt.plot(model.validation_losses)
    # plt.plot(model.train_metrics)
    # plt.plot(model.validation_metrics)
    plt.show()


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
              metric='Accuracy', peek_type='single-cls')


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
              metric='Accuracy', peek_type='single-cls')


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
              metric='Accuracy', peek_type='single-cls')


def seq_cnn_mnist():
    """test CNN with MNIST data and Sequential

    """
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('/tmp/data', one_hot=False)
    training_data = np.array([image.reshape(28, 28, 1) for image in mnist.train.images])
    training_label = mnist.train.labels
    valid_data = np.array([image.reshape(28, 28, 1) for image in mnist.validation.images])
    valid_label = mnist.validation.labels
    label_size = 10

    model = Sequential()
    model.add(Input(batch_input_shape=(None, 28, 28, 1)))
    model.add(Conv2d(3, 16, stride=2, padding=1, activation='relu'))
    # model.add(AvgPooling2D((2, 2), stride=1))
    model.add(Conv2d(3, 32, stride=2, padding=1, activation='relu'))
    # model.add(AvgPooling2D((2, 2), stride=1))
    model.add(Flatten())
    model.add(Softmax(label_size))
    model.compile('CE', optimizer=SGD(lr=1e-3))
    # model.fit(training_data, training_label, validation_data=(valid_data, valid_label),
    #           batch_size=256, verbose=1, epochs=40, metric='Accuracy', peek_type='single-cls')
    # model.fit(training_data[:1000], training_label[:1000], validation_data=(valid_data[:1000], valid_label[:1000]),
    #           batch_size=256, verbose=10, epochs=100, metric='Accuracy', peek_type='single-cls')
    model.fit(training_data[:100], training_label[:100], validation_data=(valid_data[:50], valid_label[:50]),
              batch_size=256, verbose=10, epochs=40, metric='Accuracy', peek_type='single-cls')
    plt.subplot(211)
    plt.plot(model.train_losses, label='train_losses')
    plt.plot(model.validation_losses, label='valid_losses')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplot(212)
    plt.plot(model.train_metrics, label='train_accuracy')
    plt.plot(model.validation_metrics, label='valid_accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()


if __name__ == '__main__':
    # seq_mlp_random_cls()
    # model_mlp_random_reg()
    # seq_mlp_mnist()
    # model_mlp_random_cls()
    # model_mlp_mnist()
    seq_cnn_mnist()
