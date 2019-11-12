import numpy as np

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

from simple_ml.utils.distance import euclidean
from simple_ml.preprocessing.general import Standardizer
from simple_ml.linear.model import LinearRegression, RidgeRegression
from simple_ml.nn.layer import Input, Linear, Dropout
from simple_ml.nn.model import Model
from simple_ml.nn.initializer import zeros, ones
from simple_ml.nn.optimizer import SGD, Momentum, Adam, RMSProp
from simple_ml.utils.metric import accuracy, absolute_error, square_error

import matplotlib.pyplot as plt

# fix random seed
np.random.seed(1234)

# read the data
data_path = './tmp/exp1/housing.scale'
X, y = load_svmlight_file(data_path)
# convert sparse matrix to numpy array
X = X.A
y = np.reshape(y, (-1, 1))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

print('X_train: ', X_train.shape)
print('X_test: ', X_test.shape)
print('y_train: ', y_train.shape)
print('y_test: ', y_test.shape)


def lr():
    # build the linear model
    print(10 * '#' + ' LinearRegression model ' + 10 * '#')
    linear_regression = LinearRegression()
    # train it
    linear_regression.fit(X_train, y_train)
    test_y_hat = linear_regression.predict(X_test)
    train_y_hat = linear_regression.predict(X_train)

    training_error = absolute_error(y_train, train_y_hat) / y_train.shape[0]
    test_error = absolute_error(y_test, test_y_hat) / y_test.shape[0]

    print('Training error: ', training_error)
    print('Test error: ', test_error)
    # for yp, yt in zip(test_y_hat, y_test):
    #     print(yp, yt)
    print(10 * '#' + ' LinearRegression model end ' + 10 * '#')
    print()


def rlr():
    # build the linear model
    print(10 * '#' + ' Ridge LinearRegression model ' + 10 * '#')
    ridge_regression = RidgeRegression()
    # train it
    ridge_regression.fit(X_train, y_train)
    test_y_hat = ridge_regression.predict(X_test)
    train_y_hat = ridge_regression.predict(X_train)

    training_error = absolute_error(y_train, train_y_hat) / y_train.shape[0]
    test_error = absolute_error(y_test, test_y_hat) / y_test.shape[0]

    print('Training error: ', training_error)
    print('Test error: ', test_error)
    # for yp, yt in zip(test_y_hat, y_test):
    #     print(yp, yt)
    print(10 * '#' + ' Ridge LinearRegression model end ' + 10 * '#')
    print()


def dlr():
    print(10 * '#' + ' SGD Linear model ' + 10 * '#')

    # build the linear model with gradient descent
    # define layer
    Inputs = Input(input_shape=X_train.shape[1])
    linear_out = Linear(output_dim=1, activation=None, initializer=ones)(Inputs)
    model = Model(Inputs, linear_out)
    model.compile('MSE', optimizer=SGD(lr=0.01))
    model.fit(X_train, y_train,
              verbose=-1, epochs=5000,
              validation_data=(X_test, y_test),
              batch_size=256, metric='MAE',
              shuffle=True,
              # peek_type='single-reg'
              )

    plt.subplot(211)
    plt.plot(model.train_losses, label='train_losses')
    plt.plot(model.validation_losses, label='valid_losses')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplot(212)
    plt.plot(model.train_metrics, label='train_metrics')
    plt.plot(model.validation_metrics, label='valid_metrics')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()

    train_y_hat = model.forward(X_train)
    test_y_hat = model.forward(X_test)
    training_error = absolute_error(y_train, train_y_hat) / y_train.shape[0]
    test_error = absolute_error(y_test, test_y_hat) / y_test.shape[0]

    print('Training error: ', training_error)
    print('Test error: ', test_error)
    print(10 * '#' + ' SGD Linear model end ' + 10 * '#')
    print()


def dmlr():
    print(10 * '#' + ' SGD Deep Linear model ' + 10 * '#')

    # build the linear model with gradient descent
    # define layer
    Inputs = Input(input_shape=X_train.shape[1])
    linear_out = Linear(output_dim=64, activation='swish')(Inputs)
    linear_out = Linear(output_dim=128, activation='swish')(linear_out)
    linear_out = Linear(output_dim=256, activation='swish')(linear_out)
    linear_out = Linear(output_dim=1, activation=None)(linear_out)
    model = Model(Inputs, linear_out)
    model.compile('MSE', optimizer=Momentum(lr=0.0001))
    model.fit(X_train, y_train,
              verbose=100, epochs=500,
              validation_data=(X_test, y_test),
              batch_size=256, metric='MAE',
              shuffle=True,
              # peek_type='single-reg'
              )
    # y_pred = model.forward(X_test)
    # for yp, yt in zip(y_pred, y_test):
    #     print(yp, yt)
    plt.subplot(211)
    plt.plot(model.train_losses, label='train_losses')
    plt.plot(model.validation_losses, label='valid_losses')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplot(212)
    plt.plot(model.train_metrics, label='train_metrics')
    plt.plot(model.validation_metrics, label='valid_metrics')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    print(10 * '#' + ' SGD Deep Linear model end ' + 10 * '#')
    print()


if __name__ == '__main__':
    # normalized the data
    # standardizer = Standardizer()
    # standardizer.fit(X_train)
    # X_train = standardizer.transform(X_train)
    # X_test = standardizer.transform(X_test)

    lr()
    rlr()
    dlr()
    # dmlr()
