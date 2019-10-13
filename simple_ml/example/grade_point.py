from pprint import pprint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from simple_ml.utils.distance import euclidean
from simple_ml.preprocessing.general import Standardizer
from simple_ml.linear.model import LinearRegression, RidgeRegression
from simple_ml.nn.layer import Input, Linear
from simple_ml.nn.model import Model
from simple_ml.nn.initializer import zeros, ones
from simple_ml.nn.optimizer import SGD, Momentum, Adam, RMSProp
from simple_ml.utils.metric import accuracy, mean_absolute_error, mean_square_error


def read_csv_to_xy(filepath, x_cols, y_col):
    data = pd.read_csv(filepath)
    data = data.apply(lambda x: x.fillna(x.mean()), axis=0)
    x = np.array(data[x_cols])
    y = np.array(data[y_col])
    return x, y


def read_csv_col_names(filepath):
    data = pd.read_csv(filepath)
    return [n for n in data]


TRAIN_PATH = './tmp/data/TrainSet.csv'
TEST_PATH = './tmp/data/TestSet.csv'

train_col_names = read_csv_col_names(TRAIN_PATH)
test_col_names = read_csv_col_names(TEST_PATH)
assert train_col_names == test_col_names
print('The col names are:')
pprint(train_col_names)

x_cols = train_col_names
# predict the grade point setup
x_cols.remove('Machine learning')
y_col = ['Machine learning grade point']

# predict the score setup
# x_cols.remove('Machine learning grade point')
# y_col = ['Machine learning']

# read the data
train_x, train_y = read_csv_to_xy(TRAIN_PATH, x_cols, y_col)
test_x, test_y = read_csv_to_xy(TEST_PATH, x_cols, y_col)

print('Number of training example: %d' % train_x.shape[0])
print('Number of test example: %d' % test_x.shape[0])
print('train_x shape: ', train_x.shape, ' train_y shape: ', train_y.shape)
print('test_x shape: ', test_x.shape, ' test_y shape: ', test_y.shape)
print()


def lr():
    # build the linear model
    print(10 * '#' + ' LinearRegression model ' + 10 * '#')
    linear_regression = LinearRegression()
    # train it
    linear_regression.fit(train_x, train_y)
    test_y_hat = linear_regression.predict(test_x)
    train_y_hat = linear_regression.predict(train_x)

    training_error = euclidean(train_y, train_y_hat)
    test_error = euclidean(test_y, test_y_hat)

    print('Training error: ', training_error)
    print('Test error: ', test_error)
    print(10 * '#' + ' LinearRegression model end' + 10 * '#')
    print()


def rlr():
    # build the ridge linear model
    print(10 * '#' + ' RidgeRegression model ' + 10 * '#')
    ridge_regression = RidgeRegression()
    # train it
    ridge_regression.fit(train_x, train_y)
    test_y_hat = ridge_regression.predict(test_x)
    train_y_hat = ridge_regression.predict(train_x)

    training_error = euclidean(train_y, train_y_hat)
    test_error = euclidean(test_y, test_y_hat)

    print('Training error: ', training_error)
    print('Test error: ', test_error)
    print(10 * '#' + ' RidgeRegression model end' + 10 * '#')
    print()


def dlr():
    print(10 * '#' + ' Linear model ' + 10 * '#')
    # build the linear model with gradient descent
    # define layer
    input_dim = train_x.shape[1]
    Inputs = Input(input_shape=input_dim)
    X = Linear(output_dim=1, activation=None, initializer=ones)(Inputs)
    model = Model(Inputs, X)
    # 0.01 for grade point prediction, use MSE is a lot better than MAE
    # model.compile('MSE', optimizer=SGD(lr=0.001))
    model.compile('HB', optimizer=SGD(lr=0.001))
    # 0.1 for score prediction, use MAE is slightly better than MSE
    # model.compile('MAE', optimizer=SGD(lr=0.01))
    # model.compile('HB', optimizer=SGD(lr=0.01))
    # or we can use HB (huber loss) for both two #
    model.fit(train_x, train_y,
              verbose=1000, epochs=10000,
              validation_data=(test_x, test_y),
              batch_size=128, metric=mean_square_error,
              peek_type='single-reg')
    print(10 * '#' + ' Linear model end ' + 10 * '#')
    print()


if __name__ == '__main__':
    # normalized the data
    standardizer = Standardizer()
    standardizer.fit(train_x)
    train_x = standardizer.transform(train_x)
    test_x = standardizer.transform(test_x)

    # lr()
    # rlr()
    dlr()
