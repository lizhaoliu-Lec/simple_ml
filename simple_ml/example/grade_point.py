from pprint import pprint
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from simple_ml.utils.distance import euclidean
from simple_ml.preprocessing.general import Standardizer
from simple_ml.linear.model import LinearRegression, RidgeRegression
from simple_ml.nn.layer import Input, Linear
from simple_ml.nn.model import Model
from simple_ml.nn.initializer import zeros, ones
from simple_ml.nn.regularizer import L2_Regularizer
from simple_ml.nn.optimizer import SGD, Momentum, Adam, RMSProp
from simple_ml.utils.metric import accuracy, mean_absolute_error, mean_square_error


def read_csv_to_xy(filepath, x_cols, y_col):
    data = pd.read_csv(filepath)
    data = data.apply(lambda x: x.fillna(x.mean()), axis=0)
    # data = data.apply(lambda x: x.fillna(0), axis=0)
    # data = data.apply(lambda x: x.fillna(random.randint(0, 100)), axis=0)
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

x_cols = train_col_names

x_cols.remove('Machine learning')
x_cols.remove('Machine learning grade point')

# predict the grade point setup
# y_col = ['Machine learning grade point']
# predict the score setup
y_col = ['Machine learning']

print('The feature col names are:')
pprint(train_col_names)

print('The target col names is:')
pprint(y_col)

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

    training_error = mean_absolute_error(train_y, train_y_hat) / train_y.shape[0]
    test_error = mean_absolute_error(test_y, test_y_hat) / test_y.shape[0]

    print('Training error: ', training_error)
    print('Test error: ', test_error)
    print(10 * '#' + ' LinearRegression model end' + 10 * '#')
    print()
    return linear_regression.beta


def rlr():
    # build the ridge linear model
    print(10 * '#' + ' RidgeRegression model ' + 10 * '#')
    ridge_regression = RidgeRegression()
    # train it
    ridge_regression.fit(train_x, train_y)
    test_y_hat = ridge_regression.predict(test_x)
    train_y_hat = ridge_regression.predict(train_x)

    training_error = mean_absolute_error(train_y, train_y_hat) / train_y.shape[0]
    test_error = mean_absolute_error(test_y, test_y_hat) / test_y.shape[0]

    print('Training error: ', training_error)
    print('Test error: ', test_error)
    print(10 * '#' + ' RidgeRegression model end' + 10 * '#')
    print()
    return ridge_regression.beta


def dlr():
    print(10 * '#' + ' Linear model ' + 10 * '#')
    # build the linear model with gradient descent
    # define layer
    input_dim = train_x.shape[1]
    Inputs = Input(input_shape=input_dim)
    X = Linear(output_dim=1, activation=None,
               regularizer=L2_Regularizer(1),
               initializer=ones)(Inputs)
    model = Model(Inputs, X)

    # lr = 0.001 for grade point prediction, use MSE is a lot better than MAE
    # model.compile('MSE', optimizer=SGD(lr=0.001))

    # lr = 0.01 for score prediction, use MAE is slightly better than MSE
    model.compile('MAE', optimizer=SGD(lr=0.01))

    # or we can use HB (huber loss) for both two
    # but remember lr = 0.001 for grade point prediction
    # and lr = 0.01 for score prediction
    # model.compile('HB', optimizer=SGD(lr=0.01))

    model.fit(train_x, train_y,
              verbose=500, epochs=10000,
              validation_data=(test_x, test_y),
              batch_size=16, metric='mae',
              peek_type='single-reg')
    plt.subplot(211)
    plt.plot(model.train_losses, label='train_losses')
    plt.plot(model.validation_losses, label='valid_losses')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplot(212)
    plt.plot(model.train_metrics, label='train_MAE')
    plt.plot(model.validation_metrics, label='valid_MAE')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    train_y_hat = model.forward(train_x)
    test_y_hat = model.forward(test_x)
    training_error = mean_absolute_error(train_y, train_y_hat) / train_y.shape[0]
    test_error = mean_absolute_error(test_y, test_y_hat) / test_y.shape[0]

    print('Training error: ', training_error)
    print('Test error: ', test_error)
    print(10 * '#' + ' Linear model end ' + 10 * '#')
    print()
    return X.params


if __name__ == '__main__':
    # normalized the data
    standardizer = Standardizer()
    standardizer.fit(train_x)
    train_x = standardizer.transform(train_x)
    test_x = standardizer.transform(test_x)

    dlrw = dlr()
    dlrw = np.concatenate([dlrw[1], dlrw[0].squeeze()])

    lrw = lr()
    lrw = np.array(lrw).squeeze()

    rlrw = rlr()
    rlrw = np.array(rlrw).squeeze()

    print('dlrw shape: ', dlrw.shape)
    print('lrw shape: ', lrw.shape)
    print('rlrw shape: ', rlrw.shape)

    print('distance: lrw <-> rlrw', np.linalg.norm(lrw - rlrw, 2))
    print('distance: lrw <-> dlrw', np.linalg.norm(lrw - dlrw, 2))
    print('distance: dlrw <-> rlrw', np.linalg.norm(dlrw - rlrw, 2))

    # for dw, lw, rw in zip(dlrw, lrw, rlrw):
    #     print('dw: %2.4f | lw: %2.4f | rw: %2.4f' % (dw, lw, rw))

    # save old weight for negative saving
    old_dlrw = dlrw
    old_lrw = lrw
    old_rlrw = rlrw

    # only the magtitude matters
    dlrw = np.abs(dlrw)[1:]
    lrw = np.abs(lrw)[1:]
    rlrw = np.abs(rlrw)[1:]

    # sort it in reverse order
    d_idx = np.argsort(-dlrw)
    l_idx = np.argsort(-lrw)
    r_idx = np.argsort(-rlrw)

    # print out the importance of each feature
    for i in [('d', d_idx, old_dlrw[1:]), ('l', l_idx, old_lrw[1:]), ('r', r_idx, old_rlrw[1:])]:
        print(i[0], {x_cols[k]: round(i[2][k], 2) for k in i[1].tolist()})
