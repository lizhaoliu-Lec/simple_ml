import numpy as np

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

from simple_ml.utils.distance import euclidean
from simple_ml.preprocessing.general import Standardizer
from simple_ml.linear.model import LinearRegression, RidgeRegression
from simple_ml.nn.layer import Factorization, Input
from simple_ml.nn.model import Model
from simple_ml.nn.initializer import zeros, ones
from simple_ml.nn.regularizer import L2_Regularizer, L1_Regularizer
from simple_ml.nn.optimizer import SGD, Momentum, Adam, RMSProp
from simple_ml.utils.metric import accuracy, absolute_error, square_error

import matplotlib.pyplot as plt

# fix random seed
np.random.seed(1234)


def read_data(data_path='../tmp/exp5/ml-100k/u.data', val_split=0.1, test_split=0.2):
    # read the data
    triples = []
    with open(data_path) as f:
        for line in f.readlines():
            triples.append([int(i) for i in line.split('\t')[:-1]])
    Data = np.array(triples, dtype=np.int)
    total_size = Data.shape[0]
    train_size = int(total_size - (val_split + test_split) * total_size)
    val_size = int(train_size + val_split * total_size)
    Data_train = Data[:train_size]
    Data_val = Data[train_size:val_size]
    Data_test = Data[val_size:]

    def data_to_Xy(data):
        # start from index 0
        return data[:, :-1] - 1, np.reshape(data[:, -1], (-1, 1))

    X_train, y_train = data_to_Xy(Data_train)
    X_val, y_val = data_to_Xy(Data_val)
    X_test, y_test = data_to_Xy(Data_test)

    num_user = len(np.unique(Data[:, 0]))
    num_item = len(np.unique(Data[:, 1]))
    print('Number of unique user: %d' % num_user)
    print('Number of unique item: %d' % num_item)
    print('X_train: ', X_train.shape, ' y_train: ', y_train.shape)
    print('X_val: ', X_val.shape, ' y_val: ', y_val.shape)
    print('X_test: ', X_test.shape, ' y_test: ', y_train.shape)
    print()
    return X_train, y_train, X_val, y_val, X_test, y_test, num_user, num_item


def dlr():
    print(10 * '#' + ' SGD Factorization model ' + 10 * '#')

    # build the linear model with gradient descent
    # define layer
    X_train, y_train, X_val, y_val, X_test, y_test, num_user, num_item = read_data()
    Inputs = Input(input_shape=2)
    out = Factorization(a_dim=num_user, b_dim=num_item, k=10,
                        use_bias=False,
                        regularizer=L2_Regularizer(0.1))(Inputs)
    # out = Factorization(a_dim=num_user, b_dim=num_item, k=10)(Inputs)
    model = Model(Inputs, out)
    # model.compile('MSE', optimizer=Adam(lr=0.001))
    model.compile('HB', optimizer=Adam(lr=0.001))
    model.fit(X_train, y_train,
              verbose=10, epochs=300,
              validation_data=(X_val, y_val),
              batch_size=256, metric='MAE',
              shuffle=True,
              peek_type='single-reg')

    plt.plot(model.train_losses, label='$loss_{train}$')
    plt.plot(model.validation_losses, label='$loss_{val}$')
    plt.legend()
    plt.savefig('./loss.png', dpi=300)
    plt.show()
    plt.plot(model.train_metrics, label='$MAE_{train}$')
    plt.plot(model.validation_metrics, label='$MAE_{val}$')
    plt.legend()
    plt.savefig('./metric.png', dpi=300)
    plt.show()

    train_y_hat = model.forward(X_train)
    val_y_hat = model.forward(X_val)
    test_y_hat = model.forward(X_test)
    training_error = absolute_error(y_train, train_y_hat) / y_train.shape[0]
    val_error = absolute_error(y_val, val_y_hat) / y_val.shape[0]
    test_error = absolute_error(y_test, test_y_hat) / y_test.shape[0]

    print(model.best_performance(bigger=False))
    print('Training error: ', training_error)
    print('Val error: ', val_error)
    print('Test error: ', test_error)
    print(10 * '#' + ' SGD Factorization model end ' + 10 * '#')
    print()


if __name__ == '__main__':
    # read_data()
    dlr()
