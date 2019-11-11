import numpy as np

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

from simple_ml.utils.distance import euclidean
from simple_ml.preprocessing.general import Standardizer
from simple_ml.nn.layer import Input, Linear, Dropout, Softmax
from simple_ml.nn.model import Model
from simple_ml.nn.initializer import zeros, ones, default_weight_initializer
from simple_ml.nn.regularizer import L2_Regularizer, L1_Regularizer, L1L2_Regularizer
from simple_ml.nn.optimizer import SGD, Momentum, Adam, RMSProp
from simple_ml.utils.metric import accuracy, absolute_error, square_error

import matplotlib.pyplot as plt


def convert_to_onehot(x):
    return np.array(x == 1, dtype=np.float)


# read the data
data_path = './tmp/exp2/a9a'
X_train, y_train = load_svmlight_file(data_path)
X_test, y_test = load_svmlight_file(data_path + '.t', n_features=123)
X_train, X_test = X_train.A, X_test.A
y_train, y_test = np.reshape(y_train, (-1, 1)), np.reshape(y_test, (-1, 1))
# print(y_train)
print(type(X_train), type(y_train))
print(type(X_test), type(y_test))
# X = X.A
# y = np.reshape(y, (-1, 1))
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.33, random_state=42)

print('X_train: ', X_train.shape)
print('X_test: ', X_test.shape)
print('y_train: ', y_train.shape)
print('y_test: ', y_test.shape)


def logistic_cls():
    print(10 * '#' + ' SGD Linear model ' + 10 * '#')

    # build the linear model with gradient descent
    # define layer
    Inputs = Input(input_shape=X_train.shape[1])
    # X = Linear(output_dim=64,
    #            regularizer=L2_Regularizer(1),
    #            # regularizer=L1_Regularizer(1e-2),
    #            # regularizer=L1L2_Regularizer(l2=1),
    #            activation='swish')(Inputs)
    # X = Linear(output_dim=128,
    #            regularizer=L2_Regularizer(1),
    #            # regularizer=L1_Regularizer(1e-2),
    #            # regularizer=L1L2_Regularizer(l2=1),
    #            activation='swish')(X)
    # X = Linear(output_dim=256,
    #            regularizer=L2_Regularizer(1),
    #            # regularizer=L1_Regularizer(1e-2),
    #            # regularizer=L1L2_Regularizer(l2=1),
    #            activation='swish')(X)
    X = Linear(output_dim=1,
               # regularizer=L2_Regularizer(1),
               # regularizer=L1_Regularizer(1e-2),
               # regularizer=L1L2_Regularizer(l2=1),
               activation='sigmoid')(Inputs)
    model = Model(Inputs, X)
    model.compile('BCE', optimizer=Adam(lr=0.001))
    model.fit(X_train, y_train,
              verbose=10, epochs=100,
              validation_data=(X_test, y_test),
              batch_size=128, metric='Binary_Accuracy',
              shuffle=True,
              peek_type='single-logistic-cls'
              )
    plt.subplot(211)
    plt.plot(model.train_losses, label='train_losses')
    plt.plot(model.validation_losses, label='valid_losses')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplot(212)
    plt.plot(model.train_metrics, label='train_accuracy')
    plt.plot(model.validation_metrics, label='valid_accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    print(10 * '#' + ' SGD Linear model end ' + 10 * '#')
    print()


def svm():
    print(10 * '#' + ' SGD SVM model ' + 10 * '#')

    # build the linear model with gradient descent
    # define layer
    Inputs = Input(input_shape=X_train.shape[1])
    # X = Linear(output_dim=64,
    #            regularizer=L2_Regularizer(1e-5),
    #            # regularizer=L1_Regularizer(1e-2),
    #            # regularizer=L1L2_Regularizer(l2=1),
    #            activation='swish')(Inputs)
    # X = Linear(output_dim=128,
    #            regularizer=L2_Regularizer(1e-5),
    #            # regularizer=L1_Regularizer(1e-2),
    #            # regularizer=L1L2_Regularizer(l2=1),
    #            activation='swish')(X)
    # X = Linear(output_dim=256,
    #            regularizer=L2_Regularizer(1e-5),
    #            # regularizer=L1_Regularizer(1e-2),
    #            # regularizer=L1L2_Regularizer(l2=1),
    #            activation='swish')(X)
    X = Linear(output_dim=1,
               # initializer='default_weight_initializer',
               initializer=default_weight_initializer,
               regularizer=L2_Regularizer(1e-5),
               # regularizer=L1_Regularizer(1e-2),
               # regularizer=L1L2_Regularizer(l2=1),
               activation=None)(Inputs)
    model = Model(Inputs, X)
    # model.compile('HL', optimizer=SGD(lr=0.001))
    model.compile('HL', optimizer=Adam(lr=0.001))
    model.fit(X_train, y_train,
              verbose=10, epochs=100,
              validation_data=(X_test, y_test),
              batch_size=64, metric='svm_binary_accuracy',
              shuffle=True,
              peek_type='single-svm-cls'
              )
    plt.subplot(211)
    plt.plot(model.train_losses, label='train_losses')
    plt.plot(model.validation_losses, label='valid_losses')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplot(212)
    plt.plot(model.train_metrics, label='train_accuracy')
    plt.plot(model.validation_metrics, label='valid_accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    print(10 * '#' + ' SGD SVM model end ' + 10 * '#')
    print()


if __name__ == '__main__':
    # normalized the data
    standardizer = Standardizer()
    standardizer.fit(X_train)
    X_train = standardizer.transform(X_train)
    X_test = standardizer.transform(X_test)

    # print('train set positive class portion: %.2f (%d / %d)' % (np.mean(y_train), int(np.sum(y_train)), y_train.shape[0]))
    # print('test set positive class portion: %.2f (%d / %d)' % (np.mean(y_test), int(np.sum(y_test)), y_test.shape[0]))
    # y_train = convert_to_onehot(y_train)
    # y_test = convert_to_onehot(y_test)
    # logistic_cls()

    svm()
