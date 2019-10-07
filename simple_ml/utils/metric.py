import numpy as np

__all__ = ['accuracy', 'mean_absolute_error', 'mean_square_error']


def accuracy(outputs, targets):
    y_predicts = np.argmax(outputs, axis=1)
    y_targets = np.argmax(targets, axis=1)
    acc = y_predicts == y_targets
    return np.sum(acc, axis=0)


def mean_square_error(outputs, targets):
    batch_size = outputs.shape[0]
    outputs = np.reshape(outputs, (batch_size, -1))
    targets = np.reshape(targets, (batch_size, -1))

    return np.sum(np.mean(0.5 * (outputs - targets) ** 2, axis=1), axis=0)


def mean_absolute_error(outputs, targets):
    batch_size = outputs.shape[0]
    outputs = np.reshape(outputs, (batch_size, -1))
    targets = np.reshape(targets, (batch_size, -1))

    return np.sum(np.mean(np.abs(outputs - targets), axis=1), axis=0)
