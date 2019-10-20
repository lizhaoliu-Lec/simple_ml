import numpy as np

__all__ = [
    'get_metric',
    'accuracy', 'mean_absolute_error', 'mean_square_error',
    'binary_accuracy', 'svm_binary_accuracy',
]


def binary_accuracy(outputs, targets):
    y_predicts = (outputs >= 0.5).astype(int)
    # y_targets = np.argmax(targets, axis=1)
    y_targets = targets
    acc = y_predicts == y_targets
    # print('y_predicts', y_predicts)
    # print(y_predicts.shape, 'y_predicts')
    # print(targets.shape, 'targets')
    # print(np.sum(acc, axis=0).shape, 'np.sum(acc, axis=0)')
    # print(acc.shape, 'np.sum(acc, axis=0)')
    return np.sum(acc, axis=0)


def svm_binary_accuracy(outputs, targets):

    y_predicts = np.where(outputs >= 0, 1, -1).astype(int)
    # y_targets = np.argmax(targets, axis=1)
    y_targets = targets
    acc = y_predicts == y_targets
    # print('y_predicts', y_predicts)
    # print(y_predicts.shape, 'y_predicts')
    # print(targets.shape, 'targets')
    # print(np.sum(acc, axis=0).shape, 'np.sum(acc, axis=0)')
    # print(acc.shape, 'np.sum(acc, axis=0)')
    return np.sum(acc, axis=0)


def accuracy(outputs, targets):
    y_predicts = np.argmax(outputs, axis=1)
    # y_targets = np.argmax(targets, axis=1)
    y_targets = targets.squeeze()
    acc = y_predicts == y_targets
    # print(y_predicts.shape, 'y_predicts')
    # print(targets.shape, 'targets')
    # print(np.sum(acc, axis=0).shape, 'np.sum(acc, axis=0)')
    # print(acc.shape, 'np.sum(acc, axis=0)')
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


_metric_map = {
    'mae': mean_absolute_error,
    'mse': mean_square_error,
    'accuracy': accuracy,
    'binary_accuracy': binary_accuracy,
    'binaryaccuracy': binary_accuracy,
    'svm_binary_accuracy': svm_binary_accuracy,
    'svmbinaryaccuracy': svm_binary_accuracy,
}


def get_metric(metric=None):
    if metric is None:
        return None
    elif isinstance(metric, str):
        original_metric = metric
        metric = metric.lower()
        if metric in _metric_map:
            return _metric_map[metric]
        else:
            raise ValueError('Unknown metric name `{}`'.format(original_metric))
    else:
        raise ValueError('Unknown metric type `{}`'.format(metric.__name__))
