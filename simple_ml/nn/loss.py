import numpy as np

__all__ = [
    # original class
    'MeanSquareLoss', 'CrossEntropy', 'LogLikelihoodLoss',

    # alias
    'mse', 'ce', 'MSE', 'CE',

    # factory interface
    'get_loss'
]


class Loss(object):
    @staticmethod
    def forward(y_hat, y):
        raise NotImplementedError

    @staticmethod
    def backward(y_hat, y):
        raise NotImplementedError


class MeanSquareLoss(Loss):
    """
    calculate the MSE
    """

    @staticmethod
    def forward(y_hat, y):
        return 0.5 * np.sum(np.power(y_hat - y, 2))

    @staticmethod
    def backward(y_hat, y):
        return y_hat - y


class LogLikelihoodLoss(Loss):
    """
        多分类的log loss, 主要用于前一层为softmax的情况
    """

    @staticmethod
    def forward(y_hat, y):
        assert (np.abs(np.sum(y_hat, axis=1) - 1.) < cutoff).all()
        assert (np.abs(np.sum(y, axis=1) - 1.) < cutoff).all()
        y_hat = _cutoff(y_hat)
        y = _cutoff(y)
        return -np.mean(np.sum(np.nan_to_num(y * np.log(y_hat)), axis=1))

    @staticmethod
    def backward(y_hat, y):
        """
        The loss partial by z is : y_hat * (y - y_hat) / (-1 / y_hat) = y_hat - y
        softmax + loglikelihoodCost == sigmoid + crossentropyCost
        :param y_hat:
        :param y:
        :param activator:
        :return:
        """
        assert (np.abs(np.sum(y_hat, axis=1) - 1.) < cutoff).all()
        assert (np.abs(np.sum(y, axis=1) - 1.) < cutoff).all()
        y_hat = _cutoff(y_hat)
        y = _cutoff(y)
        return y_hat - y


# alias
mse = MSE = MeanSquareLoss
ce = CE = CrossEntropy = LogLikelihoodLoss

cutoff = 1e-12


def _cutoff(z):
    return np.clip(z, cutoff, 1 - cutoff)


_loss_map = {
    'mse': MeanSquareLoss,
    'ce': CrossEntropy
}


def get_loss(loss):
    if isinstance(loss, Loss):
        return loss
    elif isinstance(loss, str):
        original_loss = loss
        loss = loss.lower()
        if loss in _loss_map:
            return _loss_map[loss]()
        else:
            raise ValueError('Unknown loss name `{}`'.format(original_loss))
    else:
        raise ValueError('Unknown loss type `{}`'.format(loss.__class__.__name__))
