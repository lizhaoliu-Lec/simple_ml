import sys
import random

import numpy as np

from .optimizer import get_optimizer
from .loss import get_loss

__all__ = [
    'Sequential', 'Model'
]


class Module(object):
    def __init__(self):
        self.optimizer = None
        self.loss = None

    def compile(self, loss, optimizer='sgd'):
        """Configures the model for training.
        # Arguments
            loss: str (name of objective function) or objective function.
                See [losses](/losses).
                If the model has multiple outputs, you can use a different loss
                on each output by passing a dictionary or a list of losses.
                The loss value that will be minimized by the model
                will then be the sum of all individual losses.
            optimizer: str (name of optimizer) or optimizer object.
                See [optimizers](/optimizers).
        # Raises
            ValueError: In case of invalid arguments for
                `optimizer`, `loss`.
        """
        loss = loss or None
        optimizer = optimizer or None
        self.optimizer = get_optimizer(optimizer)
        self.loss = get_loss(loss)

    def peak(self, X, y, peek_type, set_type, num_show=5):
        allow_set_types = ['train', 'valid']
        allow_peek_types = ['single-cls', 'single-reg']
        assert set_type in allow_set_types
        assert peek_type in allow_peek_types
        # random select examples
        sample_idx = [_ for _ in range(X.shape[0])]
        num_show = 5
        random.shuffle(sample_idx)
        sample_X = X[sample_idx[:num_show]]
        sample_y = y[sample_idx[:num_show]]
        sample_y_pred = self.forward(sample_X, False)
        if peek_type == 'single-reg':
            sample_y = np.around(sample_y, 2).squeeze()
            sample_y_pred = np.around(sample_y_pred, 2).squeeze()
            dtype = float
        elif peek_type == 'single-cls':
            sample_y = np.argmax(sample_y, axis=1)
            sample_y_pred = np.argmax(sample_y_pred, axis=1)
            dtype = int
        else:
            pass
            dtype = None
        for i in range(num_show):
            out_str = "%s-example %d/%d: expect-[%s], predict-[%s]" % (
                set_type, i + 1, num_show,
                str(dtype(sample_y[i])), str(dtype(sample_y_pred[i])))
            print(out_str)

    @staticmethod
    def convert_dtype(dtype, *args):
        converted_data = [
            data.astype(dtype) if not np.issubdtype(dtype, data.dtype) else data for data in args
        ]
        return converted_data

    @staticmethod
    def data_loader(X, y, batch_size, shuffle=False):
        total_size = y.shape[0]
        assert total_size != 0, 'provided X, y must has at least one example.'
        num_steps = total_size // batch_size
        if total_size % batch_size != 0:
            num_steps = num_steps + 1

        if shuffle:
            rand_idx = [i for i in range(total_size)]
            random.shuffle(rand_idx)
            X = X[rand_idx]
            y = y[rand_idx]

        for i in range(num_steps):
            begin = i * batch_size
            end = min(begin + batch_size, total_size)
            yield X[begin:end], y[begin:end]

    def fit(self, X: np.array, y: np.array,
            epochs=100, batch_size=64, shuffle=True,
            validation_split=0., validation_data=None,
            verbose=1, file=sys.stdout, dtype=np.float64,
            metric=None, *args, **kwargs):
        # prepare data
        train_X, train_y = self.convert_dtype(dtype, X, y)

        if 1. > validation_split > 0.:
            split = int(train_y.shape[0] * validation_split)
            valid_X, valid_y = train_X[-split:], train_y[-split:]
            train_X, train_y = train_X[:-split], train_y[:-split]
            val_size = valid_y.shape[0]
        elif validation_data is not None:
            valid_X, valid_y = validation_data
            valid_X = np.asarray(valid_X)
            valid_y = np.asarray(valid_y)
            val_size = valid_y.shape[0]
        else:
            valid_X, valid_y = None, None

        self.train(train_X=train_X, train_y=train_y,
                   epochs=epochs, batch_size=batch_size, shuffle=shuffle,
                   valid_X=valid_X, valid_y=valid_y,
                   verbose=verbose, file=file,
                   metric=metric, *args, **kwargs)

    def train(self, train_X, train_y, epochs,
              batch_size, shuffle,
              valid_X=None, valid_y=None,
              verbose=1, file=sys.stdout,
              metric=None, *args, **kwargs):
        peek_type = kwargs.get('peek_type', None)
        num_show = kwargs.get('num_show', 5)
        train_size = train_y.shape[0]
        for iter_idx in range(1, epochs + 1):

            # training
            train_losses = 0
            train_matrices = 0
            for x_batch, y_batch in self.data_loader(train_X, train_y, batch_size, shuffle):
                # forward propagation
                y_pred = self.forward(x_batch, is_training=True)

                # backward propagation
                self.backward(y_pred, y_batch)

                # optimize
                self.optimize()

                # batch losses
                train_losses += self.loss.forward(y_pred, y_batch)

                if metric is not None:
                    train_matrices += metric(y_pred, y_batch)

            run_out = "epoch %5d/%5d, train-[loss: %.4f" % (
                iter_idx, epochs,
                float(train_losses / train_size))

            if metric is not None:
                run_out += ' | metric: %.4f]; ' % float(train_matrices / train_size)
            else:
                run_out += ']; '

            if valid_X is not None and valid_y is not None:
                val_size = valid_y.shape[0]
                # valid
                valid_matrices = 0
                valid_losses = 0
                for x_batch, y_batch in self.data_loader(valid_X, valid_y, batch_size=batch_size):
                    # forward propagation
                    y_pred = self.forward(x_batch, is_training=False)

                    # batch loss
                    # valid_losses += self.loss.forward(y_pred, y_batch)
                    valid_losses += self.loss.forward(y_pred, y_batch)

                    if metric is not None:
                        valid_matrices += metric(y_pred, y_batch)

                run_out += "valid-[loss: %.4f" % (float(valid_losses / val_size))
                if metric is not None:
                    run_out += ' | metric: %.4f]; ' % float(valid_matrices / val_size)
                else:
                    run_out += ']; '

            if verbose > 0 and iter_idx % verbose == 0:
                print(run_out, file=file)
                if peek_type is not None:
                    if valid_X is not None and valid_y is not None:
                        self.peak(valid_X, valid_y,
                                  peek_type=peek_type, set_type='valid',
                                  num_show=num_show)
                    else:
                        self.peak(train_X, train_y,
                                  peek_type=peek_type, set_type='train',
                                  num_show=num_show)

    def forward(self, X, is_training=False, *args, **kwargs):
        raise NotImplementedError

    def backward(self, y_hat, y, *args, **kwargs):
        raise NotImplementedError

    def optimize(self):
        raise NotImplementedError


class Sequential(Module):
    """Linear stack of layers.
    # Arguments
        layers: list of layers to add to the model.
    # Note
        The first layer passed to a Sequential model
        should have a defined input shape. What that
        means is that it should have received an `input_shape`
        or `batch_input_shape` argument,
        or for some type of layers (recurrent, Dense...)
        an `input_dim` argument.
    # Example
        ```python
            model = Sequential()
            # first layer must have a defined input shape
            model.add(Dense(32, input_dim=500))
            # afterwards, Keras does automatic shape inference
            model.add(Dense(32))
            # also possible (equivalent to the above):
            model = Sequential()
            model.add(Dense(32, input_shape=(500,)))
            model.add(Dense(32))
            # also possible (equivalent to the above):
            model = Sequential()
            # here the batch dimension is None,
            # which means any batch size will be accepted by the model.
            model.add(Dense(32, batch_input_shape=(None, 500)))
            model.add(Dense(32))
        ```
    """

    def __init__(self):
        super(Sequential, self).__init__()
        self.layers = list()

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, optimizer='sgd', **kwargs):
        """Configures the model for training.
        # Arguments
            loss: str (name of objective function) or objective function.
                See [losses](/losses).
                If the model has multiple outputs, you can use a different loss
                on each output by passing a dictionary or a list of losses.
                The loss value that will be minimized by the model
                will then be the sum of all individual losses.
            optimizer: str (name of optimizer) or optimizer object.
                See [optimizers](/optimizers).
        # Raises
            ValueError: In case of invalid arguments for
                `optimizer`, `loss`.
        """
        super(Sequential, self).compile(loss=loss, optimizer=optimizer, )

        prev_layer = None
        for layer in self.layers:
            layer.connection(prev_layer)
            prev_layer = layer

    def forward(self, X, is_training=False, *args, **kwargs):
        """ Calculate an output Y for the given input X. """
        for layer in self.layers[:]:
            X = layer.forward(X, is_training=is_training)
        return X

    def backward(self, y_hat, y, *args, **kwargs):
        # backward propagation
        grad = self.loss.backward(y_hat, y)
        for layer in self.layers[::-1]:
            grad = layer.backward(grad)

    def optimize(self):
        # get parameter and gradients
        params = list()
        grads = list()
        for layer in self.layers:
            params += layer.params
            grads += layer.grads

        # update parameters
        self.optimizer.minimize(params, grads)


class Model(Module):
    """Layers with multiple input and multiple output.
    # Arguments
        input: the input layer
        output: the output layer
    # Note
        The first layer passed to a Model model
        should have a defined input shape. What that
        means is that it should have received an `input_shape`
        or `batch_input_shape` argument,
        or for some type of layers (recurrent, Dense...)
        an `input_dim` argument.
    # TODO
        Add the multiple input and multiple output form, now it's similar
        to Sequential model.
    # Example
        ```python
            # first layer must have a defined input shape
            input = Input(input_shape=(input_dim, ))
            dense_1 = Dense(300, activator='selu')(input)
            dropout_1 = Dropout(0.2)(dense_1)
            softmax_1 = Softmax(label_size)(dropout_1)
            model = Model(input, softmax_1)
            # also possible (equivalent to the above):
            minput = Dense(32, input_dim=60)
            output = Dense(32)(input)
            model = Model(input, output)
        ```
    """

    def __init__(self, input, output):
        super(Model, self).__init__()
        self.input = input
        self.output = output

    def forward(self, X, is_training=False, *args, **kwargs):
        layer = self.input
        while layer is not None:
            X = layer.forward(X, is_training=is_training)
            layer = layer.next_layer
        return X

    def backward(self, y_hat, y, *args, **kwargs):
        grad = self.loss.backward(y_hat, y)
        layer = self.output
        while layer is not None:
            grad = layer.backward(grad)
            layer = layer.pre_layer

    def optimize(self):
        params = list()
        grads = list()
        pre_layer = self.output
        while pre_layer is not None:
            params += pre_layer.params
            grads += pre_layer.grads
            pre_layer = pre_layer.pre_layer

        # update parameters
        self.optimizer.minimize(params, grads)
