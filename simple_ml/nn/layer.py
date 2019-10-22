import numpy as np
from .activation import get_activation
from .regularizer import get_regularizer
from .initializer import xavier_uniform_initializer
from .utils import conv_forward_im2col, conv_backward_im2col
from .utils import max_pool_forward_fast, max_pool_backward_fast
from .utils import avg_pool_forward_fast, avg_pool_backward_fast

__all__ = [
    'Input', 'FullyConnected', 'Linear', 'Dense',
    'Softmax', 'Flatten', 'Dropout', 'Activation',
    'Conv2d',
    'AvgPooling2D', 'AvgPool2D',
    'MaxPooling2D', 'MaxPool2D',
]


class Layer(object):
    """
    Base Layer.
    """

    def __init__(self):
        self.__input_shape = None
        self.__output_shape = None
        self.__pre_layer = None
        self.__next_layer = None

    @property
    def input_shape(self):
        return self.__input_shape

    @input_shape.setter
    def input_shape(self, input_shape):
        self.__input_shape = input_shape

    @property
    def output_shape(self):
        return self.__output_shape

    @output_shape.setter
    def output_shape(self, output_shape):
        self.__output_shape = output_shape

    @property
    def pre_layer(self):
        return self.__pre_layer

    @pre_layer.setter
    def pre_layer(self, pre_layer):
        self.__pre_layer = pre_layer

    @property
    def next_layer(self):
        return self.__next_layer

    @next_layer.setter
    def next_layer(self, next_layer):
        self.__next_layer = next_layer

    def connection(self, pre_layer):
        raise NotImplementedError

    @property
    def params(self):
        raise NotImplementedError

    @property
    def grads(self):
        raise NotImplementedError

    def forward(self, inputs, *args, **kwargs):
        raise NotImplementedError

    def backward(self, pre_delta, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def assert_shape(expect_shape, got_shape):
        got, expect = list(got_shape[1:]), list(expect_shape[1:])
        assert got == expect, 'expect shape: `%s`, but got `%s` instead.' % (expect, got)


class Input(Layer):
    """
    Input Layer.
    """

    def __init__(self,
                 input_shape=None,
                 batch_size=None,
                 batch_input_shape=None,
                 input_tensor=None,
                 dtype=None):
        super(Input, self).__init__()
        if input_shape and batch_input_shape:
            raise ValueError('Only provide the input_shape OR '
                             'batch_input_shape argument to '
                             'Input, not both at the same time.')
        if input_tensor is not None and batch_input_shape is None:
            # If input_tensor is set, and batch_input_shape is not set:
            # Attempt automatic input shape inference.
            try:
                batch_input_shape = np.asarray(input_tensor).shape
            except TypeError:
                if not input_shape and not batch_input_shape:
                    raise ValueError('Input was provided '
                                     'an input_tensor argument, '
                                     'but its input shape cannot be '
                                     'automatically inferred. '
                                     'You should pass an input_shape or '
                                     'batch_input_shape argument.')
        if not batch_input_shape:
            if not input_shape:
                raise ValueError('An Input layer should be passed either '
                                 'a `batch_input_shape` or an `input_shape`.')
            else:
                if isinstance(input_shape, np.int):
                    batch_input_shape = (batch_size, input_shape)
                else:
                    batch_input_shape = (batch_size,) + tuple(input_shape)
        else:
            batch_input_shape = tuple(batch_input_shape)

        if not dtype:
            if input_tensor is None:
                dtype = np.float32
            else:
                dtype = np.dtype(input_tensor)

        self.input_shape = batch_input_shape
        self.dtype = dtype
        self.connection(None)

    @property
    def params(self):
        return list()

    @property
    def grads(self):
        return list()

    def connection(self, pre_layer):
        if pre_layer is not None:
            raise ValueError('Input layer must be your first layer.')
        self.output_shape = self.input_shape

    def call(self, pre_layer=None, *args, **kwargs):
        raise ValueError('Input layer is not callable.')

    def forward(self, inputs, *args, **kwargs):
        inputs = np.asarray(inputs)
        assert self.input_shape[1:] == inputs.shape[1:]
        self.input_shape = inputs.shape
        self.output_shape = self.input_shape
        return inputs

    def backward(self, pre_delta, *args, **kwargs):
        pass


class Linear(Layer):
    def __init__(self, output_dim,
                 input_dim=None,
                 activation='relu',
                 regularizer=None,
                 initializer=xavier_uniform_initializer):
        super(Linear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.regularizer = get_regularizer(regularizer)
        self.activation = get_activation(activation)
        self.initializer = initializer
        self.input_shape = None
        self.output_shape = None
        if self.input_dim is not None:
            self.connection(None)

    @property
    def weight(self):
        return self.__W

    @property
    def bias(self):
        return self.__b

    @property
    def delta_weight(self):
        return self.__delta_W

    @property
    def delta_bias(self):
        return self.__delta_b

    @property
    def delta(self):
        return self.__delta

    @weight.setter
    def weight(self, W):
        self.__W = W

    @bias.setter
    def bias(self, b):
        self.__b = b

    @delta_weight.setter
    def delta_weight(self, delta_W):
        self.__delta_W = delta_W

    @delta_bias.setter
    def delta_bias(self, delta_b):
        self.__delta_b = delta_b

    @delta.setter
    def delta(self, delta):
        self.__delta = delta

    @property
    def params(self):
        return [self.weight, self.bias]

    @property
    def grads(self):
        return [self.delta_weight, self.delta_bias]

    def call(self, pre_layer=None, *args, **kwargs):
        self.connection(pre_layer)
        return self

    def connection(self, pre_layer):
        self.pre_layer = pre_layer
        if pre_layer is None:
            if self.input_dim is None:
                raise ValueError('input_size must not be `None` as the first layer.')
            self.input_shape = [None, self.input_dim]
            self.output_shape = [None, self.output_dim]
        else:
            pre_layer.next_layer = self
            self.input_dim = pre_layer.output_shape[1]
            self.input_shape = pre_layer.output_shape
            self.output_shape = [self.input_shape[0], self.output_dim]
        self.weight = self.initializer([self.output_dim, self.input_dim])
        self.bias = self.initializer([self.output_dim])
        self.delta_weight = np.zeros([self.output_dim, self.input_dim])
        self.delta_bias = np.zeros([self.output_dim])
        self.delta = np.zeros([self.input_dim])

    def forward(self, inputs, *args, **kwargs):
        inputs = np.asarray(inputs)
        if len(inputs.shape) == 1:
            inputs = inputs[None, :]
        self.assert_shape(self.input_shape, inputs.shape)
        self.input_shape = inputs.shape
        self.output_shape[0] = self.input_shape[0]
        self.inputs = inputs
        self.logit = np.dot(self.inputs, self.weight.T) + self.bias
        self.assert_shape(self.output_shape, self.logit.shape)
        self.output = self.activation.forward(self.logit)
        self.assert_shape(self.output_shape, self.output.shape)
        return self.output

    def backward(self, pre_delta, *args, **kwargs):
        if len(pre_delta.shape) == 1:
            pre_delta = pre_delta[None, :]
        self.assert_shape(self.output_shape, pre_delta.shape)
        batch_size, _ = self.inputs.shape
        act_delta = pre_delta * self.activation.backward(self.logit)
        self.assert_shape(self.output_shape, act_delta.shape)
        self.delta_weight = np.dot(act_delta.T, self.inputs) + self.regularizer.backward(self.weight)
        self.delta_bias = np.mean(act_delta, axis=0)
        self.delta = np.dot(act_delta, self.weight)
        self.assert_shape(self.input_shape, self.delta.shape)
        return self.delta

    @property
    def regularizer_loss(self):
        return self.regularizer.forward(self.weight)


Dense = FullyConnected = Linear


class Softmax(Linear):
    def __init__(self, output_dim, input_dim=None,
                 initializer=xavier_uniform_initializer):
        super(Softmax, self).__init__(output_dim=output_dim, input_dim=input_dim,
                                      activation='softmax', initializer=initializer)


class Flatten(Layer):
    """
    Flatten Layer.
    """

    def __init__(self):
        super(Flatten, self).__init__()

    @property
    def params(self):
        return list()

    @property
    def grads(self):
        return list()

    def call(self, pre_layer=None, *args, **kwargs):
        self.connection(pre_layer)
        return self

    def connection(self, pre_layer):
        if pre_layer is None:
            raise ValueError('Flatten could not be used as the first layer')
        self.pre_layer = pre_layer
        pre_layer.next_layer = self
        self.input_shape = pre_layer.output_shape
        self.output_shape = self._compute_output_shape(self.input_shape)

    def forward(self, input, *args, **kwargs):
        self.input_shape = input.shape
        # print(self.input_shape)
        self.output_shape = self._compute_output_shape(self.input_shape)
        return np.reshape(input, self.output_shape)

    def backward(self, pre_delta, *args, **kwargs):
        # print(self.output_shape)
        return np.reshape(pre_delta, self.input_shape)

    @staticmethod
    def _compute_output_shape(input_shape):
        if not all(input_shape[1:]):
            raise ValueError('The shape of the input to "Flatten" '
                             'is not fully defined '
                             '(got ' + str(input_shape[1:]) + '. '
                                                              'Make sure to pass a complete "input_shape" '
                                                              'or "batch_input_shape" argument to the first '
                                                              'layer in your model.')
        return input_shape[0], np.prod(input_shape[1:])


class Dropout(Layer):
    def __init__(self, dropout=0., axis=None):
        """
        Dropout Layer.
        """
        super(Dropout, self).__init__()
        self.dropout = dropout
        self.axis = axis
        self.mask = None

    @property
    def params(self):
        return list()

    @property
    def grads(self):
        return list()

    def call(self, pre_layer=None, *args, **kwargs):
        self.connection(pre_layer)
        return self

    def connection(self, pre_layer):
        if pre_layer is None:
            raise ValueError('Dropout could not be used as the first layer')
        if self.axis is None:
            self.axis = range(len(pre_layer.output_shape))
        self.output_shape = pre_layer.output_shape
        self.pre_layer = pre_layer
        pre_layer.next_layer = self

    def forward(self, inputs, is_training=True, *args, **kwargs):
        # if 0. < self.dropout < 1:
        #     if is_training:
        #         self.mask = np.random.binomial(1, 1 - self.dropout, np.asarray(inputs.shape)[self.axis])
        #         return self.mask * inputs / (1 - self.dropout)
        #     else:
        #         return inputs
        # else:
        #     return inputs
        if 0. < self.dropout < 1 and is_training:
            self.mask = np.random.binomial(1, 1 - self.dropout, np.asarray(inputs.shape)[self.axis])
            return self.mask * inputs / (1 - self.dropout)
        else:
            return inputs

    def backward(self, pre_delta, *args, **kwargs):
        if 0. < self.dropout < 1.:
            return self.mask * pre_delta
        else:
            return pre_delta


class Activation(Layer):
    def __init__(self, activation, input_shape=None):
        super(Activation, self).__init__()
        self.activator = get_activation(activation)
        self.input_shape = input_shape

    @property
    def params(self):
        return list()

    @property
    def grads(self):
        return list()

    def call(self, pre_layer=None, *args, **kwargs):
        self.connection(pre_layer)
        return self

    def connection(self, pre_layer):
        self.pre_layer = pre_layer
        if pre_layer is None:
            if self.input_shape is None:
                raise ValueError('input_size must not be `None` as the first layer.')
            self.output_shape = self.input_shape
        else:
            pre_layer.next_layer = self
            self.input_shape = pre_layer.output_shape
            self.output_shape = self.input_shape

    def forward(self, inputs, *args, **kwargs):
        """
        前向传播
        # Params
        inputs: 2-D tensors, row represents samples, col represents features

        # Return
        None
        """
        inputs = np.asarray(inputs)
        if len(inputs.shape) == 1:
            inputs = inputs[None, :]
        assert list(self.input_shape[1:]) == list(inputs.shape[1:])
        self.input_shape = inputs.shape
        self.output_shape = self.input_shape
        self.inputs = inputs
        self.output = self.activator.forward(self.inputs)
        return self.output

    def backward(self, pre_delta, *args, **kwargs):
        if len(pre_delta.shape) == 1:
            pre_delta = pre_delta[None, :]
        act_delta = pre_delta * self.activator.backward(self.inputs)
        return act_delta


class Conv2d(Layer):
    def __init__(self, kernel_size, channel_out, input_shape=None,
                 padding=0, stride=1, activation='relu', initializer=xavier_uniform_initializer):
        super(Conv2d, self).__init__()
        if not isinstance(kernel_size, (tuple, list)):
            kernel_size = (kernel_size, kernel_size)

        self._check_parameter_setting(kernel_size, channel_out, padding, stride)
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.channel_out = channel_out
        self.output_shape = None
        self.padding = padding
        self.activation = get_activation(activation)
        self.initializer = initializer
        self.stride = stride
        if self.input_shape is not None:
            self.connection(None)

    @property
    def weight(self):
        return self.__W

    @property
    def bias(self):
        return self.__b

    @property
    def delta_weight(self):
        return self.__delta_W

    @property
    def delta_bias(self):
        return self.__delta_b

    @property
    def delta(self):
        return self.__delta

    @weight.setter
    def weight(self, W):
        self.__W = W

    @bias.setter
    def bias(self, b):
        self.__b = b

    @delta_weight.setter
    def delta_weight(self, delta_W):
        self.__delta_W = delta_W

    @delta_bias.setter
    def delta_bias(self, delta_b):
        self.__delta_b = delta_b

    @delta.setter
    def delta(self, delta):
        self.__delta = delta

    @property
    def params(self):
        return [self.weight, self.bias]

    @property
    def grads(self):
        return [self.delta_weight, self.delta_bias]

    def connection(self, pre_layer):
        if pre_layer is None:
            if self.input_shape is None:
                raise ValueError('input_shape must not be `None` as the first layer.')
        else:
            self.input_shape = pre_layer.output_shape
            self.pre_layer = pre_layer
            pre_layer.next_layer = self

        assert len(self.input_shape) == 4

        if self.output_shape is None:
            self.output_shape = self._get_output_shape()

        _kh, _kw = self.kernel_size
        _cin, _cout = self.input_shape[3], self.channel_out
        self.weight = self.initializer((_cout, _cin, _kh, _kw))
        self.bias = np.zeros(_cout)
        self.delta_weight = np.zeros((_cout, _cin, _kh, _kw))
        self.delta_bias = np.zeros(_cout)

    def forward(self, input, *args, **kwargs):
        self.assert_shape(self.input_shape, input.shape)
        _kh, _kw, _cin, _ = self.weight.shape
        s, p = self.stride, self.padding
        w, b = self.weight, self.bias
        input = input.transpose(0, 3, 1, 2)
        conv_param = {'stride': s, 'pad': p}
        self.logits, self.cache = conv_forward_im2col(input, w, b, conv_param)
        self.logits = self.logits.transpose(0, 2, 3, 1)
        self.assert_shape(self.output_shape, self.logits.shape)
        output = self.activation.forward(self.logits)
        self.assert_shape(self.output_shape, output.shape)
        return output

    def backward(self, pre_delta, *args, **kwargs):
        self.assert_shape(self.output_shape, pre_delta.shape)
        pre_delta = pre_delta * self.activation.backward(self.logits)
        self.assert_shape(self.output_shape, pre_delta.shape)
        pre_delta = pre_delta.transpose(0, 3, 1, 2)
        delta, self.delta_weight, self.delta_bias = conv_backward_im2col(pre_delta, self.cache)
        self.delta = delta.transpose(0, 2, 3, 1)
        self.assert_shape(self.input_shape, self.delta.shape)
        return self.delta

    def _get_output_shape(self):
        _, H, W, channel_in = self.input_shape
        p, s, = self.padding, self.stride
        kh, kw = self.kernel_size
        assert (H + 2 * p - kh) % s == 0, 'invalid (H, padding, kernel_size): (%d, %d, %d)' % (H, p, kh)
        assert (W + 2 * p - kw) % s == 0, 'invalid (W, padding, kernel_size): (%d, %d, %d)' % (W, p, kw)
        H_hat = (H + 2 * p - kh) // s + 1
        W_hat = (W + 2 * p - kw) // s + 1
        return _, H_hat, W_hat, self.channel_out

    @staticmethod
    def _check_parameter_setting(kernel_size, channel_out, stride, padding):
        filter_height, filter_width = kernel_size
        if not isinstance(filter_height, int):
            raise ValueError('`filter_height` must be int')
        if not isinstance(filter_width, int):
            raise ValueError('`filter_width` must be int')
        if not isinstance(channel_out, int):
            raise ValueError('`filter_num` must be int')
        if not isinstance(stride, (int, tuple, list)):
            raise ValueError('`stride` must be tuple(list) or int')
        if not isinstance(padding, (int, tuple, list)):
            raise ValueError('`zero_padding` must be tuple(list) or int')


class MaxPool2D(Layer):
    def __init__(self, kernel_size, input_shape=None, stride=1):
        super(MaxPool2D, self).__init__()
        self.input_shape = input_shape
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.__delta = None
        if self.input_shape is not None:
            self.connection(None)

    @property
    def delta(self):
        return self.__delta

    @delta.setter
    def delta(self, delta):
        self.__delta = delta

    @property
    def params(self):
        return list()

    @property
    def grads(self):
        return list()

    def call(self, pre_layer=None, *args, **kwargs):
        self.connection(pre_layer)
        return self

    def connection(self, pre_layer):
        if pre_layer is None:
            if self.input_shape is None:
                raise ValueError('input_shape must not be `None` as the first layer.')
        else:
            self.input_shape = pre_layer.output_shape
            self.pre_layer = pre_layer
            pre_layer.next_layer = self

        self.output_shape = self._get_output_shape()

    def forward(self, inputs, *args, **kwargs):
        self.assert_shape(self.input_shape, inputs.shape)
        inputs = inputs.transpose(0, 3, 1, 2)
        pool_param = {'pool_height': self.kernel_size[0],
                      'pool_width': self.kernel_size[1],
                      'stride': self.stride}
        output, self.cache = max_pool_forward_fast(inputs, pool_param)
        output = output.transpose(0, 2, 3, 1)
        self.assert_shape(self.output_shape, output.shape)
        return output

    def backward(self, pre_delta, *args, **kwargs):
        self.assert_shape(self.output_shape, pre_delta.shape)
        pre_delta = pre_delta.transpose(0, 3, 1, 2)
        delta = max_pool_backward_fast(pre_delta, self.cache)
        self.delta = delta.transpose(0, 2, 3, 1)
        self.assert_shape(self.input_shape, self.delta.shape)
        return self.delta

    def _get_output_shape(self):
        _, H, W, channel_in = self.input_shape
        s = self.stride
        kh, kw = self.kernel_size
        assert (H - kh) % s == 0, 'invalid (H, kernel_size): (%d, %d)' % (H, kh)
        assert (W - kw) % s == 0, 'invalid (W, kernel_size): (%d, %d)' % (W, kw)
        H_hat = (H - kh) // s + 1
        W_hat = (W - kw) // s + 1
        return _, H_hat, W_hat, channel_in


class AvgPool2D(Layer):
    def __init__(self, kernel_size, input_shape=None, stride=1):
        super(AvgPool2D, self).__init__()
        self.input_shape = input_shape
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.__delta = None
        if self.input_shape is not None:
            self.connection(None)

    @property
    def delta(self):
        return self.__delta

    @delta.setter
    def delta(self, delta):
        self.__delta = delta

    @property
    def params(self):
        return list()

    @property
    def grads(self):
        return list()

    def call(self, pre_layer=None, *args, **kwargs):
        self.connection(pre_layer)
        return self

    def connection(self, pre_layer):
        if pre_layer is None:
            if self.input_shape is None:
                raise ValueError('input_shape must not be `None` as the first layer.')
        else:
            self.input_shape = pre_layer.output_shape
            self.pre_layer = pre_layer
            pre_layer.next_layer = self

        self.output_shape = self._get_output_shape()

    def forward(self, inputs, *args, **kwargs):
        self.assert_shape(self.input_shape, inputs.shape)
        inputs = inputs.transpose(0, 3, 1, 2)
        pool_param = {'pool_height': self.kernel_size[0],
                      'pool_width': self.kernel_size[1],
                      'stride': self.stride}
        output, self.cache = avg_pool_forward_fast(inputs, pool_param)
        output = output.transpose(0, 2, 3, 1)
        self.assert_shape(self.output_shape, output.shape)
        return output

    def backward(self, pre_delta, *args, **kwargs):
        self.assert_shape(self.output_shape, pre_delta.shape)
        pre_delta = pre_delta.transpose(0, 3, 1, 2)
        delta = avg_pool_backward_fast(pre_delta, self.cache)
        self.delta = delta.transpose(0, 2, 3, 1)
        self.assert_shape(self.input_shape, self.delta.shape)
        return self.delta

    def _get_output_shape(self):
        _, H, W, channel_in = self.input_shape
        s = self.stride
        kh, kw = self.kernel_size
        assert (H - kh) % s == 0, 'invalid (H, kernel_size): (%d, %d)' % (H, kh)
        assert (W - kw) % s == 0, 'invalid (W, kernel_size): (%d, %d)' % (W, kw)
        H_hat = (H - kh) // s + 1
        W_hat = (W - kw) // s + 1
        return _, H_hat, W_hat, channel_in


# alias names
MaxPooling2D = MaxPool2D
AvgPooling2D = AvgPool2D
