import numpy as np

from .activation import get_activation
from .regularizer import get_regularizer
from .initializer import xavier_uniform_initializer
from .utils import conv2D, im2col, col2im

__all__ = [
    'Input', 'FullyConnected', 'Linear', 'Dense',
    'Softmax', 'Flatten', 'Dropout', 'Activation',
    'Conv2d', 'AvgPooling2D', 'AvgPool2D',
    'MaxPooling2D', 'MaxPool2D', 'FastConv2d',
]


class Layer(object):
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


class Input(Layer):
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
        """
        全连接层的前向传播

        # Params
        inputs: 二维矩阵，行代表一个batch的大小，列代表特征

        # Return: 经过该层的输出结果
        """
        inputs = np.asarray(inputs)
        if len(inputs.shape) == 1:
            inputs = inputs[None, :]
        assert list(self.input_shape[1:]) == list(inputs.shape[1:])
        self.input_shape = inputs.shape
        self.output_shape[0] = self.input_shape[0]
        self.inputs = inputs
        # 关键操作，主要是输入与参数矩阵W、偏置b之间的操作，以及激活函数
        self.logit = np.dot(self.inputs, self.weight.T) + self.bias
        self.output = self.activation.forward(self.logit)
        return self.output

    def backward(self, pre_delta, *args, **kwargs):
        """
        梯度更新
        """
        if len(pre_delta.shape) == 1:
            pre_delta = pre_delta[None, :]
        batch_size, _ = self.inputs.shape
        act_delta = pre_delta * self.activation.backward(self.logit)
        # here should calculate the average value of batch
        self.delta_weight = np.dot(act_delta.T, self.inputs) + self.regularizer.backward(self.weight)
        self.delta_bias = np.mean(act_delta, axis=0)
        self.delta = np.dot(act_delta, self.weight)
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
    def __init__(self):
        """
        将输入的多维tensor展开成向量
        """
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
        self.output_shape = self._compute_output_shape(self.input_shape)
        return np.reshape(input, self.output_shape)

    def backward(self, pre_delta, *args, **kwargs):
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
        Dropout层

        # Params
        dropout: dropout的概率
        axis: 沿某个维度axis进行dropout操作，如果为None则是对所有元素进行
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
        """
        将激活函数应用到输入，得到经过激活函数的输出output

        # Params
        activator: 激活函数名
        """
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


class Filter(object):
    def __init__(self, filter_shape, initializer):
        """
        Filter，过滤器，滤波器

        # Params
        filter_shape: (filter_height, filter_width, input_depth)
        initializer: 滤波器的初始化方法
        """
        assert len(filter_shape) == 3
        self.filter_shape = filter_shape
        self.__W = initializer(filter_shape)  # filter权重矩阵
        self.__b = 0.  # filter偏置bias
        self.__delta_W = np.zeros(filter_shape)  # filter的权重矩阵梯度
        self.__delta_b = 0.  # filter的偏置梯度

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


class Conv2d(Layer):
    def __init__(self, kernel_size, channel_out, input_shape=None,
                 padding=0, stride=1, activation='relu', initializer=xavier_uniform_initializer):

        super(Conv2d, self).__init__()
        if isinstance(padding, int):
            padding = (padding, padding)
        self._check_convolution_layer(kernel_size, channel_out, padding, stride)
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.channel_out = channel_out
        self.output_shape = None
        self.padding = padding
        self.activation = get_activation(activation)
        self.initializer = initializer
        self.stride = stride if isinstance(stride, (list, tuple)) else (stride, stride)
        if self.input_shape is not None:
            self.connection(None)

    @property
    def delta(self):
        return self.__delta

    @delta.setter
    def delta(self, delta):
        self.__delta = delta

    @property
    def weight(self):
        return [f.weight for f in self.filters]

    @property
    def bias(self):
        return [f.bias for f in self.filters]

    @property
    def delta_weight(self):
        return [f.delta_weight for f in self.filters]

    @property
    def delta_bias(self):
        return [f.delta_bias for f in self.filters]

    @property
    def params(self):
        return self.weight + self.bias

    @property
    def grads(self):
        return self.delta_weight + self.delta_bias

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

        assert len(self.input_shape) == 4

        if self.output_shape is None:
            self.output_shape = self._calc_output_shape(self.input_shape, self.kernel_size,
                                                        self.stride, self.padding, self.channel_out)
        self.filters = [Filter(list(self.kernel_size) + [self.input_shape[3]], self.initializer)
                        for _ in range(self.channel_out)]

    def forward(self, input, *args, **kwargs):

        input = np.asarray(input)

        self.input_shape = input.shape
        self.output_shape[0] = input.shape[0]

        logit = np.zeros(self.output_shape)

        assert list(input.shape[1:]) == list(self.input_shape[1:])

        self.padded_input = self.pad(input, self.padding)

        x_pad = self.padded_input
        H_hat, W_hat = self.output_shape[1], self.output_shape[2]
        stride_h, stride_w = self.stride[0], self.stride[1]
        kernel_h, kernel_w = self.kernel_size[0], self.kernel_size[1]
        for _f, f in enumerate(self.filters):
            for _h in range(H_hat):
                for _w in range(W_hat):
                    _h_begin, _w_begin = _h * stride_h, _w * stride_w
                    _h_end, _w_end = _h_begin + kernel_h, _w_begin + kernel_w
                    conv_sub = np.sum(x_pad[:, _h_begin:_h_end, _w_begin:_w_end, :] * f.weight, axis=(1, 2, 3)) + f.bias
                    logit[:, _h, _w, _f] = conv_sub
        self.logit = logit

        return self.activation.forward(self.logit)

    def backward(self, pre_delta, *args, **kwargs):

        pre_delta = pre_delta * self.activation.backward(self.logit)

        delta_pad = np.zeros_like(self.padded_input)
        H_hat, W_hat = self.output_shape[1], self.output_shape[2]
        stride_h, stride_w = self.stride[0], self.stride[1]
        kernel_h, kernel_w = self.kernel_size[0], self.kernel_size[1]
        x_pad = self.padded_input
        for _f, f in enumerate(self.filters):
            for _h in range(H_hat):
                for _w in range(W_hat):
                    _h_begin, _w_begin = _h * stride_h, _w * stride_w
                    _h_end, _w_end = _h_begin + kernel_h, _w_begin + kernel_w

                    volume = np.reshape(pre_delta[:, _h, _w, _f], (-1, 1, 1, 1))

                    delta_pad[:, _h_begin:_h_end, _w_begin:_w_end, :] += f.weight * volume
                    if _h and _w:
                        f.delta_weight = np.zeros_like(f.delta_weight)
                        f.delta_bias = np.zeros_like(f.delta_bias)
                    f.delta_weight += np.sum(x_pad[:, _h_begin:_h_end, _w_begin:_w_end, :] * volume, axis=0)
                    f.delta_bias += np.sum(volume)
        padding_h, padding_w = self.padding
        delta = delta_pad[:, padding_h:-padding_h, padding_w:-padding_w, :]
        self.delta = delta

        return self.delta

    @staticmethod
    def pad(inputs, padding):
        """
        对输入的feature map进行零填充

        # Params
        inputs: 输入的feature map
        zero_padding: 需要填充的零的个数，输入是一个二元tuple，分别表示填充在高度和宽度上的零的个数
        """
        inputs = np.asarray(inputs)
        if list(padding) == [0, 0]:
            return inputs

        if inputs.ndim == 3:
            inputs = inputs[:, :, :, None]

        if inputs.ndim == 4:
            # input_batch, input_height, input_width, input_channel = inputs.shape
            # padded_input = np.zeros([input_batch, input_height + 2 * padding[0],
            #                          input_width + 2 * padding[1], input_channel])
            # padded_input[:, padding[0]:input_height + padding[0],
            # padding[1]:input_width + padding[1], :] = inputs
            #
            padded_input = np.pad(inputs, ((0, 0), (padding[0], padding[0]), (padding[1], padding[1]), (0, 0)))
            return padded_input
        else:
            raise ValueError('Your input must be a 3-D or 4-D tensor.')

    @staticmethod
    def _calc_output_size(input_len, filter_len, stride, zero_padding):
        return int((input_len + 2 * zero_padding - filter_len) / stride + 1)

    def _calc_output_shape(self, input_shape, kernel_size, stride, padding, channel_out):
        """
        计算output的shape，这个shape也是回传的误差的shape
        """
        output_height = self._calc_output_size(input_shape[1], kernel_size[0],
                                               stride[0], padding[0])
        output_width = self._calc_output_size(input_shape[2], kernel_size[1],
                                              stride[1], padding[1])
        output_channel = channel_out
        return [input_shape[0], output_height, output_width, output_channel]

    @staticmethod
    def _check_convolution_layer(kernel_size, channel_out, stride, padding):
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


class FastConv2d(Conv2d):
    def __init__(self, kernel_size, channel_out, input_shape=None,
                 padding=0, stride=1, activation='relu', initializer=xavier_uniform_initializer):
        super(Conv2d, self).__init__()

        self._check_convolution_layer(kernel_size, channel_out, padding, stride)
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
            self.output_shape = self._calc_output_shape(self.input_shape, self.kernel_size,
                                                        self.stride, self.padding, self.channel_out)

        _kh, _kw = self.kernel_size, self.kernel_size
        _cin, _cout = self.input_shape[3], self.channel_out
        self.weight = self.initializer((_kh, _kw, _cin, _cout))
        self.bias = np.zeros((1, 1, 1, _cout))
        self.delta_weight = np.zeros((_kh, _kw, _cin, _cout))
        self.delta_bias = np.zeros((1, 1, 1, _cout))

    def forward(self, input, *args, **kwargs):
        self.input = input
        self.logit = conv2D(input, self.weight, self.stride, self.padding) + self.bias
        return self.activation.forward(self.logit)

    def backward(self, pre_delta, *args, **kwargs):
        pre_delta = pre_delta * self.activation.backward(self.logit)

        _kh, _kw = self.kernel_size, self.kernel_size
        _cin, _cout = self.input_shape[3], self.channel_out
        x = self.input
        w = self.weight
        p = self.padding
        s = self.stride

        pre_delta_col = pre_delta.transpose(3, 1, 2, 0).reshape(_cout, -1)
        w_col = w.transpose(3, 2, 0, 1).reshape(_cout, -1).T
        x_col, p = im2col(x, w.shape, p, s)

        # compute gradients via matrix multiplication and reshape
        self.delta_bias = -1 * pre_delta_col.sum(axis=1).reshape(1, 1, 1, -1)
        # self.delta_weight = np.dot(pre_delta_col, x_col.T).reshape((_cout, _cin, _kh, _kw)).transpose(2, 3, 1, 0)
        self.delta_weight = -1 * (pre_delta_col @ x_col.T).reshape((_cout, _cin, _kh, _kw)).transpose(2, 3, 1, 0)

        # reshape columnized dX back into the same format as the input volume
        delta_col = np.dot(w_col, pre_delta_col)
        # delta_col = w_col @ pre_delta_col
        self.delta = col2im(delta_col, x.shape, w.shape, p, s).transpose(0, 2, 3, 1)

        return self.delta

    def _calc_output_shape(self, input_shape, kernel_size, stride, padding, channel_out):

        output_spatial = self._calc_output_size(input_shape[1], kernel_size, stride, padding)
        return [input_shape[0], output_spatial, output_spatial, channel_out]

    @staticmethod
    def _check_convolution_layer(kernel_size, channel_out, stride, padding):

        if not isinstance(kernel_size, int):
            raise ValueError('`kernel_size` must be int')
        if not isinstance(channel_out, int):
            raise ValueError('`filter_num` must be int')
        if not isinstance(stride, (int, tuple, list)):
            raise ValueError('`stride` must be tuple(list) or int')
        if not isinstance(padding, (int, tuple, list)):
            raise ValueError('`zero_padding` must be tuple(list) or int')


class MaxPool2D(Layer):
    def __init__(self, kernel_size, input_shape=None, stride=1, padding=0):
        super(MaxPool2D, self).__init__()
        self.input_shape = input_shape
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride
        self.__delta = None
        if self.input_shape is not None:
            self.connection(None)

    @property
    def delta(self):
        return self.__delta

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

        output_height = (self.input_shape[1] + self.padding[0] * 2
                         - self.kernel_size[0]) // self.stride[0] + 1
        output_width = (self.input_shape[2] + self.padding[1] * 2
                        - self.kernel_size[1]) // self.stride[1] + 1
        self.output_shape = [self.input_shape[0], output_height,
                             output_width, 1 if len(self.input_shape) == 3 else self.input_shape[3]]

    def forward(self, inputs, *args, **kwargs):
        inputs = np.asarray(inputs)
        assert list(self.input_shape[1:]) == list(inputs.shape[1:])
        self.input_shape = inputs.shape
        self.output_shape[0] = self.input_shape[0]
        if inputs.ndim == 3:
            inputs = inputs[:, :, :, None]
        if inputs.ndim == 4:
            self.inputs = inputs
        else:
            raise ValueError('Your input must be a 2-D or 3-D tensor.')

        if len(self.output_shape) == 3:
            output = np.zeros(list(self.output_shape) + [1, ])
        else:
            output = np.zeros(self.output_shape)

        x = self.pad(self.inputs, self.padding)
        H_hat, W_hat = self.output_shape[1], self.output_shape[2]
        stride_h, stride_w = self.stride[0], self.stride[1]
        kernel_h, kernel_w = self.kernel_size[0], self.kernel_size[1]
        for _h in range(H_hat):
            for _w in range(W_hat):
                _h_begin, _w_begin = _h * stride_h, _w * stride_w
                _h_end, _w_end = _h_begin + kernel_h, _w_begin + kernel_w
                output_max_sub = np.max(x[:, _h_begin:_h_end, _w_begin:_w_end, :], axis=(1, 2))
                output[:, _h, _w, :] = output_max_sub

        self.output = output
        self.inputs = x
        if len(self.output_shape) == 3:
            return output[:, :, :, 0]
        else:
            return output

    def backward(self, pre_delta, *args, **kwargs):
        if len(self.input_shape) == 3:
            __delta = np.zeros(tuple(self.input_shape) + (1,))
        else:
            __delta = np.zeros(self.inputs.shape)

        x = self.inputs
        H_hat, W_hat = self.output_shape[1], self.output_shape[2]
        stride_h, stride_w = self.stride[0], self.stride[1]
        kernel_h, kernel_w = self.kernel_size[0], self.kernel_size[1]
        channel_size = self.output.shape[-1]
        for _h in range(H_hat):
            for _w in range(W_hat):
                _h_begin, _w_begin = _h * stride_h, _w * stride_w
                _h_end, _w_end = _h_begin + kernel_h, _w_begin + kernel_w
                output_max_sub_mask = np.array(self.output[:, _h, _w, :] == x[:, _h_begin:_h_end, _w_begin:_w_end, :],
                                               dtype=np.float)
                __delta[:, _h_begin:_h_end, _w_begin:_w_end, :] += np.reshape(
                    output_max_sub_mask * pre_delta[:, _h, _w, :], (-1, 1, 1, channel_size))

        self.__delta = __delta
        if len(self.input_shape) == 3:
            return __delta[:, :, :, 0]
        else:
            return __delta

    @staticmethod
    def pad(inputs, padding):
        inputs = np.asarray(inputs)
        if list(padding) == [0, 0]:
            return inputs

        if inputs.ndim == 3:
            inputs = inputs[:, :, :, None]

        if inputs.ndim == 4:
            _, input_height, input_width, input_channel = inputs.shape
            padded_input = np.zeros([_, input_height + 2 * padding[0],
                                     input_width + 2 * padding[1], input_channel])
            padded_input[:, padding[0]:input_height + padding[0], padding[1]:input_width + padding[1], :] = inputs
            return padded_input
        else:
            raise ValueError('Your input must be a 3-D or 4-D tensor.')


class AvgPool2D(Layer):
    def __init__(self, kernel_size, input_shape=None, stride=1, padding=0):
        super(AvgPool2D, self).__init__()
        self.input_shape = input_shape
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride

        self.__delta = None
        if self.input_shape is not None:
            self.connection(None)

    @property
    def delta(self):
        return self.__delta

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

        output_height = (self.input_shape[1] + self.padding[0] * 2
                         - self.kernel_size[0]) // self.stride[0] + 1
        output_width = (self.input_shape[2] + self.padding[1] * 2
                        - self.kernel_size[1]) // self.stride[1] + 1
        self.output_shape = [self.input_shape[0], output_height,
                             output_width, 1 if len(self.input_shape) == 3 else self.input_shape[3]]

    def forward(self, inputs, *args, **kwargs):
        inputs = np.asarray(inputs)

        assert list(self.input_shape[1:]) == list(inputs.shape[1:])
        self.input_shape = inputs.shape
        self.output_shape[0] = self.input_shape[0]

        if inputs.ndim == 3:
            inputs = inputs[:, :, :, None]
        if inputs.ndim == 4:
            self.inputs = inputs
        else:
            raise ValueError('Your input must be a 3-D or 4-D tensor.')

        if len(self.output_shape) == 3:
            output = np.zeros(list(self.output_shape) + [1, ])
        else:
            output = np.zeros(self.output_shape)

        x = self.pad(self.inputs, self.padding)
        H_hat, W_hat = self.output_shape[1], self.output_shape[2]
        stride_h, stride_w = self.stride[0], self.stride[1]
        kernel_h, kernel_w = self.kernel_size[0], self.kernel_size[1]
        for _h in range(H_hat):
            for _w in range(W_hat):
                _h_begin, _w_begin = _h * stride_h, _w * stride_w
                _h_end, _w_end = _h_begin + kernel_h, _w_begin + kernel_w
                output_avg_sub = np.mean(x[:, _h_begin:_h_end, _w_begin:_w_end, :], axis=(1, 2))
                output[:, _h, _w, :] = output_avg_sub
        if len(self.output_shape) == 3:
            return output[:, :, :, 0]
        else:
            return output

    def backward(self, pre_delta, *args, **kwargs):
        if len(self.input_shape) == 3:
            __delta = np.zeros(tuple(self.input_shape) + (1,))
        else:
            __delta = np.zeros(self.inputs.shape)

        H_hat, W_hat = self.output_shape[1], self.output_shape[2]
        stride_h, stride_w = self.stride[0], self.stride[1]
        kernel_h, kernel_w = self.kernel_size[0], self.kernel_size[1]

        num_pixel = kernel_h * kernel_w
        delta_avg_sub = np.ones((1, kernel_h, kernel_w, 1)) / num_pixel
        for _h in range(H_hat):
            for _w in range(W_hat):
                _h_begin, _w_begin = _h * stride_h, _w * stride_w
                _h_end, _w_end = _h_begin + kernel_h, _w_begin + kernel_w
                __delta[:, _h_begin:_h_end, _w_begin:_w_end, :] += delta_avg_sub

        self.__delta = __delta
        if len(self.input_shape) == 3:
            return __delta[:, :, :, 0]
        else:
            return __delta

    @staticmethod
    def pad(inputs, padding):
        inputs = np.asarray(inputs)
        if list(padding) == [0, 0]:
            return inputs

        if inputs.ndim == 3:
            inputs = inputs[:, :, :, None]

        if inputs.ndim == 4:
            _, input_height, input_width, input_channel = inputs.shape
            padded_input = np.zeros([_, input_height + 2 * padding[0],
                                     input_width + 2 * padding[1], input_channel])
            padded_input[:, padding[0]:input_height + padding[0],
            padding[1]:input_width + padding[1], :] = inputs
            return padded_input
        else:
            raise ValueError('Your input must be a 3-D or 4-D tensor.')


# alias names
MaxPooling2D = MaxPool2D
AvgPooling2D = AvgPool2D
