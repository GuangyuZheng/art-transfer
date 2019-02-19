from keras.layers import *
from keras.models import Model
from recurrentshop.engine import RNNCell
from custom_rnn.attention_layer import *


def _slice(x, dim, index):
    return x[:, index * dim: dim * (index + 1)]


def get_slices(x, n):
    dim = int(K.int_shape(x)[1] / n)
    return [Lambda(_slice, arguments={'dim': dim, 'index': i}, output_shape=lambda s: (s[0], dim))(x) for i in range(n)]


class Identity(Layer):

    def call(self, x):
        return x + 0.


class ExtendedRNNCell(RNNCell):

    def __init__(self, units=None,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 # kernel_initializer='zeros',
                 recurrent_initializer='orthogonal',
                 # recurrent_initializer='zeros',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 settings=None,
                 **kwargs):
        if units is None:
            assert 'output_dim' in kwargs, 'Missing argument: units'
        else:
            kwargs['output_dim'] = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.settings = settings
        super(ExtendedRNNCell, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'activation': activations.serialize(self.activation),
            'recurrent_activation': activations.serialize(self.recurrent_activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(ExtendedRNNCell, self).get_config()
        config.update(base_config)
        return config


class NormalLSTMCell(ExtendedRNNCell):

    def build_model(self, input_shape):
        output_dim = self.output_dim
        kernel_output_dim = self.settings.hidden_state_size
        input_dim = input_shape[-1]
        output_shape = (input_shape[0], output_dim)
        x = Input(batch_shape=input_shape)
        hc_tm1 = Input(batch_shape=output_shape)
        h_tm1 = Lambda(lambda x: x[:, :kernel_output_dim])(hc_tm1)
        c_tm1 = Lambda(lambda x: x[:, kernel_output_dim:])(hc_tm1)
        kernel = Dense(kernel_output_dim * 4,
                       kernel_initializer=self.kernel_initializer,
                       kernel_regularizer=self.kernel_regularizer,
                       kernel_constraint=self.kernel_constraint,
                       use_bias=self.use_bias,
                       bias_initializer=self.bias_initializer,
                       bias_regularizer=self.bias_regularizer,
                       bias_constraint=self.bias_constraint)
        recurrent_kernel = Dense(kernel_output_dim * 4,
                                 kernel_initializer=self.recurrent_initializer,
                                 kernel_regularizer=self.recurrent_regularizer,
                                 kernel_constraint=self.recurrent_constraint,
                                 use_bias=True)

        kernel_out = kernel(x)
        recurrent_kernel_out = recurrent_kernel(h_tm1)
        kernel_i, kernel_f, kernel_c, kernel_o = get_slices(kernel_out, 4)
        recurrent_kernel_i, recurrent_kernel_f, recurrent_kernel_c, recurrent_kernel_o = get_slices(
            recurrent_kernel_out, 4)

        x_i = kernel_i
        x_f = kernel_f
        x_c = kernel_c
        x_o = kernel_o

        i = Activation(self.recurrent_activation)(add([x_i, recurrent_kernel_i]))
        f = Activation(self.recurrent_activation)(add([x_f, recurrent_kernel_f]))
        temp = Activation(self.activation)(add([x_c, recurrent_kernel_c]))
        c = add([multiply([f, c_tm1]), multiply([i, temp])])
        o = Activation(self.recurrent_activation)(add([x_o, recurrent_kernel_o]))
        c_foroutput = Activation(self.activation)(c)
        h = multiply([o, c_foroutput])

        # y = concatenate([h, h_tm1, c_tm1])
        y = concatenate([h, h, c])
        hc = concatenate([h, c])

        return Model([x, hc_tm1], [hc, Identity()(hc)])


# add gate to aligned state
class ARTTransferCell(ExtendedRNNCell):

    def build_model(self, input_shape):
        settings = self.settings
        kernel_output_dim = settings.hidden_state_size
        output_dim = self.output_dim
        # input_dim = input_shape[-1]
        output_shape = (input_shape[0], output_dim)

        lrx = Input(batch_shape=input_shape)  # input at time t
        y_tml = Input(batch_shape=output_shape)
        # h_r = Input(batch_shape=output_shape)  # previous hidden state
        # c_r = Input(batch_shape=output_shape)  # previous cell state
        h_r = Lambda(lambda x: x[:, :settings.hidden_state_size])(y_tml)  # previous hidden state
        c_r = Lambda(lambda x: x[:, settings.hidden_state_size:
                                    settings.hidden_state_size * 2])(y_tml)  # previous cell state

        h_l = Lambda(lambda x: x[:, :settings.context_size * settings.seq_len])(lrx)
        c_l = Lambda(lambda x: x[:, settings.context_size * settings.seq_len:
                                    settings.context_size * 2 * settings.seq_len])(lrx)
        aligned_h = Lambda(lambda x: x[:, settings.context_size * 2 * settings.seq_len:
                                          settings.context_size * 2 * settings.seq_len +
                                          settings.context_size])(lrx)
        aligned_c = Lambda(lambda x: x[:, settings.context_size * 2 * settings.seq_len + settings.context_size:
                                          settings.context_size * 2 * settings.seq_len +
                                          settings.context_size * 2])(lrx)
        x_r = Lambda(lambda x: x[:, settings.context_size * 2 * settings.seq_len + settings.context_size * 2:
                                    -settings.seq_len])(lrx)
        input_mask = Lambda(lambda x: x[:, -settings.seq_len:])(lrx)

        kernel = Dense(kernel_output_dim * 4,
                       kernel_initializer=self.kernel_initializer,
                       kernel_regularizer=self.kernel_regularizer,
                       kernel_constraint=self.kernel_constraint,
                       use_bias=self.use_bias,
                       bias_initializer=self.bias_initializer,
                       bias_regularizer=self.bias_regularizer,
                       bias_constraint=self.bias_constraint)
        recurrent_kernel = Dense(kernel_output_dim * 4,
                                 kernel_initializer=self.recurrent_initializer,
                                 kernel_regularizer=self.recurrent_regularizer,
                                 kernel_constraint=self.recurrent_constraint,
                                 use_bias=True)
        H = Reshape((-1, settings.context_size))(h_l)  # N, seq_len, d
        C = Reshape((-1, settings.context_size))(c_l)  # N, seq_len, d
        # print(K.int_shape(H))
        x = x_r
        h_tml, h_attention = ART(embed_dim=settings.emb_size, state_dim=settings.hidden_state_size,
                                 context_dim=settings.context_size,
                                 bias=settings.bias)([x, h_r, aligned_h, H, input_mask])
        c_tml, c_attention = ART(embed_dim=settings.emb_size, state_dim=settings.hidden_state_size,
                                 context_dim=settings.context_size,
                                 bias=settings.bias)([x, c_r, aligned_c, C, input_mask])
        # print(K.int_shape(h_tml))
        # print(K.int_shape(h_attention))
        kernel_out = kernel(x)
        recurrent_kernel_out = recurrent_kernel(h_tml)
        kernel_i, kernel_f, kernel_c, kernel_o = get_slices(kernel_out, 4)
        recurrent_kernel_i, recurrent_kernel_f, recurrent_kernel_c, recurrent_kernel_o = get_slices(
            recurrent_kernel_out, 4)

        x_i = kernel_i
        x_f = kernel_f
        x_c = kernel_c
        x_o = kernel_o

        i = Activation(self.recurrent_activation)(add([x_i, recurrent_kernel_i]))
        f = Activation(self.recurrent_activation)(add([x_f, recurrent_kernel_f]))
        temp = Activation(self.activation)(add([x_c, recurrent_kernel_c]))
        c = add([multiply([f, c_tml]), multiply([i, temp])])
        o = Activation(self.recurrent_activation)(add([x_o, recurrent_kernel_o]))
        c_foroutput = Activation(self.activation)(c)
        h = multiply([o, c_foroutput])
        # print(K.int_shape(h))
        # print(K.int_shape(c))
        y = concatenate([h, c, h_attention, c_attention])
        # print(K.int_shape(y))

        return Model([lrx, y_tml], [y, Identity()(y)])
