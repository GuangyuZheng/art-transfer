from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import Zeros
from keras.layers import Dropout
from keras.regularizers import l2


# input: embedding, s(i-1), aligned_state, h(0,1,2,....), input_mask
# shape: (None, embed_dim), (None, state_dim), (None, state_dim), (None, seq_len, context_dim), (None, seq_len)
class ART(Layer):

    def __init__(self, embed_dim, state_dim, context_dim, bias=True, **kwargs):
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.context_dim = context_dim
        self.bias = bias
        self.supports_masking = True
        super(ART, self).__init__(**kwargs)

    def build(self, input_shape):
        h_input_shape = input_shape[-1]
        self.seq_len = h_input_shape[1]
        self.W = self.add_weight(name='W',
                                 shape=(self.embed_dim, self.state_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.W_z = self.add_weight(name='W_z',
                                   shape=(self.embed_dim, self.state_dim),
                                   initializer='glorot_uniform',
                                   trainable=True)
        self.W_r = self.add_weight(name='W_r',
                                   shape=(self.embed_dim, self.state_dim),
                                   initializer='glorot_uniform',
                                   trainable=True)
        self.U = self.add_weight(name='U',
                                 shape=(self.state_dim, self.state_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.U_z = self.add_weight(name='U_z',
                                   shape=(self.state_dim, self.state_dim),
                                   initializer='glorot_uniform',
                                   trainable=True)
        self.U_r = self.add_weight(name='U_r',
                                   shape=(self.state_dim, self.state_dim),
                                   initializer='glorot_uniform',
                                   trainable=True)
        if self.bias:
            self.U_o = self.add_weight(name='U_o',
                                       shape=(self.state_dim, self.state_dim),
                                       initializer=Zeros(),
                                       regularizer=l2(0),
                                       trainable=True)
        self.C = self.add_weight(name='C',
                                 shape=(self.context_dim, self.state_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.C_z = self.add_weight(name='C_z',
                                   shape=(self.context_dim, self.state_dim),
                                   initializer='glorot_uniform',
                                   trainable=True)
        self.C_r = self.add_weight(name='C_r',
                                   shape=(self.context_dim, self.state_dim),
                                   initializer='glorot_uniform',
                                   trainable=True)
        if self.bias:
            self.C_o = self.add_weight(name='C_o',
                                       shape=(self.context_dim, self.state_dim),
                                       initializer=Zeros(),
                                       regularizer=l2(0),
                                       trainable=True)
        self.v_a = self.add_weight(name='v_a',
                                   shape=(self.state_dim, 1),
                                   initializer='glorot_uniform',
                                   trainable=True)
        self.W_a = self.add_weight(name='W_a',
                                   shape=(self.state_dim, self.state_dim),
                                   initializer='glorot_uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=(self.context_dim, self.state_dim),
                                   initializer='glorot_uniform',
                                   trainable=True)
        super(ART, self).build(input_shape)

    def call(self, inputs, **kwargs):
        embed, s_prev, aligned_state, context, input_mask = inputs
        self.s_prev = s_prev
        context = K.reshape(context, shape=(-1, self.seq_len, self.context_dim))

        s_prev_tile = K.dot(self.s_prev, self.W_a)  # None, state_dim
        s_prev_tile = K.expand_dims(s_prev_tile, axis=1)
        s_prev_tile = K.tile(s_prev_tile, (1, self.seq_len, 1))
        e = K.tanh(s_prev_tile + K.dot(context, self.U_a))  # None, seq_len, state_dim
        e = K.dot(e, self.v_a)  # None, seq_len, 1
        e = K.squeeze(e, axis=-1)  # None, seq_len
        e = K.exp(e)
        e = e * input_mask
        alpha = e / (K.sum(e, axis=-1, keepdims=True) + K.epsilon())  # None, seq_len
        c = K.batch_dot(alpha, context)  # None, state_dim <- (None, seq_len) (None, seq_len, state_dim)

        if self.bias:
            o = K.sigmoid(Dropout(0.5)(K.dot(aligned_state, self.U_o)) + Dropout(0.5)(K.dot(c, self.C_o)))  # None, state_dim
            c_hat = (1 - o) * aligned_state + o * c
        else:
            c_hat = c

        r = K.sigmoid((K.dot(embed, self.W_r)) + (K.dot(s_prev, self.U_r)) + (K.dot(c_hat, self.C_r)))  # None, state_dim
        z = K.sigmoid((K.dot(embed, self.W_z)) + (K.dot(s_prev, self.U_z)) + (K.dot(c_hat, self.C_z)))  # None, state_dim
        s_hat = K.tanh(K.dot(embed, self.W) + K.dot(r * s_prev, self.U) + K.dot(c_hat, self.C))  # None, state_dim
        s = (1 - z) * s_prev + z * s_hat  # None, state_dim
        return [s, alpha]

    def get_e(self, args):
        s, h = args
        e = K.tanh(K.dot(s, self.W_a) + K.dot(h, self.U_a))  # None, state_dim
        e = K.dot(e, self.v_a)  # None, 1
        return e

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], self.state_dim), (input_shape[0][0], self.seq_len)]