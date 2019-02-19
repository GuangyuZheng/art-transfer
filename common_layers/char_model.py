from keras.layers import Input, Dropout, Lambda, LeakyReLU, Conv1D, multiply, Reshape, Embedding
from keras.models import Model
import keras.backend as K


def char_model(settings):
    maxWordLen = settings.word_len
    dropout_rate = settings.dropout_rate
    char_embd_dim = settings.ch_emb_size
    char_size = settings.char_size

    char_input = Input(shape=(maxWordLen,), dtype='int32')
    cembed_layer = Embedding(char_size + 1, char_embd_dim)

    c_emb = cembed_layer(char_input)
    # c_emb = Dropout(dropout_rate, (None, 1, None))(c_emb)
    c_mask = Lambda(lambda x: K.cast(K.not_equal(x, 0), 'float32'))(char_input)
    cc = Conv1D(char_embd_dim, 3, padding='same')(c_emb)
    cc = LeakyReLU()(cc)
    cc = multiply([cc, Reshape((-1, 1))(c_mask)])
    cc = Lambda(lambda x: K.sum(x, 1))(cc)
    char_model = Model(char_input, cc)
    return char_model
