from keras.layers import Input, Dropout, Lambda, LeakyReLU, Conv1D, multiply, Reshape, Embedding, GRU, LSTM, Bidirectional
from keras.models import Model
import keras.backend as K
from lambda_utilities.masked_function import masked_global_max_pooling_1d


def cnn_char_model(settings):
    char_embd_dim = settings.ch_emb_size
    char_cnt = settings.char_cnt

    char_input = Input(shape=(None,), dtype='int32')
    cembed_layer = Embedding(char_cnt + 1, char_embd_dim)

    c_emb = cembed_layer(char_input)
    c_mask = Lambda(lambda x: K.cast(K.not_equal(x, 0), 'float32'))(char_input)
    cc = Conv1D(char_embd_dim, 3, padding='same')(c_emb)
    cc = LeakyReLU()(cc)
    cc = multiply([cc, Reshape((-1, 1))(c_mask)])
    cc = Lambda(lambda x: K.sum(x, 1))(cc)
    char_model = Model(char_input, cc)
    return char_model


def gru_char_model(settings):
    char_embd_dim = settings.ch_emb_size
    char_cnt = settings.char_cnt

    char_input = Input(shape=(None,), dtype='int32')
    cembed_layer = Embedding(input_dim=char_cnt + 1, output_dim=char_embd_dim, mask_zero=True)
    c_mask = Lambda(lambda x: K.cast(K.not_equal(x, 0), 'float32'))(char_input)

    c_emb = cembed_layer(char_input)
    cc = Bidirectional(GRU(units=50, return_sequences=True))(c_emb)
    cc = Lambda(masked_global_max_pooling_1d)([cc, c_mask])
    char_model = Model(char_input, cc)
    return char_model


def lstm_char_model(settings):
    char_embd_dim = settings.ch_emb_size
    char_cnt = settings.char_cnt

    char_input = Input(shape=(None,), dtype='int32')
    cembed_layer = Embedding(input_dim=char_cnt + 1, output_dim=char_embd_dim, mask_zero=True)
    c_mask = Lambda(lambda x: K.cast(K.not_equal(x, 0), 'float32'))(char_input)

    c_emb = cembed_layer(char_input)
    cc = Bidirectional(LSTM(units=50, return_sequences=True))(c_emb)
    cc = Lambda(masked_global_max_pooling_1d)([cc, c_mask])
    char_model = Model(char_input, cc)
    return char_model
