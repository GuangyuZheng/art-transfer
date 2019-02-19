from keras.layers import Input, Dropout, Lambda, LeakyReLU, Conv1D, multiply, Reshape, Embedding, GRU, LSTM, Bidirectional
from keras.models import Model
import keras.backend as K


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

    c_emb = cembed_layer(char_input)
    cc = Bidirectional(GRU(units=100))(c_emb)
    cc = Dropout(0.5)(cc)
    char_model = Model(char_input, cc)
    return char_model


def lstm_char_model(settings):
    char_embd_dim = settings.ch_emb_size
    char_cnt = settings.char_cnt

    char_input = Input(shape=(None,), dtype='int32')
    cembed_layer = Embedding(input_dim=char_cnt + 1, output_dim=char_embd_dim, mask_zero=True)

    c_emb = cembed_layer(char_input)
    cc = Bidirectional(LSTM(units=100))(c_emb)
    cc = Dropout(0.5)(cc)
    char_model = Model(char_input, cc)
    return char_model
