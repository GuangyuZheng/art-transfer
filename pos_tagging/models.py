import sys

sys.path.append('..')
from common_layers.no_transfer_encoder import *
from common_layers.transfer_encoder import *
from common_layers.char_model import *
from keras_contrib.layers import CRF


def bi_lstm_no_transfer_model(settings):
    seq_len = settings.seq_len
    word_len = settings.max_char_len
    emb_size = settings.emb_size
    vec = settings.vec
    class_num = len(settings.label_index)

    sen_input = Input(shape=(seq_len,), dtype='int32')
    sen_mask = Lambda(lambda x: K.cast(K.not_equal(x, 0), 'float32'))(sen_input)  # None, seq_len
    sen_cinput = Input(shape=(seq_len, word_len), dtype='int32')
    sen_embed = Embedding(input_dim=len(vec), output_dim=emb_size, weights=[vec],
                          name='left_embed', mask_zero=True, trainable=True)(sen_input)
    sen_embed = Dropout(0.5, (None, 1, None))(sen_embed)
    if settings.char_encoding == 'cnn':
        cm = cnn_char_model(settings)
    else:
        cm = lstm_char_model(settings)
    cm.name = 'char_model'
    sen_cembed = TimeDistributed(cm, name='left_char_embed')(sen_cinput)
    embed = concatenate([sen_embed, sen_cembed])
    rnn_y1 = multi_layers_bidirectional_lstm_no_transfer_encoder(embed, settings, layer_num=1)
    rnn_y1 = Dropout(0.5)(rnn_y1)
    crf = CRF(class_num, name='crf')
    crf_result = crf(rnn_y1, mask=sen_mask)

    model = Model(inputs=[sen_input, sen_cinput], outputs=[crf_result])
    model.compile(loss=crf.loss_function, optimizer='Adagrad', metrics=[crf.accuracy])
    return model


def bi_gru_no_transfer_model(settings):
    seq_len = settings.seq_len
    word_len = settings.max_char_len
    emb_size = settings.emb_size
    vec = settings.vec
    class_num = len(settings.label_index)

    sen_input = Input(shape=(seq_len,), dtype='int32')
    sen_mask = Lambda(lambda x: K.cast(K.not_equal(x, 0), 'float32'))(sen_input)  # None, seq_len
    sen_cinput = Input(shape=(seq_len, word_len), dtype='int32')
    sen_embed = Embedding(input_dim=len(vec), output_dim=emb_size, weights=[vec],
                          name='left_embed', mask_zero=True, trainable=True)(sen_input)
    sen_embed = Dropout(0.5, (None, 1, None))(sen_embed)
    if settings.char_encoding == 'cnn':
        cm = cnn_char_model(settings)
    else:
        cm = gru_char_model(settings)
    cm.name = 'char_model'
    sen_cembed = TimeDistributed(cm, name='left_char_embed')(sen_cinput)
    embed = concatenate([sen_embed, sen_cembed])
    rnn_y1 = multi_layers_bidirectional_gru_no_transfer_encoder(embed, settings, layer_num=1)
    rnn_y1 = Dropout(0.5)(rnn_y1)
    crf = CRF(class_num, name='crf')
    crf_result = crf(rnn_y1, mask=sen_mask)

    model = Model(inputs=[sen_input, sen_cinput], outputs=[crf_result])
    model.compile(loss=crf.loss_function, optimizer='Adagrad', metrics=[crf.accuracy])
    return model


def bi_art_lstm_model(settings):
    seq_len = settings.seq_len
    word_len = settings.max_char_len
    emb_size = settings.emb_size
    vec = settings.vec
    class_num = len(settings.label_index)

    sen_input = Input(shape=(seq_len,), dtype='int32')
    sen_mask = Lambda(lambda x: K.cast(K.not_equal(x, 0), 'float32'))(sen_input)  # None, seq_len
    sen_cinput = Input(shape=(seq_len, word_len), dtype='int32')
    sen_embed = Embedding(input_dim=len(vec), output_dim=emb_size, input_length=seq_len, weights=[vec],
                          name='left_embed', mask_zero=True, trainable=True)(sen_input)
    sen_embed = Dropout(0.5, (None, 1, None))(sen_embed)
    if settings.char_encoding == 'cnn':
        cm = cnn_char_model(settings)
    else:
        cm = lstm_char_model(settings)
    cm.name = 'char_model'
    sen_cembed = TimeDistributed(cm, name='left_char_embed')(sen_cinput)
    left_embed = concatenate([sen_embed, sen_cembed])
    right_embed = left_embed

    _, merged_right_rnn_y1 = bidirectional_art_lstm_encoder(left_embed, right_embed, sen_mask, settings)
    merged_right_rnn_y1 = Dropout(0.5)(merged_right_rnn_y1)
    crf = CRF(class_num, name='crf')
    crf_result = crf(merged_right_rnn_y1, mask=sen_mask)

    model = Model(inputs=[sen_input, sen_cinput], outputs=[crf_result])
    model.compile(loss=crf.loss_function, optimizer='Adagrad', metrics=[crf.accuracy])
    return model


def bi_art_gru_model(settings):
    seq_len = settings.seq_len
    word_len = settings.max_char_len
    emb_size = settings.emb_size
    vec = settings.vec
    class_num = len(settings.label_index)

    sen_input = Input(shape=(seq_len,), dtype='int32')
    sen_mask = Lambda(lambda x: K.cast(K.not_equal(x, 0), 'float32'))(sen_input)  # None, seq_len
    sen_cinput = Input(shape=(seq_len, word_len), dtype='int32')
    sen_embed = Embedding(input_dim=len(vec), output_dim=emb_size, input_length=seq_len, weights=[vec],
                          name='left_embed', mask_zero=True, trainable=True)(sen_input)
    sen_embed = Dropout(0.5, (None, 1, None))(sen_embed)
    if settings.char_encoding == 'cnn':
        cm = cnn_char_model(settings)
    else:
        cm = gru_char_model(settings)
    cm.name = 'char_model'
    sen_cembed = TimeDistributed(cm, name='left_char_embed')(sen_cinput)
    left_embed = concatenate([sen_embed, sen_cembed])
    right_embed = left_embed

    _, merged_right_rnn_y1 = bidirectional_art_gru_encoder(left_embed, right_embed, sen_mask, settings)
    merged_right_rnn_y1 = Dropout(0.5)(merged_right_rnn_y1)
    crf = CRF(class_num, name='crf')
    crf_result = crf(merged_right_rnn_y1, mask=sen_mask)

    model = Model(inputs=[sen_input, sen_cinput], outputs=[crf_result])
    model.compile(loss=crf.loss_function, optimizer='Adagrad', metrics=[crf.accuracy])
    return model


