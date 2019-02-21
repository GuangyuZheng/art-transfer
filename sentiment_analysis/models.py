import sys

sys.path.append("..")
from common_layers.no_transfer_encoder import *
from common_layers.transfer_encoder import *
from common_layers.char_model import *


# multi layers bidirectional lstm
def multi_layers_bilstm_no_transfer_model(settings, layer_num=1):
    seq_len = settings.seq_len
    w_emb_size = settings.w_emb_size
    dropout_rate = settings.dropout_rate
    vec = settings.vec

    sen_input = Input(shape=(seq_len,), dtype='int32')
    sen_mask = Lambda(lambda x: K.cast(K.not_equal(x, 0), 'float32'))(sen_input)  # None, seq_len
    sen_embed = Embedding(input_dim=len(vec), output_dim=w_emb_size, input_length=seq_len, weights=[vec],
                          name='left_embed', mask_zero=True, trainable=True)(sen_input)

    embed = sen_embed
    settings.emb_size = w_emb_size

    rnn_y1 = multi_layers_bidirectional_lstm_no_transfer_encoder(embed, settings, layer_num=layer_num)
    rnn_y1 = Lambda(masked_global_max_pooling_1d)([rnn_y1, sen_mask])
    rnn_y1 = Dropout(dropout_rate)(rnn_y1)
    densed = Dense(1, activation="sigmoid")(rnn_y1)

    model = Model(inputs=[sen_input], outputs=[densed])
    return model


# bidirectional attentive recurrent transfer learning
def bidirectional_art_lstm_model(settings):
    seq_len = settings.seq_len
    w_emb_size = settings.w_emb_size
    dropout_rate = settings.dropout_rate
    vec = settings.vec

    sen_input = Input(shape=(seq_len,), dtype='int32')
    sen_mask = Lambda(lambda x: K.cast(K.not_equal(x, 0), 'float32'))(sen_input)  # None, seq_len
    sen_embed = Embedding(input_dim=len(vec), output_dim=w_emb_size, input_length=seq_len, weights=[vec],
                          name='left_embed', mask_zero=True, trainable=True)(sen_input)

    left_embed = sen_embed
    settings.emb_size = w_emb_size

    right_embed = left_embed

    _, merged_right_rnn_y1 = bidirectional_art_lstm_encoder(left_embed, right_embed, sen_mask, settings)
    merged_right_rnn_y1 = Lambda(masked_global_max_pooling_1d)([merged_right_rnn_y1, sen_mask])
    merged_right_rnn_y1 = Dropout(dropout_rate)(merged_right_rnn_y1)
    right_densed = Dense(1, activation="sigmoid")(merged_right_rnn_y1)

    model_input = [sen_input]

    model = Model(inputs=model_input, outputs=[right_densed])
    return model


# bidirectional attentive recurrent transfer learning
def bidirectional_art_gru_model(settings):
    seq_len = settings.seq_len
    w_emb_size = settings.w_emb_size
    dropout_rate = settings.dropout_rate
    vec = settings.vec

    sen_input = Input(shape=(seq_len,), dtype='int32')
    sen_mask = Lambda(lambda x: K.cast(K.not_equal(x, 0), 'float32'))(sen_input)  # None, seq_len
    sen_embed = Embedding(input_dim=len(vec), output_dim=w_emb_size, input_length=seq_len, weights=[vec],
                          name='left_embed', mask_zero=True, trainable=True)(sen_input)

    left_embed = sen_embed
    settings.emb_size = w_emb_size

    right_embed = left_embed

    _, merged_right_rnn_y1 = bidirectional_art_gru_encoder(left_embed, right_embed, sen_mask, settings)
    merged_right_rnn_y1 = Lambda(masked_global_max_pooling_1d)([merged_right_rnn_y1, sen_mask])
    merged_right_rnn_y1 = Dropout(dropout_rate)(merged_right_rnn_y1)
    right_densed = Dense(1, activation="sigmoid")(merged_right_rnn_y1)

    model_input = [sen_input]

    model = Model(inputs=model_input, outputs=[right_densed])
    return model


# bidirectional attentive recurrent transfer learning version two
def bidirectional_art_lstm_model_v2(settings):
    seq_len = settings.seq_len
    emb_size = settings.w_emb_size
    vec = settings.vec

    input_layer = Input(shape=(seq_len,), dtype='int32')
    input_mask = Lambda(lambda x: K.cast(K.not_equal(x, 0), 'float32'))(input_layer)  # None, seq_len

    left_embed = Embedding(input_dim=len(vec), output_dim=emb_size, input_length=seq_len, trainable=True,
                           name='left_embed', mask_zero=True)(input_layer)
    right_embed = left_embed

    merged_right_rnn_y1, h_att, c_att, back_h_att, back_c_att = bidirectional_art_lstm_encoder_v2(left_embed,
                                                                                                  right_embed,
                                                                                                  input_mask, settings)
    merged_right_rnn_y1 = Lambda(masked_global_max_pooling_1d)([merged_right_rnn_y1, input_mask])
    merged_right_rnn_y1 = Dropout(0.2)(merged_right_rnn_y1)
    right_densed = Dense(1, activation="sigmoid")(merged_right_rnn_y1)

    model = Model(inputs=[input_layer], outputs=[right_densed])
    h_attention_model = Model(inputs=[input_layer], outputs=[h_att])
    c_attention_model = Model(inputs=[input_layer], outputs=[c_att])
    backward_h_attention_model = Model(inputs=[input_layer], outputs=[back_h_att])
    backward_c_attention_model = Model(inputs=[input_layer], outputs=[back_c_att])
    return model, h_attention_model, c_attention_model, backward_h_attention_model, backward_c_attention_model
