import sys
sys.path.append("..")
from custom_rnn.lstm_cell import *
from recurrentshop import RecurrentSequential
from custom_rnn.cells import *


# bidirectional attentive recurrent transfer learning version two
def bidirectional_art_lstm_encoder_v2(left_embed, right_embed, input_mask, settings):
    seq_len = settings.seq_len
    hidden_state_size = settings.hidden_state_size
    emb_size = settings.emb_size

    # forward part
    input_mask_tile = Lambda(lambda x: K.tile(K.expand_dims(x, axis=1), (1, seq_len, 1)))(input_mask)

    # backward part
    reversed_input_mask = Lambda(lambda x: K.reverse(x, axes=1))(input_mask)
    reversed_input_mask_tile = Lambda(lambda x: K.tile(K.expand_dims(x, axis=1), (1, seq_len, 1)))(reversed_input_mask)

    # left(open domain) part
    # # forward part
    rnn_left_layer = RecurrentSequential(return_sequences=True, name="merged_left_rnn", trainable=True)
    rnn_left_layer.add(NormalLSTMCell(hidden_state_size * 2, input_dim=emb_size, settings=settings))
    left_rnn_y = rnn_left_layer(left_embed)
    left_rnn_h = Lambda(lambda x: x[:, :, :hidden_state_size])(left_rnn_y)
    left_rnn_c = Lambda(lambda x: x[:, :, hidden_state_size:])(left_rnn_y)
    # # backward part
    reversed_rnn_left_layer = RecurrentSequential(return_sequences=True, name="reversed_merged_left_rnn",
                                                  trainable=True)
    reversed_rnn_left_layer.add(
        NormalLSTMCell(hidden_state_size * 2, input_dim=emb_size, settings=settings))
    reversed_left_embed = Lambda(lambda x: K.reverse(x, axes=1))(left_embed)
    reversed_left_rnn_y = reversed_rnn_left_layer(reversed_left_embed)
    backward_left_rnn_y = Lambda(lambda x: K.reverse(x, axes=1))(reversed_left_rnn_y)
    backward_left_rnn_h = Lambda(lambda x: x[:, :, :hidden_state_size])(backward_left_rnn_y)
    backward_left_rnn_c = Lambda(lambda x: x[:, :, hidden_state_size:])(backward_left_rnn_y)

    merged_left_rnn_h = concatenate([left_rnn_h, backward_left_rnn_h])
    merged_left_rnn_c = concatenate([left_rnn_c, backward_left_rnn_c])

    reversed_merged_left_rnn_h = Lambda(lambda x: K.reverse(x, axes=1))(merged_left_rnn_h)
    reversed_merged_left_rnn_c = Lambda(lambda x: K.reverse(x, axes=1))(merged_left_rnn_c)

    settings.context_size = K.int_shape(merged_left_rnn_h)[-1]
    context_size = settings.context_size

    def flatten_and_repeat(x):
        x = K.batch_flatten(x)
        x = K.expand_dims(x, axis=1)
        x = K.tile(x, [1, seq_len, 1])
        x = K.reshape(x, (-1, seq_len, seq_len * context_size))
        return x

    merged_left_rnn_h = Lambda(flatten_and_repeat)(merged_left_rnn_h)
    merged_left_rnn_c = Lambda(flatten_and_repeat)(merged_left_rnn_c)

    reversed_merged_left_rnn_h = Lambda(flatten_and_repeat)(reversed_merged_left_rnn_h)
    reversed_merged_left_rnn_c = Lambda(flatten_and_repeat)(reversed_merged_left_rnn_c)

    # right(specific domain) part
    # # forward part
    rnn_transfer = RecurrentSequential(return_sequences=True, name="right_rnn")
    rnn_transfer.add(ARTTransferCell(hidden_state_size * 2 + seq_len * 2,
                                     input_dim=emb_size + context_size * 2 * seq_len + seq_len,
                                     settings=settings))
    right_rnn_y = rnn_transfer(concatenate([merged_left_rnn_h, merged_left_rnn_c, right_embed, input_mask_tile]))
    right_rnn_y1 = Lambda(lambda x: x[:, :, :hidden_state_size])(right_rnn_y)
    h_attentions = Lambda(lambda x: x[:, :, hidden_state_size * 2: hidden_state_size * 2 + seq_len])(right_rnn_y)
    c_attentions = Lambda(lambda x: x[:, :, hidden_state_size * 2 + seq_len:])(right_rnn_y)
    # # backward part
    reversed_rnn_transfer = RecurrentSequential(return_sequences=True, name="reversed_right_rnn")
    reversed_right_embed = Lambda(lambda x: K.reverse(x, axes=1))(right_embed)
    reversed_rnn_transfer.add(ARTTransferCell(hidden_state_size * 2 + seq_len * 2,
                                              input_dim=emb_size + context_size * 2 * seq_len + seq_len,
                                              settings=settings))
    reversed_right_rnn_y = reversed_rnn_transfer(
        concatenate([reversed_merged_left_rnn_h, reversed_merged_left_rnn_c, reversed_right_embed,
                     reversed_input_mask_tile]))
    backward_right_rnn_y = Lambda(lambda x: K.reverse(x, axes=1))(reversed_right_rnn_y)
    backward_right_rnn_y1 = Lambda(lambda x: x[:, :, :hidden_state_size])(backward_right_rnn_y)
    backward_h_attentions = Lambda(lambda x: x[:, :, hidden_state_size * 2: hidden_state_size * 2 + seq_len])(
        backward_right_rnn_y)
    backward_c_attentions = Lambda(lambda x: x[:, :, hidden_state_size * 2 + seq_len:])(backward_right_rnn_y)

    merged_right_rnn_y1 = concatenate([right_rnn_y1, backward_right_rnn_y1])

    # merged_right_rnn_y1 = scaled_dot_product_attention_encoder(merged_right_rnn_y1, input_mask, settings)

    return merged_right_rnn_y1, h_attentions, c_attentions, backward_h_attentions, backward_c_attentions


# bidirectional attentive recurrent transfer learning
def bidirectional_art_lstm_encoder(left_embed, right_embed, input_mask, settings):
    seq_len = settings.seq_len
    hidden_state_size = settings.hidden_state_size
    emb_size = K.int_shape(left_embed)[-1]
    settings.emb_size = emb_size

    # backward part
    reversed_left_embed = Lambda(lambda x: K.reverse(x, axes=1))(left_embed)

    # left(open domain) part
    # # forward part
    rnn_left_layer = RecurrentSequential(return_sequences=True, name="merged_left_rnn", trainable=True)
    rnn_left_layer.add(NormalLSTMCell(hidden_state_size * 2, input_dim=emb_size, settings=settings))
    left_rnn_y = rnn_left_layer(left_embed)
    left_rnn_h = Lambda(lambda x: x[:, :, :hidden_state_size])(left_rnn_y)
    left_rnn_c = Lambda(lambda x: x[:, :, hidden_state_size:])(left_rnn_y)
    # # backward part
    reversed_input_mask = Lambda(lambda x: K.reverse(x, axes=1))(input_mask)
    reversed_rnn_left_layer = RecurrentSequential(return_sequences=True, name="reversed_merged_left_rnn",
                                                  trainable=True)
    reversed_rnn_left_layer.add(
        NormalLSTMCell(hidden_state_size * 2, input_dim=emb_size, settings=settings))
    reversed_left_rnn_y = reversed_rnn_left_layer(reversed_left_embed)
    reversed_left_rnn_h = Lambda(lambda x: x[:, :, :hidden_state_size])(reversed_left_rnn_y)
    reversed_left_rnn_c = Lambda(lambda x: x[:, :, hidden_state_size:])(reversed_left_rnn_y)

    backward_left_rnn_h = Lambda(lambda x: K.reverse(x, axes=1))(reversed_left_rnn_h)

    merged_left_rnn_y1 = concatenate([left_rnn_h, backward_left_rnn_h])

    # right(specific domain) part
    # # forward
    art_cell = ARTLSTMCell(units=hidden_state_size, h_context=left_rnn_h, c_context=left_rnn_c, context_mask=input_mask,
                           settings=settings)
    aligned_hs = left_rnn_h
    aligned_cs = left_rnn_c
    right_rnn_y1 = RNN(art_cell, return_sequences=True)(concatenate([right_embed, aligned_hs, aligned_cs]))
    # # backward
    re_art_cell = ARTLSTMCell(units=hidden_state_size, h_context=reversed_left_rnn_h, c_context=reversed_left_rnn_c,
                              context_mask=reversed_input_mask, settings=settings)
    reversed_right_embed = Lambda(lambda x: K.reverse(x, axes=1))(right_embed)
    reversed_aligned_hs = reversed_left_rnn_h
    reversed_aligned_cs = reversed_left_rnn_c
    reversed_right_rnn_y1 = RNN(re_art_cell, return_sequences=True) \
        (concatenate([reversed_right_embed, reversed_aligned_hs, reversed_aligned_cs]))
    backward_right_rnn_y1 = Lambda(lambda x: K.reverse(x, axes=1))(reversed_right_rnn_y1)
    merged_right_rnn_y1 = concatenate([right_rnn_y1, backward_right_rnn_y1])

    return merged_left_rnn_y1, merged_right_rnn_y1


# bidirectional attentive recurrent transfer learning gru version
def bidirectional_art_gru_encoder(left_embed, right_embed, input_mask, settings):
    seq_len = settings.seq_len
    hidden_state_size = settings.hidden_state_size
    emb_size = K.int_shape(left_embed)[-1]
    settings.emb_size = emb_size

    # backward part
    reversed_left_embed = Lambda(lambda x: K.reverse(x, axes=1))(left_embed)

    # left(open domain) part
    # # forward part
    rnn_left_layer = GRU(units=hidden_state_size, return_sequences=True, name="merged_left_gru")
    left_rnn_y = rnn_left_layer(left_embed)
    left_rnn_h = left_rnn_y
    # # backward part
    reversed_input_mask = Lambda(lambda x: K.reverse(x, axes=1))(input_mask)
    reversed_rnn_left_layer = GRU(units=hidden_state_size, return_sequences=True, name="reversed_merged_left_gru")
    reversed_left_rnn_y = reversed_rnn_left_layer(reversed_left_embed)
    reversed_left_rnn_h = reversed_left_rnn_y

    backward_left_rnn_h = Lambda(lambda x: K.reverse(x, axes=1))(reversed_left_rnn_h)

    merged_left_rnn_y1 = concatenate([left_rnn_h, backward_left_rnn_h])

    # right(specific domain) part
    # # forward
    art_cell = ARTGRUCell(units=hidden_state_size, h_context=left_rnn_h, context_mask=input_mask,
                          settings=settings)
    aligned_hs = left_rnn_h
    right_rnn_y1 = RNN(art_cell, return_sequences=True)(concatenate([right_embed, aligned_hs]))
    # # backward
    re_art_cell = ARTGRUCell(units=hidden_state_size, h_context=reversed_left_rnn_h,
                             context_mask=reversed_input_mask, settings=settings)
    reversed_right_embed = Lambda(lambda x: K.reverse(x, axes=1))(right_embed)
    reversed_aligned_hs = reversed_left_rnn_h
    reversed_right_rnn_y1 = RNN(re_art_cell, return_sequences=True) \
        (concatenate([reversed_right_embed, reversed_aligned_hs]))
    backward_right_rnn_y1 = Lambda(lambda x: K.reverse(x, axes=1))(reversed_right_rnn_y1)
    merged_right_rnn_y1 = concatenate([right_rnn_y1, backward_right_rnn_y1])

    return merged_left_rnn_y1, merged_right_rnn_y1
