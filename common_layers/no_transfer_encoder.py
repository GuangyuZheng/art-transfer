import sys

sys.path.append("..")
from custom_rnn.lstm_cell import *
from recurrentshop import RecurrentSequential
from lambda_utilities.masked_function import *


# bidirectional LSTM multiple layers
def multi_layers_bidirectional_lstm_no_transfer_encoder(embed, settings, layer_num):
    seq_len = settings.seq_len
    hidden_state_size = settings.hidden_state_size
    emb_size = K.int_shape(embed)[-1]
    settings.emb_size = emb_size

    rnn_input = embed

    for i in range(1, layer_num+1):
        if i == 1:
            input_dim = emb_size
        else:
            input_dim = hidden_state_size * 2
        # forward layer
        rnn = RecurrentSequential(return_sequences=True, name="left_rnn_" + str(i))
        rnn.add(NormalLSTMCell(hidden_state_size * 2, input_dim=input_dim, settings=settings))
        rnn_y = rnn(rnn_input)
        rnn_y1 = Lambda(lambda x: x[:, :, :hidden_state_size])(rnn_y)

        # backword layer
        reversed_rnn_input = Lambda(lambda x: K.reverse(x, axes=1))(rnn_input)
        reversed_rnn = RecurrentSequential(return_sequences=True, name="reversed_left_rnn_" + str(i))
        reversed_rnn.add(NormalLSTMCell(hidden_state_size * 2, input_dim=input_dim, settings=settings))
        reverse_rnn_y = reversed_rnn(reversed_rnn_input)  # None, seq_len, output_dim
        reverse_rnn_y1 = Lambda(lambda x: x[:, :, :hidden_state_size])(reverse_rnn_y)
        backward_rnn_y1 = Lambda(lambda x: K.reverse(x, axes=1))(reverse_rnn_y1)
        merged_rnn_y1 = concatenate([rnn_y1, backward_rnn_y1])  # None, seq_len, hidden_state_size * 2
        # merged_rnn_y1 = Dropout(0.2)(merged_rnn_y1)
        rnn_input = merged_rnn_y1

    return rnn_input


# bidirectional gru multiple layers
def multi_layers_bidirectional_gru_no_transfer_encoder(embed, settings, layer_num):
    seq_len = settings.seq_len
    hidden_state_size = settings.hidden_state_size
    emb_size = K.int_shape(embed)[-1]
    settings.emb_size = emb_size

    rnn_input = embed

    for i in range(1, layer_num+1):
        # forward layer
        rnn = GRU(units=hidden_state_size, return_sequences=True, name="left_gru_" + str(i))
        rnn_y = rnn(rnn_input)
        rnn_y1 = Lambda(lambda x: x[:, :, :hidden_state_size])(rnn_y)

        # backword layer
        reversed_rnn_input = Lambda(lambda x: K.reverse(x, axes=1))(rnn_input)
        reversed_rnn = GRU(units=hidden_state_size, return_sequences=True,  name="reversed_left_gru_" + str(i))
        reverse_rnn_y = reversed_rnn(reversed_rnn_input)  # None, seq_len, output_dim
        reverse_rnn_y1 = Lambda(lambda x: x[:, :, :hidden_state_size])(reverse_rnn_y)
        backward_rnn_y1 = Lambda(lambda x: K.reverse(x, axes=1))(reverse_rnn_y1)
        merged_rnn_y1 = concatenate([rnn_y1, backward_rnn_y1])  # None, seq_len, hidden_state_size * 2
        # merged_rnn_y1 = Dropout(0.2)(merged_rnn_y1)
        rnn_input = merged_rnn_y1

    return rnn_input