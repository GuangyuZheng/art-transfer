class Settings(object):
    def __init__(self):
        self.seq_len = 250
        self.w_emb_size = 300
        self.hidden_state_size = 150
        self.context_size = 150
        self.dropout_rate = 0.5
        self.epochs = 15
        self.bias = False
        self.domains = ['books',
                        'dvd',
                        'electronics',
                        'kitchen', ]
        self.available_experiments = ['bi_lstm_no_transfer',
                                      'bi_art_lstm',
                                      'bi_art_lstm_v2',
                                      'rest_to_one_bi_lstm_no_transfer',
                                      'rest_to_one_bi_art_lstm', ]