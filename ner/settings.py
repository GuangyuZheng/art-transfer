class Settings(object):
    def __init__(self):
        self.emb_size = 50
        self.ch_emb_size = 50
        self.hidden_state_size = 300
        self.context_size = 300
        self.epochs = 10
        self.batch_size = 128
        self.char_encoding = 'rnn'
        self.bias = True
        self.use_char_embed = True


class TwitterSettings(Settings):
    def __init__(self):
        Settings.__init__(self)
        self.CAT = ['product', 'sportsteam', 'tvshow', 'musicartist', 'movie', 'other', 'person', 'facility', 'company',
                    'geo-loc']
        self.POSITION = ['I', 'B', 'E', 'S']

        self.label_index = ['O'] + ["{}-{}".format(position, cat) for cat in self.CAT for position in self.POSITION]
        self.seq_len = 40
        self.max_char_len = 34
        self.epochs = 100


class CoNLLSettings(Settings):
    def __init__(self):
        Settings.__init__(self)
        self.CAT = ['PER', 'ORG', 'LOC', 'MISC']
        self.POSITION = ['I', 'B', 'E', 'S']

        self.label_index = ['O'] + ["{}-{}".format(position, cat) for cat in self.CAT for position in self.POSITION]
        self.seq_len = 125
        self.max_char_len = 17
        self.epochs = 100
