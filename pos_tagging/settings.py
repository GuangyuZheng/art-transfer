class Settings(object):
    def __init__(self):
        self.emb_size = 50
        self.ch_emb_size = 50
        self.hidden_state_size = 300
        self.context_size = 300
        self.epochs = 10
        self.batch_size = 128
        self.char_encoding = 'cnn'
        self.bias = True
        self.use_char_embed = True


class TwitterSettings(Settings):
    def __init__(self):
        Settings.__init__(self)
        self.seq_len = 40
        self.max_char_len = 30
        self.epochs = 100
        self.label_index = ['PRP$', 'VBG', 'VBD', 'VBN', 'HT', 'POS', "''", 'VBP', 'WDT', 'USR', 'JJ',
                            'WP', 'VBZ', 'DT', 'RT', 'NONE', 'RP', 'VPP', 'NN', 'TO', ')', '(', 'FW', ',', '.', 'CC',
                            'PRP', 'RB', 'TD', ':', 'NNS', 'NNP', 'VB', 'WRB', 'URL', 'LS', 'PDT', 'RBS', 'RBR', 'O',
                            'CD', 'EX', 'IN', 'MD', 'NNPS', 'JJS', 'JJR', 'SYM', 'UH']


class PTBSettings(Settings):
    def __init__(self):
        Settings.__init__(self)
        self.seq_len = 251
        self.max_char_len = 17
        self.epochs = 40
        self.label_index = ['PRP$', 'VBG', 'VBD', '``', 'VBN', 'POS', "''", 'VBP', 'WDT', 'JJ',
                            'WP', 'VBZ', 'DT', '#', 'RP', '$', 'NN', 'FW', ',', '.', 'TO', 'PRP', 'RB', '-LRB-',
                            ':', 'NNS', 'NNP', 'VB', 'WRB', 'CC', 'LS', 'PDT', 'RBS', 'RBR', 'CD', 'EX', 'IN', 'WP$',
                            'MD', 'NNPS', '-RRB-', 'JJS', 'JJR', 'SYM', 'UH']
