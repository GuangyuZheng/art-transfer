import sys
sys.path.append('..')
from utils import save_pickle
import os
import numpy as np
from ner.settings import Settings

TASKS = ['twitter',
         'conll', ]
save_prefix = os.path.join(os.getcwd(), 'dataset')


def set_embedding(word2embedding, word_index, settings):
    vec = np.zeros((len(word_index) + 1, settings.emb_size))
    emb_hit_cnt = 0
    for word, embedding in word2embedding.items():
        if word not in word_index:
            continue
        emb_hit_cnt += 1
        ind = word_index[word]
        vec[ind, :embedding.shape[0]] = embedding
    return vec


if __name__ == "__main__":
    char_set, word_set = set(), set()
    word2embedding = None
    settings = Settings()
    for task in TASKS:
        t = __import__(task+'_preprocess')
        data_list = [t.TRAIN_DATA, t.DEV_DATA]
        if hasattr(t, 'TEST_DATA'):
            data_list.append(t.TEST_DATA)
        char_index, _ = t.create_char_index(data_list)
        for k, v in char_index.items():
            char_set.add(k)
        word_index, _ = t.create_word_index(data_list)
        for k, v in word_index.items():
            word_set.add(k)
        word2embedding = t.read_word2embedding()
    char_index, char_cnt = {}, 3
    for char in char_set:
        char_index[char] = char_cnt
        char_cnt += 1
    word_index, word_cnt = {}, 1
    for word in word_set:
        word_index[word] = word_cnt
        word_cnt += 1
    save_pickle(word_index, os.path.join(save_prefix, 'word2index.pickle'))
    save_pickle(char_index, os.path.join(save_prefix, 'char2index.pickle'))
    vec = set_embedding(word2embedding, word_index, settings)
    np.save(os.path.join(save_prefix, 'vec.npy'), vec)
    print("word cnt:", str(len(word_index)))
    print("char cht:", str(len(char_index)))