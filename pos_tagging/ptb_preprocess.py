import sys
sys.path.append('..')
import numpy as np
import os
from pos_tagging.settings import PTBSettings

settings = PTBSettings()
LABEL_INDEX = settings.label_index
CLASS_NUM = len(LABEL_INDEX) + 1
MAX_LEN = settings.seq_len
MAX_CHAR_LEN = settings.max_char_len


DIR = os.path.join(os.getcwd(), 'dataset')
DATA_DIR = os.path.join(DIR, 'ptb')
TRAIN_DATA = os.path.join(DATA_DIR, 'train.tsv')
DEV_DATA = os.path.join(DATA_DIR, 'dev.tsv')
TEST_DATA = os.path.join(DATA_DIR, 'test.tsv')

HASH_FILE = os.path.join(DIR, 'words.lst')
EMB_FILE = os.path.join(DIR, 'embeddings.txt')

LIST_FILE = os.path.join(DIR, 'eng.list')

RARE_WORD = False
RARE_CHAR = False

USE_DEV = True  # False
LABELING_RATE = 1.0  # 1.0


def process(word):
    word = word.lower()
    word = "".join(c if not c.isdigit() else '0' for c in word)
    return word


def create_word_index(filenames):
    word_index, word_cnt = {}, 1

    for filename in filenames:
        for line in open(filename):
            if line.strip() == '':
                continue
            word = line.strip().split()[0]
            word = process(word)
            if word in word_index:
                continue
            word_index[word] = word_cnt
            word_cnt += 1
    return word_index, word_cnt


def create_char_index(filenames):
    char_index, char_cnt = {}, 3

    for filename in filenames:
        for line in open(filename):
            if line.strip() == '':
                continue
            word = line.strip().split()[0]
            for c in word:
                if c not in char_index:
                    char_index[c] = char_cnt
                    char_cnt += 1
    return char_index, char_cnt


def cnt_line(filename):
    ret = 0
    flag = False
    for line in open(filename):
        if line.strip() == '':
            if flag:
                ret += 1
            flag = False
        else:
            flag = True
    if flag:
        ret += 1
    return ret


def read_data(filename, word_index):
    line_cnt = cnt_line(filename)
    x, y = np.zeros((line_cnt, MAX_LEN), dtype=np.int32), np.zeros((line_cnt, MAX_LEN), dtype=np.int32)
    mask = np.zeros((line_cnt, MAX_LEN), dtype=np.float32)
    i, j = 0, 0
    for line in open(filename):
        inputs = line.strip().split()
        if len(inputs) < 2:
            if j > 0:
                i, j = i + 1, 0
            continue
        word, label = inputs[0], inputs[-1]
        word = process(word)
        word_ind, label_ind = word_index[word], LABEL_INDEX.index(label)
        x[i, j] = word_ind
        y[i, j] = label_ind
        mask[i, j] = 1.0
        j += 1
    # y = process_labels(y, mask)
    return x, y, mask


def read_char_data(filename, char_index):
    line_cnt = cnt_line(filename)
    x = np.zeros((line_cnt, MAX_LEN, MAX_CHAR_LEN), dtype=np.int32)
    i, j = 0, 0
    for line in open(filename):
        if line.strip() == '':
            i += 1
            j = 0
            continue
        inputs = line.strip().split()
        label = inputs[1]
        word = inputs[0]
        for k, c in enumerate(word):
            if k + 1 >= MAX_CHAR_LEN:
                break
            x[i, j, k + 1] = char_index[c]
        x[i, j, 0] = 1
        if len(word) + 1 < MAX_CHAR_LEN:
            x[i, j, len(word) + 1] = 2
        j += 1
    return x


def read_word2embedding():
    words = []
    for line in open(HASH_FILE):
        words.append(line.strip())
    word2embedding = {}
    for i, line in enumerate(open(EMB_FILE)):
        if words[i] in word2embedding:
            continue
        inputs = line.strip().split()
        word2embedding[words[i]] = np.array([float(e) for e in inputs], dtype=np.float32)
    return word2embedding


def evaluate(py, y_, m_):
    if len(py.shape) > 1:
        py = np.argmax(py, axis=2)
    py = py.flatten()
    y, m = y_.flatten(), m_.flatten()
    acc = 1.0 * (np.array(y == py, dtype = np.int32) * m).sum() / m.sum()

    return acc
