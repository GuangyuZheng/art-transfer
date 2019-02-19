import sys

sys.path.append('..')
import numpy as np
import os
from ner.settings import TwitterSettings

settings = TwitterSettings()
LABEL_INDEX = settings.label_index
CLASS_NUM = len(LABEL_INDEX)
MAX_LEN = settings.seq_len
MAX_CHAR_LEN = settings.max_char_len

DIR = os.path.join(os.getcwd(), 'dataset')
DATA_DIR = os.path.join(DIR, 'twitter')
TRAIN_DATA = os.path.join(DATA_DIR, 'ner.train.txt')
DEV_DATA = os.path.join(DATA_DIR, 'ner.dev.txt')
TEST_DATA = os.path.join(DATA_DIR, 'ner.test.txt')

HASH_FILE = os.path.join(DIR, 'words.lst')
EMB_FILE = os.path.join(DIR, 'embeddings.txt')

LIST_FILE = os.path.join(DIR, 'eng.list')

RARE_WORD = False
RARE_CHAR = False

USE_DEV = True  # False
LABELING_RATE = 0.1  # 1.0


def label_decode(label):
    if label == 'O':
        return 'O', 'O'
    s = label.split('-')
    return s[0], "-".join(s[1:])


def process_labels(y, m):
    def old_match(y_prev, y_next):
        l_prev, l_next = LABEL_INDEX[y_prev], LABEL_INDEX[y_next]
        c1_prev, c2_prev = label_decode(l_prev)
        c1_next, c2_next = label_decode(l_next)
        if c2_prev != c2_next:
            return False
        if c1_next == 'B':
            return False
        return True

    ret = np.zeros(y.shape, dtype=y.dtype)
    for i in range(y.shape[0]):
        j = 0
        while j < y.shape[1]:
            if m[i, j] == 0:
                break
            if y[i, j] == 0:
                j += 1
                continue
            k = j + 1
            while k < y.shape[1] and old_match(y[i, j], y[i, k]):
                k += 1
            _, c2 = label_decode(LABEL_INDEX[y[i, j]])
            if k - j == 1:
                ret[i, j] = LABEL_INDEX.index('S-{}'.format(c2))
            else:
                ret[i, j] = LABEL_INDEX.index('B-{}'.format(c2))
                ret[i, k - 1] = LABEL_INDEX.index('E-{}'.format(c2))
                for p in range(j + 1, k - 1):
                    ret[i, p] = LABEL_INDEX.index('I-{}'.format(c2))
            j = k
    return ret


def process(word):
    word = word.lower()
    word = "".join(c if not c.isdigit() else '0' for c in word)
    return word


def cnt_line(filename):
    line_cnt = 0
    cur_flag = False
    for line in open(filename):
        if line.strip() == '':
            if cur_flag:
                line_cnt += 1
            cur_flag = False
            continue
        cur_flag = True
    if cur_flag: line_cnt += 1
    return line_cnt


def create_word_index(filenames):
    word_index, word_cnt = {}, 1

    for sign, filename in enumerate(filenames):
        for line in open(filename):
            if line.strip() == '': continue
            word = line.strip().split()[0]
            word = process(word)
            if word in word_index: continue
            word_index[word] = word_cnt
            word_cnt += 1
    return word_index, word_cnt


def create_char_index(filenames):
    char_index, char_cnt = {}, 3

    for filename in filenames:
        for line in open(filename):
            if line.strip() == '': continue
            word = line.strip().split()[0]
            for c in word:
                if c not in char_index:
                    char_index[c] = char_cnt
                    char_cnt += 1
    return char_index, char_cnt


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
        word, label = inputs[0], inputs[1]
        word = process(word)
        word_ind, label_ind = word_index[word], LABEL_INDEX.index(label)
        x[i, j] = word_ind
        y[i, j] = label_ind
        mask[i, j] = 1.0
        j += 1
    y = process_labels(y, mask)
    return x, y, mask


def read_char_data(filename, char_index):
    line_cnt = cnt_line(filename)
    x = np.zeros((line_cnt, MAX_LEN, MAX_CHAR_LEN), dtype=np.int32)
    mask = np.zeros((line_cnt, MAX_LEN, MAX_CHAR_LEN), dtype=np.float32)
    i, j = 0, 0
    for line in open(filename):
        inputs = line.strip().split()
        if len(inputs) < 2:
            if j > 0:
                i, j = i + 1, 0
            continue
        word, label = inputs[0], inputs[1]
        for k, c in enumerate(word):
            if k + 1 >= MAX_CHAR_LEN:
                break
            x[i, j, k + 1] = char_index[c]
            mask[i, j, k + 1] = 1.0
        x[i, j, 0] = 1
        mask[i, j, 0] = 1.0
        if len(word) + 1 < MAX_CHAR_LEN:
            x[i, j, len(word) + 1] = 2
            mask[i, j, len(word) + 1] = 1.0
        j += 1
    return x


def extract_ent(y, m):
    def new_match(y_prev, y_next):
        l_prev, l_next = LABEL_INDEX[y_prev], LABEL_INDEX[y_next]
        c1_prev, c2_prev = label_decode(l_prev)
        c1_next, c2_next = label_decode(l_next)
        if c2_prev != c2_next: return False
        if c1_next not in ['I', 'E']:
            return False
        return True

    ret = set()
    i = 0
    while i < y.shape[0]:
        if m[i] == 0:
            i += 1
            continue
        c1, c2 = label_decode(LABEL_INDEX[y[i]])
        if c1 in ['O', 'I', 'E']:
            i += 1
            continue
        if c1 == 'S':
            ret.add((i, i + 1, c2))
            i += 1
            continue
        j = i + 1
        end = False
        while m[j] != 0 and not end and new_match(y[i], y[j]):
            ic1, ic2 = label_decode(LABEL_INDEX[y[j]])
            if ic1 == 'E': end = True
            j += 1
        if not end:
            i += 1
            continue
        ret.add((i, j, c2))
        i = j
    return ret


def evaluate(py, y_, m_, full=False):
    if len(py.shape) > 1:
        py = np.argmax(py, axis=2)
    # print(py.shape)
    py = py.flatten()
    y, m = y_.flatten(), m_.flatten()
    # print(py.shape)
    # print(y.shape)
    acc = 1.0 * (np.array(y == py, dtype=np.int32) * m).sum() / m.sum()
    # print(acc)
    tp, fp, fn = 0, 0, 0
    p_ent = extract_ent(py, m)
    y_ent = extract_ent(y, m)
    for ent in p_ent:
        if ent in y_ent:
            tp += 1
        else:
            fp += 1
    for ent in y_ent:
        if ent not in p_ent:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    if full:
        return acc, f1, prec, recall
    return acc, f1


def read_word2embedding():
    words = []
    for line in open(HASH_FILE):
        words.append(line.strip())
    word2embedding = {}
    for i, line in enumerate(open(EMB_FILE)):
        inputs = line.strip().split()
        word2embedding[words[i]] = np.array([float(e) for e in inputs], dtype=np.float32)
    return word2embedding
