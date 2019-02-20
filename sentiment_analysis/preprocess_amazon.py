import os
import nltk
import numpy as np
import random
import sys

sys.path.append('..')
from sentiment_analysis.settings import Settings

setting = Settings()
domains = setting.domains
w_emb_size = setting.w_emb_size
seq_len_threshold = setting.seq_len

amazon_prefix = os.path.join(os.getcwd(), 'origin_data', 'AMAZON')
save_prefix = os.path.join(os.getcwd(), 'dataset', 'AMAZON')
if not os.path.isdir(save_prefix):
    os.makedirs(save_prefix)
root_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
print(root_path)
# glove_path = os.path.join(root_path, 'glove', 'glove.6B.' + str(w_emb_size) + 'd.txt')
glove_path = os.path.join(root_path, 'glove', 'glove.42B.'+str(w_emb_size)+'d.txt')
BLANK = 0

word2id = {}
id2word = {}
id2vec = {}
id2vec[BLANK] = np.zeros(setting.w_emb_size, )
wid = 1


def extract_review_text(domain, polarity):
    global wid
    with open(os.path.join(amazon_prefix, domain, polarity+'.txt')) as f:
        lines = f.readlines()
        for line in lines:
            words = line.strip().split()
            for word in words:
                if word not in word2id:
                    word2id[word] = wid
                    id2word[wid] = word
                    id2vec[wid] = np.zeros(setting.w_emb_size, )
                    wid += 1


def get_id2vec_dict():
    with open(glove_path, 'r', encoding='utf-8') as f:
        data = f.readline()
        while data:
            s = data.split()
            word = s[0]
            vec = [float(x) for x in s[1:]]
            if word in word2id:
                id = word2id[word]
                id2vec[id] = vec
            data = f.readline()
    return id2vec


def shuffle_data(x, y):
    subscript = list(range(len(x)))
    random.Random(19960714).shuffle(subscript)
    shuffle_x = []
    shuffle_y = []
    for i in subscript:
        shuffle_x.append(x[i])
        shuffle_y.append(y[i])
    return shuffle_x, shuffle_y


def split_data(domain, word2id):
    positive_data_X = []
    positive_data_Y = []
    negative_data_X = []
    negative_data_Y = []
    review_prefix = os.path.join(amazon_prefix, domain)
    with open(os.path.join(review_prefix, 'positive.txt'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            x = []
            tokens = line.lower().split()
            for token in tokens:
                word_id = word2id[token]
                x.append(word_id)
            if len(x) < seq_len_threshold:
                while len(x) < seq_len_threshold:
                    x.append(BLANK)
            else:
                x = x[:seq_len_threshold]
            positive_data_X.append(x)
            positive_data_Y.append(1)
    with open(os.path.join(review_prefix, 'negative.txt'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            x = []
            tokens = line.lower().split()
            for token in tokens:
                word_id = word2id[token]
                x.append(word_id)
            if len(x) < seq_len_threshold:
                while len(x) < seq_len_threshold:
                    x.append(BLANK)
            else:
                x = x[:seq_len_threshold]
            negative_data_X.append(x)
            negative_data_Y.append(0)
    positive_data_X, positive_data_Y = shuffle_data(positive_data_X, positive_data_Y)
    negative_data_X, negative_data_Y = shuffle_data(negative_data_X, negative_data_Y)
    test_X = positive_data_X[:200] + negative_data_X[:200]
    test_Y = positive_data_Y[:200] + negative_data_Y[:200]
    valid_X = positive_data_X[200:300] + negative_data_X[200:300]
    valid_Y = positive_data_Y[200:300] + negative_data_Y[200:300]
    train_X, train_Y = shuffle_data(positive_data_X[300:] + negative_data_X[300:],
                                    positive_data_Y[300:] + negative_data_Y[300:])

    np.save(os.path.join(save_prefix, domain + '_trainX.npy'), train_X)
    np.save(os.path.join(save_prefix, domain + '_trainY.npy'), train_Y)
    np.save(os.path.join(save_prefix, domain + '_validX.npy'), valid_X)
    np.save(os.path.join(save_prefix, domain + '_validY.npy'), valid_Y)
    np.save(os.path.join(save_prefix, domain + '_testX.npy'), test_X)
    np.save(os.path.join(save_prefix, domain + '_testY.npy'), test_Y)


def save_id2vec_data(id2vec_dict):
    vec = np.zeros((len(id2vec_dict), setting.w_emb_size))
    for id in id2vec_dict:
        vec[id] = id2vec_dict[id]
    np.save(os.path.join(save_prefix, 'vec.npy'), vec)


def save_pickle(d, path):
    import pickle
    print('save pickle to', path)
    with open(path, mode='wb') as f:
        pickle.dump(d, f, True)


if __name__ == "__main__":
    for domain in domains:
        print("preprocessing " + domain + "...")
        for polarity in ['positive', 'negative']:
            extract_review_text(domain, polarity)
        print(len(word2id), len(id2vec))
    save_pickle(word2id, os.path.join(save_prefix, 'word2id.pickle'))
    save_pickle(id2word, os.path.join(save_prefix, 'id2word.pickle'))

    i2v = get_id2vec_dict()
    save_id2vec_data(i2v)
    for domain in domains:
        print("generating dataset for " + domain + "...")
        split_data(domain, word2id)
    print("vocab size: " + str(len(word2id)))
    print("vec size: " + str(len(i2v)))
