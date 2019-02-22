import argparse
import sys
import os
import numpy as np
sys.path.append('..')
from pos_tagging.models import bi_art_lstm_model
from pos_tagging.settings import *
from pos_tagging.experiments import convert_to_one_hot
from utils import load_pickle
import pos_tagging.sample as sample

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--source_domain', '-s', type=type(""))
parser.add_argument('--target_domain', '-t', type=type(""))
parser.add_argument('--rates', '-r', type=float, default=0.1)
args = parser.parse_args()

source_domain = str(args.source_domain).strip()
target_domain = str(args.target_domain).strip()
labeling_rates = args.rates

dataset_path = os.path.join(os.getcwd(), 'dataset')
vec = np.load(os.path.join(dataset_path, 'vec.npy'))
word_index = load_pickle(os.path.join(dataset_path, 'word2index.pickle'))
char_index = load_pickle(os.path.join(dataset_path, 'char2index.pickle'))

model_path = os.path.join(os.getcwd(), 'model', 'bi_art_lstm_'+source_domain+'_to_'+target_domain+'_'+str(labeling_rates)+'.h5')

if target_domain == 'twitter':
    settings = TwitterSettings()
else:
    settings = PTBSettings()


def init_data(domain, labeling_rate):
    t = __import__(domain + '_preprocess')
    data_list = [t.TRAIN_DATA, t.DEV_DATA]
    if hasattr(t, 'TEST_DATA'):
        data_list.append(t.TEST_DATA)

    wx, y, m = t.read_data(t.TRAIN_DATA, word_index)
    x = t.read_char_data(t.TRAIN_DATA, char_index)

    if hasattr(t, 'DEV_DATA'):
        dev_wx, dev_y, dev_m = t.read_data(t.DEV_DATA, word_index)
        wx, y, m = np.vstack((wx, dev_wx)), np.vstack((y, dev_y)), np.vstack((m, dev_m))
        dev_x = t.read_char_data(t.DEV_DATA, char_index)
        x = np.vstack((x, dev_x))
    twx, ty, tm = t.read_data(t.TEST_DATA, word_index)
    tx = t.read_char_data(t.TEST_DATA, char_index)

    if labeling_rate < 1.0:
        ind = sample.create_sample_index(labeling_rate, x.shape[0])
        x, y, wx, m = sample.sample_arrays((x, y, wx, m), ind)

    settings.vec = vec
    settings.char_cnt = len(char_index)

    y_one_hot = convert_to_one_hot(y, settings)
    ty_one_hot = convert_to_one_hot(ty, settings)

    return x, wx, y, y_one_hot, m, tx, twx, ty, ty_one_hot, tm


x, wx, y, y_one_hot, m, tx, twx, ty, ty_one_hot, tm = init_data(target_domain, labeling_rates)
model = bi_art_lstm_model(settings)
model.load_weights(model_path)

test_input = [tx, twx]
py = model.predict([twx, tx], batch_size=128, verbose=1)
target_task = __import__(target_domain + '_preprocess')
acc = target_task.evaluate(py, ty, tm)
result = source_domain + ' to ' + target_domain + ' ' + str(labeling_rates) + ' ' + str(acc)
print(result)