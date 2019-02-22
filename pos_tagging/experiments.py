import sys

sys.path.append('..')
from pos_tagging.models import *
from pos_tagging.settings import *
from utils import *
import numpy as np
import os
import pos_tagging.sample as sample

source_setitngs = None
target_settings = None
source_task = None
target_task = None
source_domain = None
target_domain = None
USE_DEV = True

dataset_path = os.path.join(os.getcwd(), 'dataset')
vec = np.load(os.path.join(dataset_path, 'vec.npy'))
word_index = load_pickle(os.path.join(dataset_path, 'word2index.pickle'))
char_index = load_pickle(os.path.join(dataset_path, 'char2index.pickle'))


def convert_to_one_hot(y, settings):
    line_cnt = y.shape[0]
    LABEL_INDEX = settings.label_index
    CLASS_NUM = len(LABEL_INDEX)
    MAX_LEN = settings.seq_len
    y_one_hot = np.zeros((line_cnt, MAX_LEN, CLASS_NUM), dtype=np.int32)
    for i in range(len(y)):
        for j in range(len(y[i])):
            ind = int(y[i, j])
            y_one_hot[i, j, ind] = 1
    return y_one_hot


def init_data(domain, labeling_rate, form='target'):
    global source_setitngs, target_settings, source_task, target_task
    settings = None
    if domain == 'twitter':
        settings = TwitterSettings()
    elif domain == 'ptb':
        settings = PTBSettings()
    t = __import__(domain + '_preprocess')
    data_list = [t.TRAIN_DATA, t.DEV_DATA]
    if hasattr(t, 'TEST_DATA'):
        data_list.append(t.TEST_DATA)

    wx, y, m = t.read_data(t.TRAIN_DATA, word_index)
    x = t.read_char_data(t.TRAIN_DATA, char_index)

    if USE_DEV and hasattr(t, 'DEV_DATA'):
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

    if form == 'source':
        source_setitngs = settings
        source_task = t
    else:
        target_settings = settings
        target_task = t
    return x, wx, y, y_one_hot, m, tx, twx, ty, ty_one_hot, tm


def bi_lstm_no_transfer(domain, labeling_rate):
    x, wx, y, y_one_hot, m, tx, twx, ty, ty_one_hot, tm = init_data(domain, labeling_rate, form='target')
    print("bi rnn no transfer: " + domain)
    model_directory = os.path.join(os.getcwd(), 'model')
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    train = True
    model = bi_lstm_no_transfer_model(target_settings)
    model.summary()
    model_path = os.path.join(model_directory, 'bi_lstm_no_transfer_' + domain + '_' + str(labeling_rate) + '.h5')
    if train or (os.path.isfile(model_path) is False):
        minLoss = 10000
        maxAcc = 0
        for k in range(0, target_settings.epochs):
            r = model.fit([wx, x], y_one_hot, verbose=1, epochs=1, batch_size=target_settings.batch_size)
            if k >= 0 and (k+1) % 1 == 0:
                print("evaluation, round:", k, "  ", domain)
                py = model.predict([twx, tx], batch_size=target_settings.batch_size, verbose=1)
                acc = target_task.evaluate(py, ty, tm)
                print('acc:', acc)
                if acc > maxAcc:
                    model.save_weights(model_path, overwrite=True)
                    maxAcc = acc
    model.load_weights(model_path)

    py = model.predict([twx, tx], batch_size=target_settings.batch_size, verbose=1)
    acc = target_task.evaluate(py, ty, tm)
    result = 'bi-lstm no transfer labeling rates ' + str(labeling_rate) + ': ' + domain + ' ' + str(acc)
    return result


def bi_gru_no_transfer(domain, labeling_rate):
    x, wx, y, y_one_hot, m, tx, twx, ty, ty_one_hot, tm = init_data(domain, labeling_rate, form='target')
    print("bi rnn no transfer: " + domain)
    model_directory = os.path.join(os.getcwd(), 'model')
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    train = True
    model = bi_gru_no_transfer_model(target_settings)
    model.summary()
    model_path = os.path.join(model_directory, 'bi_gru_no_transfer_' + domain + '_' + str(labeling_rate) + '.h5')
    if train or (os.path.isfile(model_path) is False):
        minLoss = 10000
        maxAcc = 0
        for k in range(0, target_settings.epochs):
            r = model.fit([wx, x], y_one_hot, verbose=1, epochs=1, batch_size=target_settings.batch_size)
            if k >= 0 and (k+1) % 1 == 0:
                print("evaluation, round:", k, "  ", domain)
                py = model.predict([twx, tx], batch_size=target_settings.batch_size, verbose=1)
                acc = target_task.evaluate(py, ty, tm)
                print('acc:', acc)
                if acc > maxAcc:
                    model.save_weights(model_path, overwrite=True)
                    maxAcc = acc
    model.load_weights(model_path)

    py = model.predict([twx, tx], batch_size=target_settings.batch_size, verbose=1)
    acc = target_task.evaluate(py, ty, tm)
    result = 'bi-gru no transfer labeling rates ' + str(labeling_rate) + ': ' + domain + ' ' + str(acc)
    return result


def bi_art_lstm_transfer(source_domain, target_domain, labeling_rate):
    init_data(source_domain, 1, form='source')
    x, wx, y, y_one_hot, m, tx, twx, ty, ty_one_hot, tm = init_data(target_domain, labeling_rate, form='target')
    print("bi art lstm transfer: " + source_domain + ' to ' + target_domain + ' labeling rate: ' + str(labeling_rate))
    model_directory = os.path.join(os.getcwd(), 'model')
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    train = True

    model_left = bi_lstm_no_transfer_model(source_setitngs)
    model_left.summary()
    model_left_path = os.path.join(model_directory, 'bi_lstm_no_transfer_' + source_domain + '_1.0.h5')
    model_left.load_weights(model_left_path)
    freeze_embed = model_left.get_layer('left_embed').get_weights()
    freeze_rnn = model_left.get_layer('left_rnn_1').get_weights()
    freeze_reversed_rnn = model_left.get_layer('reversed_left_rnn_1').get_weights()
    freeze_char_embed = model_left.get_layer('left_char_embed').get_weights()

    model_merged = bi_art_lstm_model(target_settings)
    model_merged.summary()
    model_merged.get_layer('left_embed').set_weights(freeze_embed)
    model_merged.get_layer('merged_left_rnn').set_weights(freeze_rnn)
    model_merged.get_layer('reversed_merged_left_rnn').set_weights(freeze_reversed_rnn)
    model_merged.get_layer('left_char_embed').set_weights(freeze_char_embed)
    model_merged_path = os.path.join(model_directory, 'bi_art_lstm_' + source_domain + '_to_' + target_domain
                                     + '_' + str(labeling_rate) + '.h5')

    model_merged.load_weights(model_merged_path)
    if train or (os.path.isfile(model_merged_path) is False):
        minLoss = 10000
        maxAcc = 0
        for k in range(0, target_settings.epochs):
            r = model_merged.fit([wx, x], y_one_hot, verbose=1, epochs=1, batch_size=target_settings.batch_size)
            print("evaluation, round:", k, "  ", source_domain, 'to', target_domain)
            py = model_merged.predict([twx, tx], batch_size=target_settings.batch_size, verbose=1)
            acc = target_task.evaluate(py, ty, tm)
            if acc > maxAcc:
                model_merged.save_weights(model_merged_path, overwrite=True)
                maxAcc = acc
            print('acc:', acc, 'maxAcc:', maxAcc)
    else:
        model_merged.load_weights(model_merged_path)
        py = model_merged.predict([twx, tx], batch_size=target_settings.batch_size, verbose=1)
        maxAcc = target_task.evaluate(py, ty, tm)
    result = "bi art lstm transfer: " + source_domain + ' to ' + target_domain + ' labeling rate: ' + str(labeling_rate) \
             + ': ' + str(maxAcc)
    return result


def bi_art_gru_transfer(source_domain, target_domain, labeling_rate):
    init_data(source_domain, 1, form='source')
    x, wx, y, y_one_hot, m, tx, twx, ty, ty_one_hot, tm = init_data(target_domain, labeling_rate, form='target')
    print("bi art gru transfer: " + source_domain + ' to ' + target_domain + ' labeling rate: ' + str(labeling_rate))
    model_directory = os.path.join(os.getcwd(), 'model')
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    train = True

    model_left = bi_gru_no_transfer_model(source_setitngs)
    model_left.summary()
    model_left_path = os.path.join(model_directory, 'bi_gru_no_transfer_' + source_domain + '_1.0.h5')
    model_left.load_weights(model_left_path)
    freeze_embed = model_left.get_layer('left_embed').get_weights()
    freeze_rnn = model_left.get_layer('left_gru_1').get_weights()
    freeze_reversed_rnn = model_left.get_layer('reversed_left_gru_1').get_weights()
    freeze_char_embed = model_left.get_layer('left_char_embed').get_weights()

    model_merged = bi_art_gru_model(target_settings)
    model_merged.summary()
    model_merged.get_layer('left_embed').set_weights(freeze_embed)
    model_merged.get_layer('merged_left_gru').set_weights(freeze_rnn)
    model_merged.get_layer('reversed_merged_left_gru').set_weights(freeze_reversed_rnn)
    model_merged.get_layer('left_char_embed').set_weights(freeze_char_embed)
    model_merged_path = os.path.join(model_directory, 'bi_art_gru_' + source_domain + '_to_' + target_domain
                                     + '_' + str(labeling_rate) + '.h5')

    if train or (os.path.isfile(model_merged_path) is False):
        minLoss = 10000
        maxAcc = 0
        for k in range(0, target_settings.epochs):
            r = model_merged.fit([wx, x], y_one_hot, verbose=1, epochs=1, batch_size=target_settings.batch_size)
            print("evaluation, round:", k, "  ", source_domain, 'to', target_domain)
            py = model_merged.predict([twx, tx], batch_size=target_settings.batch_size, verbose=1)
            acc = target_task.evaluate(py, ty, tm)
            if acc > maxAcc:
                model_merged.save_weights(model_merged_path, overwrite=True)
                maxAcc = acc
            print('acc:', acc, 'maxAcc:', maxAcc)
    else:
        model_merged.load_weights(model_merged_path)
        py = model_merged.predict([twx, tx], batch_size=target_settings.batch_size, verbose=1)
        maxAcc = target_task.evaluate(py, ty, tm)
    result = "bi art gru transfer: " + source_domain + ' to ' + target_domain + ' labeling rate: ' + str(labeling_rate) \
             + ': ' + str(maxAcc)
    return result
