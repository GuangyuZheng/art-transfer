import sys
sys.path.append('..')
from sentiment_analysis.models import *
from sentiment_analysis.settings import Settings
from utils import *
import numpy as np

settings = Settings()
vec = np.load(os.path.join(os.getcwd(), 'dataset', 'AMAZON', 'vec.npy'))
settings.vec = vec
data_prefix = os.path.join(os.getcwd(), 'dataset', 'AMAZON')

model_directory = os.path.join(os.getcwd(), 'model')
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

model_directory = os.path.join(os.getcwd(), 'model', 'blank')
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

model_directory = os.path.join(os.getcwd(), 'model', 'tmp')
if not os.path.exists(model_directory):
    os.makedirs(model_directory)


def rest_to_one_bi_lstm_no_transfer(domain, layer_num=1, try_times=5):
    print("bi lstm rest to one no transfer: " + domain)
    model_directory = os.path.join(os.getcwd(), 'model')
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    train = False
    model = multi_layers_bilstm_no_transfer_model(settings, layer_num=layer_num)
    # model.compile(loss='binary_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    model_path = os.path.join(model_directory, 'bi_lstm_rest_to_one_' + domain + '.h5')
    blank_model_path = os.path.join(model_directory, 'blank', 'bi_lstm_rest_to_one_' + domain + '.h5')
    tmp_model_path = os.path.join(model_directory, 'tmp', 'bi_lstm_rest_to_one_' + domain + '.h5')
    model.save_weights(blank_model_path, overwrite=True)

    trainX = None
    trainY = None
    validX = None
    validY = None
    for d in settings.domains:
        if d == domain:
            continue
        else:
            if trainX is None:
                trainX = np.load(os.path.join(data_prefix, d + '_trainX.npy'))
                trainY = np.load(os.path.join(data_prefix, d + '_trainY.npy'))
                validX = np.load(os.path.join(data_prefix, d + '_validX.npy'))
                validY = np.load(os.path.join(data_prefix, d + '_validY.npy'))
            else:
                d_trainX = np.load(os.path.join(data_prefix, d + '_trainX.npy'))
                d_trainY = np.load(os.path.join(data_prefix, d + '_trainY.npy'))
                d_validX = np.load(os.path.join(data_prefix, d + '_validX.npy'))
                d_validY = np.load(os.path.join(data_prefix, d + '_validY.npy'))
                trainX = np.concatenate((trainX, d_trainX), axis=0)
                trainY = np.concatenate((trainY, d_trainY), axis=0)
                validX = np.concatenate((validX, d_validX), axis=0)
                validY = np.concatenate((validY, d_validY), axis=0)
    testX = np.load(os.path.join(data_prefix, domain + '_testX.npy'))
    testY = np.load(os.path.join(data_prefix, domain + '_testY.npy'))

    if train or (os.path.isfile(model_path) is False):
        maxAcc = 0
        for i in range(try_times):
            print('time ' + str(i))
            minLoss = 10000
            model.load_weights(blank_model_path)
            model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
            for k in range(0, settings.epochs):
                r = model.fit(trainX, trainY, verbose=1, epochs=1, batch_size=128, validation_data=(validX, validY))
                print("evaluation, round:", k, "  ", domain)
                # temPWA = sentiment_analysis_show_acc(model, testX, testY)
                if r.history['val_loss'][0] < minLoss:
                    model.save_weights(tmp_model_path, overwrite=True)
                    minLoss = r.history['val_loss'][0]
            model.load_weights(tmp_model_path)
            temPWA = sentiment_analysis_show_acc(model, testX, testY)
            if temPWA > maxAcc:
                model.save_weights(model_path, overwrite=True)
                maxAcc = temPWA
                print("update model")
    model.load_weights(model_path)

    maxAcc = sentiment_analysis_show_acc(model, testX, testY)
    result = 'bi lstm rest to one no transfer: ' + domain + ' ' + str(maxAcc)
    return result


def bi_lstm_no_transfer(domain, experiment, layer_num=1, try_times=5):
    print(experiment + ": " + domain)
    model_directory = os.path.join(os.getcwd(), 'model')
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    train = True

    model = multi_layers_bilstm_no_transfer_model(settings, layer_num=layer_num)

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    model_path = os.path.join(model_directory, experiment+'_'+domain+'.h5')
    blank_model_path = os.path.join(model_directory, 'blank', experiment+'_'+domain+'.h5')
    tmp_model_path = os.path.join(model_directory, 'tmp', experiment+'_'+domain+'.h5')
    model.save_weights(blank_model_path, overwrite=True)

    trainX = np.load(os.path.join(data_prefix, domain + '_trainX.npy'))
    trainY = np.load(os.path.join(data_prefix, domain + '_trainY.npy'))
    validX = np.load(os.path.join(data_prefix, domain + '_validX.npy'))
    validY = np.load(os.path.join(data_prefix, domain + '_validY.npy'))
    testX = np.load(os.path.join(data_prefix, domain + '_testX.npy'))
    testY = np.load(os.path.join(data_prefix, domain + '_testY.npy'))

    test_input = [testX]

    if train or (os.path.isfile(model_path) is False):
        maxAcc = 0
        for i in range(try_times):
            print('time ' + str(i))
            minLoss = 10000
            model.load_weights(blank_model_path)
            model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
            for k in range(0, settings.epochs):
                r = model.fit(trainX, trainY, verbose=1, epochs=1, batch_size=64,
                              validation_data=(validX, validY))
                print("evaluation, round:", k, "  ", domain)
                # temPWA = sentiment_analysis_show_acc(model, test_input, testY)
                if r.history['val_loss'][0] < minLoss:
                    model.save_weights(tmp_model_path, overwrite=True)
                    minLoss = r.history['val_loss'][0]
            model.load_weights(tmp_model_path)
            temPWA = sentiment_analysis_show_acc(model, test_input, testY)
            if temPWA > maxAcc:
                model.save_weights(model_path, overwrite=True)
                maxAcc = temPWA
                print("update model")
            print("maxAcc: " + str(maxAcc))
    model.load_weights(model_path)

    maxAcc = sentiment_analysis_show_acc(model, test_input, testY)
    result = experiment + ' ' + domain + ' ' + str(maxAcc)
    return result


def rnn_source_to_target_bidirectional(source_domain, target_domain, transfer_type, try_times=5):
    print(transfer_type + ': ' + source_domain + ' to ' + target_domain)
    model_directory = os.path.join(os.getcwd(), 'model')
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    loadPreviousTargetModel = False
    train = True

    model_left = multi_layers_bilstm_no_transfer_model(settings, layer_num=1)
    model_left_path = os.path.join(model_directory, 'bi_lstm_no_transfer_' + source_domain + '.h5')
    model_merged_path = os.path.join(model_directory,
                                     transfer_type + '_' + source_domain + '_to_' + target_domain + '.h5')
    model_left.load_weights(model_left_path)
    freeze_embed = model_left.get_layer('left_embed').get_weights()
    freeze_rnn = model_left.get_layer('left_rnn_1').get_weights()
    freeze_reversed_rnn = model_left.get_layer('reversed_left_rnn_1').get_weights()

    trainX = np.load(os.path.join(data_prefix, target_domain + '_trainX.npy'))
    trainY = np.load(os.path.join(data_prefix, target_domain + '_trainY.npy'))
    validX = np.load(os.path.join(data_prefix, target_domain + '_validX.npy'))
    validY = np.load(os.path.join(data_prefix, target_domain + '_validY.npy'))
    testX = np.load(os.path.join(data_prefix, target_domain + '_testX.npy'))
    testY = np.load(os.path.join(data_prefix, target_domain + '_testY.npy'))

    train_input = [trainX]
    valid_input = [validX]
    test_input = [testX]

    batch_size = 64
    print('transfer type: ' + transfer_type)
    if transfer_type == 'bi_art_lstm':
        model_merged = bidirectional_art_lstm_model(settings)
    elif transfer_type == 'bi_art_lstm_v2':
        model_merged, h_attention_model, c_attention_model, backward_h_attention_model, backward_c_attention_model \
            = bidirectional_art_lstm_model_v2(settings)
    else:
        print("No transfer type matched!")
        exit(0)
    model_merged.summary()
    model_merged.get_layer('left_embed').set_weights(freeze_embed)
    model_merged.get_layer('merged_left_rnn').set_weights(freeze_rnn)
    model_merged.get_layer('reversed_merged_left_rnn').set_weights(freeze_reversed_rnn)

    if loadPreviousTargetModel:
        model_merged.load_weights(model_merged_path)

    blank_model_merged_path = os.path.join(model_directory,
                                           'blank',
                                           transfer_type + '_' + source_domain + '_to_' + target_domain + '.h5')
    model_merged.save_weights(blank_model_merged_path, overwrite=True)
    if train or (os.path.isfile(model_merged_path) is False):
        minLoss = 10000
        maxAcc = 0
        for i in range(try_times):
            print("time " + str(i))
            model_merged.load_weights(blank_model_merged_path)
            model_merged.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
            for k in range(0, settings.epochs):
                r = model_merged.fit(train_input, trainY, verbose=1, epochs=1, batch_size=batch_size)
                print("merged evaluation, round:", k, "  ", source_domain, "to ", target_domain)
                temPWA = sentiment_analysis_show_acc(model_merged, test_input, testY, batch_size=batch_size)
                if temPWA > maxAcc:
                    model_merged.save_weights(model_merged_path, overwrite=True)
                    maxAcc = temPWA
                print("maxAcc: " + str(maxAcc))
    model_merged.load_weights(model_merged_path)
    PWA_merged = sentiment_analysis_show_acc(model_merged, test_input, testY, batch_size=batch_size)
    result = transfer_type + ': ' + source_domain + ' to ' + target_domain + ' ' + str(PWA_merged)
    return result


def rnn_rest_to_one_transfer_bidirectional(domain, transfer_type, try_times):
    print(transfer_type + ': rest to ' + domain)
    model_directory = os.path.join(os.getcwd(), 'model')
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    loadPreviousTargetModel = False
    train = True

    model_left = multi_layers_bilstm_no_transfer_model(settings)
    model_left_path = os.path.join(model_directory, 'bi_lstm_rest_to_one_' + domain + '.h5')
    model_merged_path = os.path.join(model_directory,
                                     transfer_type + '_rest_to_' + domain + '.h5')
    model_left.load_weights(model_left_path)
    freeze_embed = model_left.get_layer('left_embed').get_weights()
    freeze_rnn = model_left.get_layer('left_rnn_1').get_weights()
    freeze_reversed_rnn = model_left.get_layer('reversed_left_rnn_1').get_weights()

    trainX = np.load(os.path.join(data_prefix, domain + '_trainX.npy'))
    trainY = np.load(os.path.join(data_prefix, domain + '_trainY.npy'))
    validX = np.load(os.path.join(data_prefix, domain + '_validX.npy'))
    validY = np.load(os.path.join(data_prefix, domain + '_validY.npy'))
    testX = np.load(os.path.join(data_prefix, domain + '_testX.npy'))
    testY = np.load(os.path.join(data_prefix, domain + '_testY.npy'))

    train_input = [trainX]
    valid_input = [validX]
    test_input = [testX]

    batch_size = 64
    epoch = settings.epochs
    print('transfer type: ' + transfer_type)
    if transfer_type == 'rest_to_one_bi_art_lstm':
        model_merged = bidirectional_art_lstm_model(settings)
    else:
        print("No transfer type matched!")
        exit(0)
    model_merged.summary()
    model_merged.get_layer('left_embed').set_weights(freeze_embed)
    model_merged.get_layer('merged_left_rnn').set_weights(freeze_rnn)
    model_merged.get_layer('reversed_merged_left_rnn').set_weights(freeze_reversed_rnn)

    if loadPreviousTargetModel:
        model_merged.load_weights(model_merged_path)

    blank_model_merged_path = os.path.join(model_directory,
                                           'blank',
                                           transfer_type + '_rest_to_' + domain + '.h5')
    model_merged.save_weights(blank_model_merged_path, overwrite=True)
    if train or (os.path.isfile(model_merged_path) is False):
        minLoss = 10000
        maxAcc = 0
        for i in range(try_times):
            print("time " + str(i))
            model_merged.load_weights(blank_model_merged_path)
            model_merged.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
            for k in range(0, epoch):
                r = model_merged.fit(train_input, trainY, verbose=1, epochs=1, batch_size=batch_size)
                print("merged evaluation, round:", k, "rest", "to ", domain)
                temPWA = sentiment_analysis_show_acc(model_merged, test_input, testY, batch_size=batch_size)
                if temPWA > maxAcc:
                    model_merged.save_weights(model_merged_path, overwrite=True)
                    maxAcc = temPWA
                print("maxAcc: " + str(maxAcc))
    model_merged.load_weights(model_merged_path)
    PWA_merged = sentiment_analysis_show_acc(model_merged, test_input, testY, batch_size=batch_size)
    result = transfer_type + ': ' + ' rest to ' + domain + ' ' + str(PWA_merged)
    return result