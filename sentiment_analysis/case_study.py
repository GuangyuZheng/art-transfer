import sys
sys.path.append('..')
from sentiment_analysis.models import *
from sentiment_analysis.settings import Settings
import os
from utils import *

vec = np.load(os.path.join(os.getcwd(), 'dataset', 'AMAZON', 'vec.npy'))
data_prefix = os.path.join(os.getcwd(), 'dataset', 'AMAZON')
model_directory = os.path.join(os.getcwd(), 'model')


def get_attention_values(inputX, data, file_name):
    id2word = load_pickle('id2word.pickle')
    with open(file_name, 'w', encoding='utf-8') as f:
        for i in range(len(inputX)):
            sen_ids = inputX[i]
            n = 0
            for id in sen_ids:
                if id != 0:
                    n += 1
                else:
                    break
            f.write(str(n)+'\n')
            attention_values = data[i]
            for x in range(len(attention_values)):
                for y in range(len(attention_values[0])):
                    f.write(str(attention_values[x][y])+'\t')
                f.write('\n')
            for id in sen_ids:
                if id != 0:
                    f.write(str(id2word[id]) + '\t')
                else:
                    break
            f.write('\n')


if __name__ == "__main__":
    source_domain = 'books'
    target_domain = 'dvd'
    settings = Settings()
    vec = np.load(os.path.join(os.getcwd(), 'dataset', 'AMAZON', 'vec.npy'))
    id2ch = load_pickle('id2char.pickle')
    settings.vec = vec
    settings.char_size = len(id2ch)
    # source_to_target(source, target, 'attentive')

    model_merged, h_model, c_model, bh_model, bc_model = bidirectional_art_model(settings)

    model_merged_path = os.path.join(model_directory,
                                     'bi_art_' + source_domain + '_to_' + target_domain + '.h5')
    model_merged.load_weights(model_merged_path)
    h_model.load_weights(model_merged_path, by_name=True)
    c_model.load_weights(model_merged_path, by_name=True)
    bh_model.load_weights(model_merged_path, by_name=True)
    bc_model.load_weights(model_merged_path, by_name=True)

    trainX = np.load(os.path.join(data_prefix, target_domain + '_trainX.npy'))
    trainCX = np.load(os.path.join(data_prefix, target_domain + '_trainCX.npy'))
    trainY = np.load(os.path.join(data_prefix, target_domain + '_trainY.npy'))
    validX = np.load(os.path.join(data_prefix, target_domain + '_validX.npy'))
    validCX = np.load(os.path.join(data_prefix, target_domain + '_validCX.npy'))
    validY = np.load(os.path.join(data_prefix, target_domain + '_validY.npy'))
    testX = np.load(os.path.join(data_prefix, target_domain + '_testX.npy'))
    testCX = np.load(os.path.join(data_prefix, target_domain + '_testCX.npy'))
    testY = np.load(os.path.join(data_prefix, target_domain + '_testY.npy'))

    if settings.use_char_embed:
        test_input = [testX, testCX]
    else:
        test_input = testX

    PWA_merged = sentiment_analysis_show_acc(model_merged, test_input, testY)

    h_attentions = h_model.predict(test_input)
    c_attentions = c_model.predict(test_input)
    bh_attentions = bh_model.predict(test_input)
    bc_attetnions = bc_model.predict(test_input)

    # print(h_attentions[0][1])

    get_attention_values(testX, h_attentions, 'h_attentions.txt')
    get_attention_values(testX, c_attentions, 'c_attentions.txt')
    get_attention_values(testX, bh_attentions, 'bh_attentions.txt')
    get_attention_values(testX, bc_attetnions, 'bc_attentions.txt')