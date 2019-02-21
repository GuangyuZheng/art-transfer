import argparse
import sys
import os
import numpy as np
sys.path.append('..')
from sentiment_analysis.models import bidirectional_art_lstm_model_v2
from sentiment_analysis.settings import Settings
from utils import sentiment_analysis_show_acc

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--source_domain', '-s', type=type(""))
parser.add_argument('--target_domain', '-t', type=type(""))
args = parser.parse_args()

source_domain = str(args.source_domain).strip()
target_domain = str(args.target_domain).strip()

settings = Settings()
vec = np.load(os.path.join(os.getcwd(), 'dataset', 'AMAZON', 'vec.npy'))
settings.vec = vec
model = bidirectional_art_lstm_model_v2(settings)[0]
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

if source_domain == "others":
    model_path = os.path.join(os.getcwd(), 'model', 'bi_art_lstm_rest_to_'+target_domain+'.h5')
else:
    model_path = os.path.join(os.getcwd(), 'model', 'bi_art_lstm_v2_'+source_domain+'_to_'+target_domain+'.h5')

model.load_weights(model_path)

data_prefix = os.path.join(os.getcwd(), 'dataset', 'AMAZON')
testX = np.load(os.path.join(data_prefix, target_domain + '_testX.npy'))
testY = np.load(os.path.join(data_prefix, target_domain + '_testY.npy'))
test_input = [testX]
# acc = model.evaluate(test_input, testY, batch_size=64)
PWA_merged = sentiment_analysis_show_acc(model, test_input, testY, batch_size=16)
result = source_domain + ' to ' + target_domain + ' ' + str(PWA_merged)
print(result)


