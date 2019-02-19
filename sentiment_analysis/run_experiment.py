import argparse
import sys
sys.path.append('..')
from sentiment_analysis.experiments import *
from sentiment_analysis.settings import Settings

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--experiment', '-e', type=type(""), default=None)
parser.add_argument('--domain', '-d', type=type(""), default=None)
parser.add_argument('--source_domain', '-s', type=type(""))
parser.add_argument('--target_domain', '-t', type=type(""))
parser.add_argument('--try_times', '-tt', type=int, default=5)
args = parser.parse_args()

experiment = str(args.experiment).strip()
domain = str(args.domain).strip()
source_domain = str(args.source_domain).strip()
target_domain = str(args.target_domain).strip()
try_times = args.try_times

settings = Settings()
available_experiments = settings.available_experiments

if experiment in available_experiments:
    result = ""
    if experiment == 'bi_lstm_no_transfer':
        result = bi_lstm_no_transfer(domain, experiment, layer_num=1, try_times=try_times)
    elif experiment == 'bi_art_lstm' or experiment == 'bi_art_lstm_v2':
        result = rnn_source_to_target_bidirectional(source_domain, target_domain, experiment, try_times=try_times)
    elif experiment == 'rest_to_one_bi_lstm_no_transfer':
        result = rest_to_one_bi_lstm_no_transfer(domain, try_times=try_times)
    elif experiment == 'rest_to_one_bi_art_lstm':
        result = rnn_rest_to_one_transfer_bidirectional(domain, experiment, try_times=try_times)

    if 'no_transfer' in experiment:
        print(result)
    else:
        print(result)
        fileName = experiment+'_result.txt'
        with open(fileName, 'a', encoding='utf-8') as f:
            f.write(result+'\n')
else:
    print('experiments unmatched!')
