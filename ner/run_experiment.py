import argparse
import sys
sys.path.append('..')
from ner.experiments import *

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--experiment', '-e', type=type(""), default=None)
parser.add_argument('--domain', '-d', type=type(""), default=None)
parser.add_argument('--rates', '-r', type=float, default=0.1)
parser.add_argument('--source_domain', '-s', type=type(""))
parser.add_argument('--target_domain', '-t', type=type(""))
args = parser.parse_args()

experiment = str(args.experiment).strip()
domain = str(args.domain).strip()
source_domain = str(args.source_domain).strip()
target_domain = str(args.target_domain).strip()
labeling_rates = args.rates

available_experiments = ['bi_lstm_no_transfer',
                         'bi_art_lstm', ]


if experiment in available_experiments:
    fileName = experiment + '_result.txt'
    with open(fileName, 'a', encoding='utf-8') as f:
        result = ""
        if experiment == 'bi_lstm_no_transfer':
            result = bi_lstm_no_transfer(domain, labeling_rates)
        elif experiment == 'bi_art_lstm':
            result = bi_art_lstm_transfer(source_domain, target_domain, labeling_rates)
        f.write(result+'\n')
        print(result)
else:
    print('experiments unmatched!')
