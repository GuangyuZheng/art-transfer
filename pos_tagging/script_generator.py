import sys
sys.path.append('..')
from pos_tagging.settings import Settings


def pos_tagging_scirpt():
    settings = Settings()
    available_experiments = ['rnn_no_transfer',
                             'rnn_rest_to_one_no_transfer',
                             'art',
                             'at_tl',
                             'sdp_no_transfer',
                             'sdp_rest_to_one_no_transfer',
                             'sdp_rest_to_one_concat_transfer', ]
    for experiment in available_experiments:
        fileName = 'exec_'+experiment+'.sh'
        with open(fileName, 'w') as f:
            if 'no_transfer' in experiment:
                for domain in settings.domains:
                    f.write('python run_experiment.py --experiment '+experiment+' --domain \''+domain+'\'\n')
            else:
                target_domains = settings.domains
                for t in target_domains:
                    f.write('python run_experiment.py --experiment '+experiment+' --target_domain \''+t+'\'\n')


if __name__ == "__main__":
    pos_tagging_scirpt()
