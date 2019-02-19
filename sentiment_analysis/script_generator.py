import sys
sys.path.append('..')
from sentiment_analysis.settings import Settings


def sentiment_analysis_scirpt():
    settings = Settings()
    available_experiments = settings.available_experiments
    for experiment in available_experiments:
        fileName = 'exec_'+experiment+'.sh'
        with open(fileName, 'w') as f:
            if 'no_transfer' in experiment or 'rest_to_one' in experiment:
                for domain in settings.domains:
                    f.write('python run_experiment.py --experiment '+experiment+' --domain \''+domain+'\'\n')
            else:
                source_domains = settings.domains
                target_domains = settings.domains
                for s in source_domains:
                    for t in target_domains:
                        if s == t:
                            continue
                        else:
                            f.write('python run_experiment.py --experiment '+experiment+' --source_domain \''+s+'\' --target_domain \''+t+'\'\n')


if __name__ == "__main__":
    sentiment_analysis_scirpt()