import  sys
sys.path.append('..')
from utils import get_average_precision
from sentiment_analysis.settings import Settings


if __name__ == "__main__":
    settings = Settings()
    available_experiments = settings.available_experiments

    for experiment in available_experiments:
        if 'art' in experiment:
            print(experiment + ': ' + str(get_average_precision(experiment+'_result.txt')))
