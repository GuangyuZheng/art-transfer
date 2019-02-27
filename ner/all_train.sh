python merge_data.py
python run_experiment.py --experiment bi_lstm_no_transfer -d conll -r 1
python run_experiment.py --experiment bi_lstm_no_transfer -d twitter -r 1
python run_experiment.py --experiment bi_art_lstm --s conll -t twitter -r 0.1
python run_experiment.py --experiment bi_art_lstm --s twitter -t conll -r 0.01

