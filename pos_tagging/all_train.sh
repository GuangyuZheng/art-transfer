python merge_data.py
python run_experiment.py --experiment bi_lstm_no_transfer -d ptb -r 1
python run_experiment.py --experiment bi_art_lstm --s ptb -t twitter -r 0.1
python run_experiment.py --experiment bi_art_lstm --s ptb -t twitter -r 0.01

