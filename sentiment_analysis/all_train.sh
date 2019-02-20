#!/bin/sh

#python preprocess_amazon.py

domains=('books' 'dvd' 'electronics' 'kitchen')
for src_domain in ${domains[@]};
do
    python run_experiment.py -e bi_lstm_no_transfer -d ${src_domain}
	for tar_domain in  ${domains[@]};
	do
		if [[ ${src_domain} != ${tar_domain} ]];
		then
			python run_experiment.py -e bi_art_lstm -s ${src_domain} -t ${tar_domain}
		fi
	done
done

for tar_domain in ${domains[@]};
do
    python run_experiment.py -e rest_to_one_bi_lstm_no_transfer -d ${tar_domain}
    python run_experiment.py -e rest_to_one_bi_art_lstm -d ${tar_domain}
done
