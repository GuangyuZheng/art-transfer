# ART
### Implementation for ICLR 2019 "Transfer Learning for Sequences via Learning to Collocate"

# Requirements
- Python (>=3.5)
- Keras (>=2.2.4)
- TensorFlow (>=1.10.0)
- [Recurrentshop (==1.0.0)](/https://github.com/farizrahman4u/recurrentshop)
- [keras-contrib](https://github.com/keras-team/keras-contrib)

# Get Required Data
```
./download.sh
```
- Data and processing codes for POS tagging and NER are from [https://github.com/kimiyoung/transfer]

# Test
## Sentiment Analysis Task
```
cd sentiment_analysis
python test_art_model.py -s [source_domain] -t [target_domain]
```
## POS Tagging Task
```
cd pos_tagging
python test_art_model.py -s [source_domain] -t [target_domain] -r [rates]
```
## Named Entity Recognition Task
```
cd ner
python test_art_model.py -s [source_domain] -t [target_domain] -r [rates]
```

# Train

## Sentiment Analysis Task
```
cd sentiment_analysis
./all_train.sh
```
### POS Tagging Task
```
cd pos_tagging
./all_train.sh
```
### Named Entity Recognition Task
```
cd ner
./all_train.sh
```

# Citation
```
@inproceedings{cui2018transfer,
    title={Transfer Learning for Sequences via Learning to Collocate},
    author={Wanyun Cui and Guangyu Zheng and Zhiqiang Shen and Sihang Jiang and Wei Wang},
    booktitle={International Conference on Learning Representations},
    year={2019},
    url={https://openreview.net/forum?id=ByldlhAqYQ},
}
```