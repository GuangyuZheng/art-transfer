# ART
### Implementation for ICLR 2019 "Transfer Learning for Sequences via Learning to Collocate"

# Requirements
- Python (>=3.5)
- Keras (>=2.2.4)
- TensorFlow (>=1.10.0)
- [Recurrentshop (==1.0.0)](/https://github.com/farizrahman4u/recurrentshop)

# Get Required Data
```
./download.sh
```
- Data and processing codes for POS tagging and NER are from [https://github.com/kimiyoung/transfer]

# Get started

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
