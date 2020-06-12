Using automatic attention score based word list to analyze SEC filings: 8-K, 10-K, and changes in 10-K reports of U.S. banks
===============

Code and data for "Using automatic attention score based word list to analyze SEC filings: 8-K, 10-K, and changes in 10-K reports of U.S. banks"

## Prerequisites
This code is written in python. To use it you will need:
- Python 3.6
- [PyTorch [PyTorch] = 1.2.0](https://pytorch.org/)
- A recent version of [scikit-learn](https://scikit-learn.org/)
- A recent version of [Numpy](http://www.numpy.org)
- A recent version of [NLTK](http://www.nltk.org)
- [Tensorflow = 1.15.2](https://www.tensorflow.org)

## Getting started
The data can be accessed at https://vr25.github.io/us_bank_sec/

Please download all the data (csv files) from [Box](https://rpi.box.com/s/wiofkzqvin7hplraolan5lnt05fgduo6).

```python3 ridge_reg.py arg1 arg2 arg3```

All 8-K or 10-K:

arg1: ```all_8-K_Q2.csv``` or ```all_8-K_data.csv``` or ```all_10-K_sec_1A.csv``` or ```all_10-K_sec_7.csv``` or ```all_10-K_full.csv```

arg2: None

arg3: None


Common 8-K and 10-K:

arg1: ```all_8-K_Q2.csv``` or ```all_8-K_data.csv```

arg2: ```all_10-K_sec_1A.csv``` or ```all_10-K_sec_7.csv``` or ```all_10-K_full.csv```

arg3: concat_t, concat_t1, changes(arg2 is always all_10-K_full.csv)


All the results will be saved in mse_scores.txt


The [results](https://docs.google.com/spreadsheets/d/17ixZNnsLHj0JHHOL2UrT7-I560pL8mA_34mnQmIGcfs/edit?usp=sharing) are updated too.
