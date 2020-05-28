The data can be accessed at https://vr25.github.io/us_bank_sec/

Please download all the data (csv files) from [Box](https://rpi.box.com/s/wiofkzqvin7hplraolan5lnt05fgduo6).

python3 ridge_reg.py arg1 arg2 arg3

arg1: all_8-K_Q2.csv or all_8-K_data.csv
arg2: all_10-K_sec_1A.csv or all_10-K_sec_7.csv or all_10-K_full.csv
arg3: concat_t, concat_t1, changes(arg2 is always all_10-K_full.csv)

All the results will be saved in mse_scores.txt
