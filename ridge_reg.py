import os
import sys
import csv
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
import re
import numpy as np
from sklearn.svm import SVR
import time
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import metrics
import copy
from multiscorer import MultiScorer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from scipy.sparse import hstack
from sklearn import linear_model

csv.field_size_limit(sys.maxsize)

start = time.time()

mse_file = open('mse_scores.txt', 'w')

def tfidf_vect(X):
	vect = TfidfVectorizer()
	v = vect.fit(X)
	X_vect  = v.transform(X)
	return X_vect

def compute(X_vect,y):

	scorer = MultiScorer({'mse' : (mean_squared_error, {})})

	#KernelRidgeRegression model - default degree=3
	model = KernelRidge(kernel='poly', alpha=0.1, gamma=0.1)

    # Perform 10-fold cross validation
	scores = cross_val_score(model, X_vect, y, cv=10, scoring=scorer)
	results = scorer.get_results()

	final_scores = []

	for metric_name in results.keys():
		average_score = np.average(results[metric_name])
		print('%s : %f' % (metric_name, average_score))
		final_scores.append(average_score)

	mse_file.write(str(final_scores[0]) + '\n')


def run_experiment(word_type, target, target_prev, data1, data2, data3, arg3, baseline):
    '''
    print(word_type)
    print(target)
    print(data1)
    print(data2)
    '''

    vect_shape = 0

    if arg3 == None:

        for w in word_type:
            for t in target:
                print("w: ", w, "t: ", t)
                X = data1[w].astype('U').values
                X_vect= tfidf_vect(X)
                y = data1[t]
                vect_shape = X_vect.shape
                compute(X_vect, y)

            mse_file.write(w + " " + t + " " + str(vect_shape) + "\n\n")
            

    elif arg3 == 'concat_t' or arg3 == 'concat_t1':

        for w in ['words_x']:
            for t in target:
                X_8K = data1[w].astype('U').values
                X_vect_8K = tfidf_vect(X_8K)

                X_10K = data1['words_y'].astype('U').values
                y_10K = data1[t]
                X_vect_10K = tfidf_vect(X_10K)

                X_vect_concat = csr_matrix(pd.concat([pd.DataFrame(X_vect_8K.todense()), pd.DataFrame(X_vect_10K.todense())], axis=1))
                compute(X_vect_concat, y_10K)


    elif arg3 == 'changes':

        for w in ['words']:
            for t in target:
                X_8K = data1[w].astype('U').values
                X_vect_8K = tfidf_vect(X_8K)
                y_8K = data1[t]
                vect_shape = X_vect_8K.shape
                mse_file.write("8K" + '\n')
                compute(X_vect_8K, y_8K)


                X_10K = data2[w].astype('U').values
                X_vect_10K = tfidf_vect(X_10K)
                X_10K1 = data3[w].astype('U').values
                X_vect_10K1 = tfidf_vect(X_10K1)
                y_10K1 = data3[t]
                X_vect_diff = pd.DataFrame(X_vect_10K1.todense()).subtract(pd.DataFrame(X_vect_10K.todense()), fill_value=0)
                mse_file.write("10-K changes" + '\n')
                compute(X_vect_diff, y_10K1)


    if baseline == True:
        run_baseline(word_type, target, target_prev, data1, data2, data3)


def run_baseline(word_type, target, target_prev, data1, data2, data3):
    #Baseline
    mse_file.write("\n-----BASELINE-----\n")

    for i in range(len(target)):        

        X = np.array(data1[target_prev[i]]).reshape(-1,1)
        y = np.array(data1[target[i]]).reshape(-1,1)

        scorer = MultiScorer({'mse' : (mean_squared_error, {})})

        model = LinearRegression()

        scores = cross_val_score(model, X, y, cv=10, scoring=scorer)
        results = scorer.get_results()

        final_scores = []

        for metric_name in results.keys():
            average_score = np.average(results[metric_name])
            print('{0:.20f}'.format(average_score))
            final_scores.append(average_score)

        mse_file.write(str(final_scores[0]) + '\n')


#Input args
input_args = sys.argv[1:]


if len(input_args) == 1:
    df = pd.read_csv(input_args[0], engine='python')

    word_type = ['words', 'sentiment_words', 'expanded_syntactic_words', 'attention_words']
    target = ['roa_scaled', 'eps_scaled', 'tobinq_scaled', 'tier1_c_scaled', 'leverage_scaled', 'Z_score_c_scaled']
    target_prev = ['roa_scaled_prev', 'eps_scaled_prev', 'tobinq_scaled_prev', 'tier1_c_scaled_prev', 'leverage_scaled_prev', 'Z_score_c_scaled_prev']

    run_experiment(word_type=word_type, target=target, target_prev=target_prev, data1=df, data2=None, data3=None, arg3=None, baseline=True)



if len(input_args) == 3:
    df1 = pd.read_csv(input_args[0], engine='python')
    df2 = pd.read_csv(input_args[1], engine='python')

    df2['prev_cik_year'] = df2['cik'].astype(str) + "_" + (df2['fyear'] - 1).astype(str)

    #8K(t) and 10-K(t) concat
    if input_args[2] == 'concat_t':
        df_common = pd.merge(df1, df2, on='cik_year')
        target = ['roa_scaled_y', 'eps_scaled_y', 'tobinq_scaled_y', 'tier1_c_scaled_y', 'leverage_scaled_y', 'Z_score_c_scaled_y']
        mse_file.write("\n" + "8K(t) and 10-K(t) concat" + "\n")
        run_experiment(word_type=None, target=target, target_prev=None, data1=df_common, data2=None, data3=None, arg3='concat_t', baseline=False)


    elif input_args[2] == 'concat_t1':
        #8K(t) and 10-K(t+1) concat
        df_common = pd.merge(df1, df2, left_on='cik_year', right_on='prev_cik_year')
        target = ['roa_scaled_y', 'eps_scaled_y', 'tobinq_scaled_y', 'tier1_c_scaled_y', 'leverage_scaled_y', 'Z_score_c_scaled_y']
        mse_file.write("\n" + "8K(t) and 10-K(t+1) concat" + "\n")
        run_experiment(word_type=None, target=target, target_prev=None, data1=df_common, data2=None, data3=None, arg3='concat_t1', baseline=False)


    elif input_args[2] == 'changes':
        #8K(t) and 10-K(t+1) - 10K(t) changes
        word_type = ['words']
        target = ['roa_scaled', 'eps_scaled', 'tobinq_scaled', 'tier1_c_scaled', 'leverage_scaled', 'Z_score_c_scaled']

        cik_list = df1['cik'].tolist()
        year_list = df1['fyear'].tolist()
        cik_year_list_8K = df1['cik_year'].tolist()
        cik_year_list_10K = df2['cik_year'].tolist()

        df8K = pd.DataFrame() # 8K(t)
        df10K = pd.DataFrame() # 10K(t)
        df10K1 = pd.DataFrame() # 10K(t+1)

        for i in range(len(df1)):
            cik = cik_list[i]
            year = year_list[i]
            cy = cik_year_list_8K[i]

            year1 = year + 1

            cik_year1 = str(cik) + "_" + str(year1)

            if cy in cik_year_list_10K and cik_year1 in cik_year_list_10K:
                df8K = df8K.append(df1.loc[df1['cik_year']==cy], ignore_index=True)
                df8K = df8K.drop_duplicates(subset='cik_year')

                df10K = df10K.append(df2.loc[df2['cik_year']==cy], ignore_index=True)
                df10K = df10K.drop_duplicates(subset='cik_year')

                df10K1 = df10K1.append(df2.loc[df2['cik_year']==cik_year1], ignore_index=True)
                df10K1 = df10K1.drop_duplicates(subset='cik_year')

        mse_file.write("\n" + "8K(t) and 10-K(t+1) - 10K(t) changes" + "\n")
        run_experiment(word_type=word_type, target=target, target_prev=None, data1=df8K, data2=df10K, data3=df10K1, arg3='changes', baseline=False)


    else:
        df_common = pd.merge(df1, df2, left_on='cik_year', right_on='prev_cik_year')

        word_type = ['words_x']
        target = ['roa_scaled_x', 'eps_scaled_x', 'tobinq_scaled_x', 'tier1_c_scaled_x', 'leverage_scaled_x', 'Z_score_c_scaled_x']
        target_prev = ['roa_scaled_prev_x', 'eps_scaled_prev_x', 'tobinq_scaled_prev_x', 'tier1_c_scaled_prev_x', 'leverage_scaled_prev_x', 'Z_score_c_scaled_prev_x']
        mse_file.write("\n" + "Data1 - 8K" + "\n")
        run_experiment(word_type=word_type, target=target, target_prev=target_prev, data1=df_common, data2=None, data3=None, arg3=None, baseline=True)

        word_type = ['words_y']
        target = ['roa_scaled_y', 'eps_scaled_y', 'tobinq_scaled_y', 'tier1_c_scaled_y', 'leverage_scaled_y', 'Z_score_c_scaled_y']
        target_prev = ['roa_scaled_prev_y', 'eps_scaled_prev_y', 'tobinq_scaled_prev_y', 'tier1_c_scaled_prev_y', 'leverage_scaled_prev_y', 'Z_score_c_scaled_prev_y']
        mse_file.write("\n" + "Data2 - 10-K" + "\n")
        run_experiment(word_type=word_type, target=target, target_prev=target_prev, data1=df_common, data2=None, data3=None, arg3=None, baseline=True)


print("Total execution time: ", time.time() - start)
