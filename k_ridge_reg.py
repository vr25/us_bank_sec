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

	#KernelRidgeRegression model
    #default degree=3
	model = KernelRidge(kernel='poly', alpha=0.1, gamma=0.1)

	#GradientBoostingRegressor()

	#KernelRidge(kernel='poly', alpha=0.1, gamma=0.1)

	#MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.001,batch_size='auto', learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

	# Perform 10-fold cross validation
	scores = cross_val_score(model, X_vect, y, cv=10, scoring=scorer)
	results = scorer.get_results()

	final_scores = []

	for metric_name in results.keys():
		average_score = np.average(results[metric_name])
		print('%s : %f' % (metric_name, average_score))
		final_scores.append(average_score)

	mse_file.write(str(final_scores[0]) + '\n')


word_type = ['words', 'sentiment_words', 'expanded_syntactic_words', 'attention_words']
target = ['roa_scaled', 'eps_scaled', 'tobinq_scaled', 'tier1_c_scaled', 'leverage_scaled', 'Z_score_c_scaled']

vect_shape = 0

#Common data
#Uncomment this!
print("8K")
df_8K_t = pd.read_csv("all_8-K_Q2.csv")
cik_year_list_8K = df_8K_t['cik_year'].tolist()
print("length: ", len(df_8K_t))


print("10K")
df_10K_t = pd.read_csv("all_10-K_full.csv", engine='python')
df_10K_t['prev_cik_year'] = df_10K_t['cik'].astype(str) + "_" + (df_10K_t['fyear'] - 1).astype(str)
cik_year_list_10K = df_10K_t['cik_year'].tolist()
print("length: ", len(df_10K_t))

'''
#Uncomment this
#Changes
cik_list = df_8K_t['cik'].tolist()
year_list = df_8K_t['fyear'].tolist()

df2 = pd.DataFrame() # 8K(t)
df3 = pd.DataFrame() # 10K(t)
df4 = pd.DataFrame() # 10K(t+1)

df = df_10K_t

for i in range(len(df_8K_t)):
	
	cik = cik_list[i]
	year = year_list[i]
	cy = cik_year_list_8K[i]

	year1 = year + 1

	cik_year1 = str(cik) + "_" + str(year1)
	
	if cy in cik_year_list_10K and cik_year1 in cik_year_list_10K:
		df2 = df2.append(df_8K_t.loc[df_8K_t['cik_year']==cy], ignore_index=True)
		df2 = df2.drop_duplicates(subset='cik_year')

		df3 = df3.append(df.loc[df['cik_year']==cy], ignore_index=True)
		df3 = df3.drop_duplicates(subset='cik_year')

		df4 = df4.append(df.loc[df['cik_year']==cik_year1], ignore_index=True)
		df4 = df4.drop_duplicates(subset='cik_year')

print("df2: ", len(df2), "df3: ", len(df3), "df4: ", len(df4))
'''

df_common = pd.merge(df_8K_t, df_10K_t, on='cik_year') #left_on='cik_year', right_on='prev_cik_year')
print(df_common.head(1))
print("common length: ", len(df_common), df_common.columns)
df2 = df_common


for w in ['words_x']: #word_type:

  for t in target:

    print("w: ", w, "t: ", t)
    
    
    X_8K = df2[w].astype('U').values
    X_vect_8K = tfidf_vect(X_8K)
    #y_8K = df2[t]
    #vect_shape = X_vect_8K.shape
    #compute(X_vect_8K, y_8K)
     
    
    X_10K = df2['words_y'].astype('U').values
    #y_10K = df_common[t]
    X_vect_10K = tfidf_vect(X_10K)

    #X_10K1 = df4['words'].astype('U').values
    #X_vect_10K1 = tfidf_vect(X_10K1)
    t1 = t + "_y"
    y_10K1 = df2[t1]  #df4[t]
    X_vect_concat = csr_matrix(pd.concat([pd.DataFrame(X_vect_8K.todense()), pd.DataFrame(X_vect_10K.todense())], axis=1))
    compute(X_vect_concat, y_10K1)

    #X_vect_diff = pd.DataFrame(X_vect_10K1.todense()).subtract(pd.DataFrame(X_vect_10K.todense()), fill_value=0)
    #compute(X_vect_diff, y_10K1)
    

    '''
    #common
    w1 = w + "_x"
    t1 = t + "_x"
    X_common = df_common[w1].astype('U').values
    y_common = df_common[t1]
    X_vect_common = tfidf_vect(X_common)
    vect_shape = X_vect_common.shape
    compute(X_vect_common, y_common)

    
    #8K
    X_8K = df_8K_t[w].astype('U').values
    y_8K = df_8K_t[t]
    X_vect_8K = tfidf_vect(X_8K)
    vect_shape = X_vect_8K.shape
    compute(X_vect_8K, y_8K)				

    #All 10K		
    X_1i0K = df_10K_t[w].astype('U').values
    y_10K = df_10K_t[t]#.astype('float').values
    X_vect_10K = tfidf_vect(X_10K)
    vect_shape = X_vect_10K.shape
    compute(X_vect_10K, y_10K)
    '''   
        		
  mse_file.write(w + " " + t + " " + str(vect_shape) + "\n")
					
mse_file.close()


'''
Baseline
#Uncomment this!
df = df_common #df_10K_t

for t in target:
    t1 = t + "_x"
    prev = t + "_prev_x"
    X_10K1 = np.array(df[prev]).reshape(-1,1)
    y_10K1 = np.array(df[t1]).reshape(-1,1)

    scorer = MultiScorer({'mse' : (mean_squared_error, {})})

    model = LinearRegression()

    scores = cross_val_score(model, X_10K1, y_10K1, cv=10, scoring=scorer)
    results = scorer.get_results()

    final_scores = []

    for metric_name in results.keys():

        average_score = np.average(results[metric_name])
        print('{0:.20f}'.format(average_score))
        final_scores.append(average_score)

    mse_file.write(str(final_scores[0]) + '\n')

mse_file.close()
'''

print("Total execution time: ", time.time() - start)