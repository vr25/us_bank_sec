import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from bs4 import BeautifulSoup
import re
import os
import multiprocessing as mp
import time
import pandas as pd
import sys
import re
import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import csv
from sklearn.preprocessing import MinMaxScaler

csv.field_size_limit(sys.maxsize)

start = time.time()

s1 = stopwords.words('english')
s1.append('%')
s1.append('#')
s1.append('$')

s2 = open('atire_puurula.txt', 'r')
s2_con = s2.readlines()
s2_con = [w.strip('\n') for w in s2_con]

s = s1 + s2_con

tags_list = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WP', 'WP$']
 
ps = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

f1_con = open('1696_stemmed_sentiment_wordlist.txt', 'r').read().split('\n')

f2_con = open('exp_sim_vocab.txt', 'r').read().split('\n')

f3_con = open('exp_syn_vocab.txt', 'r').read().split('\n')

f4_con = open('lem_word_attn.txt', 'r').read().split('\n')

#words, sentiment words, attention words
def words(mda_con):
  tokens = tokenizer.tokenize(mda_con)
  clean_tokens = tokens[:]

  for token in tokens:
    if len(token) <= 2:
      clean_tokens.remove(token)

    elif token in s:
      clean_tokens.remove(token)

  stemmed_tokens = [ps.stem(word) for word in clean_tokens]

  sentiment_stemmed_tokens = stemmed_tokens[:]

  attention_stemmed_tokens = stemmed_tokens[:] 

  for t in stemmed_tokens:
    if t not in f1_con:
      sentiment_stemmed_tokens.remove(t)

    if t not in f4_con:
      attention_stemmed_tokens.remove(t)

  return (' ').join(stemmed_tokens), (' ').join(sentiment_stemmed_tokens), (' ').join(attention_stemmed_tokens) 


#exp_syn_words
def exp_syn_words(mda_con):
	tokens = tokenizer.tokenize(mda_con)
	f_con_pos = nltk.pos_tag(tokens)
	clean_tokens = f_con_pos[:]

	for (x,y) in f_con_pos:
		if len(x) <= 2:
			clean_tokens.remove((x,y))
		elif x in s:
			clean_tokens.remove((x,y)) # 2. Remove stopwords

	stemmed_tokens = [(ps.stem(x),y) for (x,y) in clean_tokens]  #3. Stemming

	sentiment_stemmed_tokens = stemmed_tokens[:] # 4. Keep only sentiment words (without expansion)

	for (a,b) in stemmed_tokens:
		if '_'.join((a,b)) not in f3_con:
			sentiment_stemmed_tokens.remove((a,b))

	v = []

	for (x,y) in sentiment_stemmed_tokens:

		if y not in tags_list:
			continue
		else:
			v.append(tag_replace(x,y))

	return (' ').join(v)

#replace POS tags
def tag_replace(x,y):
  if y in ['JJ', 'JJR', 'JJS']:
    y = 'JJ'
  elif y in ['NN', 'NNS', 'NNP', 'NNPS']:
    y = 'NN'
  elif y in ['PRP', 'PRP$']:
    y = 'PRP'
  elif y in ['RB', 'RBR', 'RBS']:
    y = 'RB'
  elif y in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
    y = 'VB'
  elif y in ['WP', 'WP$']:
    y = 'WP'
    
  return(x + '_' + y)


#min-max scale values
def scale_values(X):
    scaler = MinMaxScaler()
    scaler.fit(X.values.reshape(-1,1))
    print(scaler.data_min_)  
    print(scaler.data_max_)  
    return scaler.transform(X.values.reshape(-1,1))

'''
#Uncomment this!

i = 0

df = pd.DataFrame(columns=['cik_year', 'tokenized_words', 'len_tokenized_words', 'words', 'len_words', 'sentiment_words', 'len_sentiment_words', 'expanded_syntactic_words', 'len_expanded_syntactic_words', 'attention_words', 'len_attention_words'])

basepath = './section_1A/'
#basepath = './section_7/'
#basepath = './full_10-K/'
#basepath = './all_8-K_data_Q2/'
#basepath = './all_8-K_data/'

files_list = os.listdir(basepath)

for fi in files_list:

	cik = fi.split("_")[1]
	year = fi.split("_")[0].split("-")[0]
	cik_year = str(cik) + "_" + str(year)
	df.at[i,'cik_year'] = cik_year

	f_con = open(basepath + fi, 'r').read()

	#Remove HTML tags
	data = BeautifulSoup(f_con, "html.parser").text
	#Remove hexcode
	data = data.encode('ascii', errors='ignore')
	#Downcase
	data = data.lower()
	#Remove decimal point and comma from the numbers    (?<=\d)
	data = re.sub(r'[,\.](?=\d)','',data.decode('utf8'))
	#Remove \n
	data = re.sub(r'[\n]+', ' ', data)
	#Replace abc.def.ghi with ##ghi
	data = re.sub(r'(\w)+[\.](?=\w)+','#',data)
	#Remove everything except letters, digits, $, %, # (for above website/url/email case)
	data = re.sub(r'[^a-z0-9$%#\-/ ]', '', data)
	#Replace - with blank
	data = re.sub(r'[\-]', ' ', data)
	#Replace / with blank
	data = re.sub(r'[/]', ' ', data)
	#Replace ...... with .
	data = re.sub('\.+', '.', data)
	#Replace  . with blank
	data = re.sub(' . ', ' ', data)
	#Replace numbers with #
	data = re.sub(r'[0-9]+', '#', data)
	#Replace no. with ''
	data = re.sub(r'no.', '', data)
	#Replace all whitespaces to a single whitespace
	data = re.sub(' +', ' ', data)
	data = data.lstrip()

	df.at[i,'tokenized_words'] = data
	df.at[i, 'len_tokenized_words'] = len(data)
	df.at[i,'words'] = words(data)[0]
	df.at[i, 'len_words'] = len(words(data)[0])
	df.at[i,'sentiment_words'] = words(data)[1]
	df.at[i, 'len_sentiment_words'] = len(words(data)[1])
	df.at[i,'expanded_syntactic_words'] = exp_syn_words(data)
	df.at[i, 'len_expanded_syntactic_words'] = len(exp_syn_words(data))
	df.at[i,'attention_words'] = words(data)[2]
	df.at[i, 'len_attention_words'] = len(words(data)[2])

	i = i + 1
'''

df_perf_var = pd.read_csv('merged_data_risk.csv', usecols=['cik','fyear', 'roa', 'eps', 'tobinq', 'tier1_c', 'leverage', 'Z_score_c'])
df_perf_var['cik_year'] = df_perf_var['cik'].astype(str) + "_" + df_perf_var['fyear'].astype(str)
df_perf_var = df_perf_var.dropna()

target = ['roa', 'eps', 'tobinq', 'tier1_c', 'leverage', 'Z_score_c']

for t in target:
    s = t + "_scaled"
    df_perf_var[s] = scale_values(df_perf_var[t]) 

#Uncomment this!
df_ = pd.merge(df_v, df_perf_var, on='cik_year')

#Find previous year scores
cik_list = df_['cik'].tolist()
year_list = df_['fyear'].tolist()
cik_year_list = df_['cik_year'].tolist()

df2 = pd.DataFrame()
df3 = pd.DataFrame()

for i in range(len(df_)):
    cik = cik_list[i]
    year = year_list[i]
    cy = cik_year_list[i]

    year1 = year + 1

    cik_year1 = str(cik) + "_" + str(year1)

    if cik_year1 in cik_year_list:
        df2 = df2.append(df_.loc[df_['cik_year']==cy], ignore_index=True)
        df2 = df2.drop_duplicates(subset='cik_year')

        df3 = df3.append(df_.loc[df_['cik_year']==cik_year1],  ignore_index=True)
        df3 = df3.drop_duplicates(subset='cik_year')

for t in target:
	s1 = t + "_prev"
    s2 = t + "_scaled"
    s3 = t + "_scaled_prev"

    df3[s1] = df2[t]
    df3[s3] = df2[s2]

df3.to_csv('all_10-K_sec_1A.csv')

print("Total duration (sec):" , time.time() - start)