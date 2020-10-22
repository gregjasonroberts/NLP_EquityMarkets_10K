
#import libraries

import pandas as pd
import numpy as np
import xlrd
import pickle
import random

from edgar import Company
from edgar import TXTML

from bs4 import BeautifulSoup

import lxml.html
import lxml.html.soupparser
import requests, re
print('LXML parser, BeautifulSoup, and SEC edgar Libraries imported.')
print()

import nltk
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from html import unescape

import time
from tqdm import tqdm
import urllib.request
import urllib.error

import my_project_library as my_lib
import Temp_library as tlib

# create a dataframe from a word matrix
years_in_scope = [2019,2018,2017,2016,2015,2014]
def wmdf(wm, feat_names):
    # create an index for each row
    doc_names =  years_in_scope
    df = pd.DataFrame(data=wm.toarray(), index=doc_names,
                      columns=feat_names).transpose()
    return(df)

# create a spaCy tokenizer
spacy.load('en')
nlp = spacy.lang.en.English()
nlp.max_length = 1500000  #a handful of companies have so much text it exceeds the character limit on the library

# remove html entities from docs and set everything to lowercase
def my_preprocessor(doc):
    return(unescape(doc).lower())

# tokenize the doc and lemmatize its tokens
def my_tokenizer(doc):
    tokens = nlp(doc, disable = ['ner', 'parser']) #disable RAM-hungry intensive parts of the pipeline we don't need for lemmatization
    return([token.lemma_ for token in tokens])


'''We are building a dataset that will compile the relative changes in negative word
frequency to each filing, year after year.'''
def build_dataset(ticker_dict):

    #Iniitialize temporary dtatframe
    data_comp = pd.DataFrame(columns=['Company','Year','Filing_Date', 'Cos_Sim','Word_chg'])
    spx_length= len(ticker_dict)
    i=0

    for i in range(455,spx_length):
        try:

            pairs = {k: ticker_dict[k] for k in list(ticker_dict)[i:i+1]}
            for company_name, cik in pairs.items():
                print(i, company_name, cik)

            #Pull the 10K filings are parse the text for the Risk Factors sections
            documents = my_lib.pull_10K(company_name, cik)
            # print("checkpoint: at the 10k_pull")
            #Error handing for newly listed with not enough 10ks to process...
            if len(documents) < len(years_in_scope):
                print("document length is short")
                continue
            #Pull the risk sections for each 10k filing
            risk_sections = [my_lib.pull_risk_section(document) for document in documents]
            #tokenize and stem each each token from the document to find the YoY change
            negative_words = my_lib.word_filter() #retrieves the neg word list
            #the CountVectorizer counts the number of times a token appears in the document
            custom_vec = CountVectorizer(preprocessor=my_preprocessor, tokenizer=my_tokenizer, stop_words='english')
            cwm = custom_vec.fit_transform(risk_sections)
            tokens = custom_vec.get_feature_names()
            if len(tokens) < 10:
                print("token count is too small")
                continue
            counts = wmdf(cwm, tokens)
            tf_percent = (counts / counts.sum())  #Use TF_IDF to assign frequencies of the tokens.
            negative_frequency = tf_percent.reindex(negative_words).dropna()  #applies word list to the dataset
            nf_sum = negative_frequency.sum()  #aggregate frequencies in each year by our assigned vocab

            #run cosine similarity on the TDIF vectorized dataset
            tfIdfVectorizer=TfidfVectorizer(vocabulary=negative_words, use_idf=True)
            tfIdf = tfIdfVectorizer.fit_transform(risk_sections)
            df = pd.DataFrame(tfIdf.T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=[years_in_scope])
            df = df[(df.T != 0).any()]
            similarity = cosine_similarity(df.T)
            similarity = pd.DataFrame(similarity, index=years_in_scope,columns=years_in_scope)
            cos_sim = {'Company':[],'Year':[], 'Filing_Date':[],'Cos_Sim':[],'Word_chg':[]  }
            file_dates = my_lib.file_date(company_name, cik, similarity.shape[1])
            # print("checkpoint: at the file_date")
            # for year, values in similarity.items():
            for x in range(similarity.shape[1]):
                if x+1 == similarity.shape[1]: break
                cos_sim['Company'].append(company_name)
                cos_sim['Filing_Date'].append(file_dates[x])
                value = (similarity.iloc[x,x+1])
                cos_sim['Cos_Sim'].append(value)
                year = similarity.columns[x]
                cos_sim['Year'].append(year)
                nf_chg = (nf_sum.iloc[x]-nf_sum.iloc[x+1])/ nf_sum.iloc[x+1] #the relative change in the negative frequency
                cos_sim['Word_chg'].append(nf_chg)

            cos_sim_df = pd.DataFrame(cos_sim)
            data_comp = data_comp.append(cos_sim_df, ignore_index=True)
            if i % 2 == 0: #save our data every 10 entries due to SEC throttling
                with open('/home/gregjroberts1/Projects/Data/comp_builder_update.p', 'wb') as f:
                    pickle.dump(data_comp, f)

        except ValueError:
            pass
    #data_comp.reset_index(drop=True, inplace=True) #reset the index for final dataset
    return data_comp

"""
Build our dataframe from wikipedia list of companies and respective CIK code table
"""
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
sp_table = my_lib.scrape_wiki(url).find('table')
data_df = pd.read_html(str(sp_table))[0]
#data_df['Security'].replace({'Alphabet Inc. (Class A)': 'Alphabet Inc.'}, inplace=True)

#removing duplicative ticker and newly added with Amcor lacking sufficient filings
#or names like Citigroup and PSEG had filings that were too large to process, exceeding
#lxml threshholds
ticker_issues = []
remove_tickers = data_df[data_df['Security'].isin(
    ['American Electric Power','American International Group','Ameriprise Financial','Boston Properties',
    'Berkshire Hathaway','Chubb Limited','CMS Energy','Dominion Energy','Entergy Corp.','Equity Residential',
    'Evergy','Exelon Corp.','Healthpeak Properties','Huntington Bancshares','Linde plc','Lincoln National',
    'Principal Financial Group','Realty Income Corporation','International Business Machines','Iron Mountain Incorporated',
    'Southern Company','Welltower Inc.','UDR, Inc.','JPMorgan Chase & Co.','Kimco Realty','MetLife Inc.',
    'Morgan Stanley','PPL Corp.','Prudential Financial','Sempra Energy','Ventas Inc','Wells Fargo']
    )].index
data_df.drop(remove_tickers, inplace=True)
#standardize our CIK and make them all 10 digits in length
#in order to make it easier to access the documents from the SEC Edgar site
data_df['CIK']=data_df['CIK'].apply(lambda x: '{0:0>10}'.format(x))
data = data_df[['Symbol','Security','GICS Sector', 'GICS Sub Industry', 'CIK']]
with open('/home/gregjroberts1/Projects/Data/spx_wiki_table.p', 'wb') as f:
    pickle.dump(data, f)
ticker_dict = data.set_index('Security').to_dict()['CIK']

get_dataframe = build_dataset(ticker_dict)
'''build a temp dataframe to retry the bad tickers'''
# temp_df = data_df.iloc[remove_tickers]
# temp_data = temp_df[['Symbol','Security','GICS Sector', 'GICS Sub Industry', 'CIK']]
# temp_ticker_dict = temp_data.set_index('Security').to_dict()['CIK']
# get_dataframe = build_dataset(temp_ticker_dict)

get_dataframe
#
#
# data = data_df[['Symbol','Security','GICS Sector', 'GICS Sub Industry', 'CIK']]
# with open('/home/gregjroberts1/Projects/Data/spx_wiki_table.p', 'wb') as f:
#     pickle.dump(data, f)
# ticker_dict = data.set_index('Security').to_dict()['CIK']
#
# get_dataframe = build_dataset(ticker_dict)
# get_dataframe

with open('/home/gregjroberts1/Projects/Data/comp_builder_update.p', 'wb') as f:
    pickle.dump(get_dataframe, f)
#save down our final dataset
with open('/home/gregjroberts1/Projects/Data/spx_word_analysis.p', 'wb') as f:
    pickle.dump(get_dataframe, f)
