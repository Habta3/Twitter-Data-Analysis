import pandas as pd
import pandas as pd
import json
from pandas.io.json import json_normalize
import os
import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from gensim.models import CoherenceModel
from wordcloud import STOPWORDS,WordCloud
import gensim
from gensim import corpora
from pprint import pprint
import re

import pandas as pd

twetts_df = pd.read_json('africa_twitter_data.json', lines=True)
import numpy as np
keywords=input("Enter keywords to be included")
print(twetts_df["source"])
print(twetts_df["created_at"])
print(twetts_df["full_text"])
print(twetts_df["truncated"])
print(twetts_df["display_text_range"])
print(twetts_df["entities"])
print(twetts_df["metadata"])
print(twetts_df["in_reply_to_status_id"])
print(twetts_df["retweet_count"])








# df = pd.DataFrame(
#     {
#         "A": [23,89,90,np.nan,90,200],
#         "full_text": [23,89,90,np.NAN,90,200],
#
#     }
# )


def check_column(df):
    column=[column for column in df.columns]
    return column
def chek_keywords(keyword):
    keywords=[keyword for keyword in twetts_df["full_text"].values]
    return keywords
def impute_columns(df):
    # df["full_text"] = df.fillna(method='ffill', inplace=True)
    # df["A"] = df.fillna(method='ffill', inplace=True)

    twetts_df['quoted_status'].fillna(method='ffill', inplace=True)
    twetts_df['full_text'].fillna(method='ffill', inplace=True)
    twetts_df['in_reply_to_status_id'].fillna(method='ffill', inplace=True)

    # df["A"] = df['A'].replace('', df['full_text'].mean(), inplace=True)
    # df["full_text"] = df['A'].replace('', "missed", inplace=True)
    # df["A"]=df["full_text"].replace('',"missed",inplace=True)

    # filled_column=[col for col in df.isna()]
    # if filled_column==True:
    #     df["A"].fillna("missed")
    #     df["full_text"].fillna("None")
    #     df=df.fillna(method='ffill', inplace=True)
    #
    #     #df["full_text"].interpolate(method='linear', limit_direction='backward', inplace=True)
    #     df["full_text"]=df['A'].replace('',df['A'].mean() , inplace=True)
    return df
def data_normalization(df):
    df = pd.io.json.json_normalize(df)
    return df
def data_transform(df):
    df=df.groupby('id')["retweet_count"].mean()
    return df
print(check_column(twetts_df))
# print(chek_keywords(keywords))
print("interpolated ",impute_columns(twetts_df))
print(data_normalization(twetts_df))
print((data_transform(twetts_df)))



twetts_df.dropna()
print(twetts_df)
print(len(twetts_df))
print(twetts_df.head())

class PrepareData:
    def __init__(self, df):
        self.df=df
    def preprocess_data(self):
        tweet_df=self.df.loc[self.df['lang']=="eng"]

        #test preprocessing
        tweet_df["full_text"]=twetts_df['full_text'].astype(str)
        print("original tweet")
        tweet_orig=[t for t in tweet_df["full_text"]]
        print(tweet_orig)
        twetts_df['full_text']=twetts_df['full_text'].apply(lambda x: x.lower())
        twetts_df["full_text"]=twetts_df['full_text'].apply(lambda x: x.translate(str.maketrans(' ',' ',string.punctuation)))

        #Convert tweets to list words
        sentence_list=[tweet for tweet in twetts_df['full_text']]
        word_list=[sentence.split() for sentence in sentence_list]
        print(sentence_list)
        print(word_list)
        # create word dictionary with Word_Id and Word
        word_to_id=corpora.Dictionary(word_list)
        dictionary=[word_to_id.doc2bow(tweet) for tweet in word_list]
        return word_list,word_to_id,dictionary

from textblob import TextBlob
import sys
import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
import pycountry
import re
import string
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
def create_wordcloud(text):
 mask = np.array(Image.open("cloud.png"))
 stopwords = set(STOPWORDS)
 wc = WordCloud(background_color="white",
 mask = mask,
 max_words=3000,
 stopwords=stopwords,
 repeat=True)
 wc.generate(str(text))
 wc.to_file("wc.png")
 print("Word Cloud Saved Successfully")
 path="wc.png"
 #display(Image.open(path))

tweet_list = pd.DataFrame(twetts_df)
create_wordcloud(tw_list["full_text"].values)

pre1=PrepareData(twetts_df)
pre1.preprocess_data()
word_list,id2word,corpus=pre1.preprocess_data()
print(corpus)

id_words=[[(id2word[id],count) for id, count in line] for line in corpus]
print("Id Words:",id_words)

#Build LDA Model
lda_model = gensim.models.ldamodel.LdaModel(corpus,
                                           id2word=id2word,
                                           num_topics=5,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
pprint(lda_model.print_topics())

pprint(lda_model.show_topics(formatted=False))

#Model Analysis
#perplexit is the measure of model quality that tells how amodel predicts samples
#cohrent matrix tests the models accuracy
print(" ----------------*****************************************-----------------------------------")
print('\nPerplexity: ', lda_model.log_perplexity(corpus))
print(" ----------------*****************************************-----------------------------------")
doc_lda = lda_model[corpus]


# Compute Coherence Score
# coherence_model_lda = CoherenceModel(model=lda_model, texts=word_list, dictionary=id2word, coherence='c_v')
# coherence_lda = coherence_model_lda.get_coherence()
# print('\n Ldamodel Coherence Score/Accuracy on Tweets: ', coherence_lda)

import pyLDAvis.gensim_models as gensimvis
import pickle
import pyLDAvis
# Visualize the topics
pyLDAvis.enable_notebook()

LDAvis_prepared = gensimvis.prepare(lda_model, corpus, id2word)
LDAvis_prepared
