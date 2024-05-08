import sys
##Install nltk package for tweets clean ups and stop words removal
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')

from sentence_transformers import SentenceTransformer
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

import string
from string import digits
#import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import gensim
from gensim.utils import simple_preprocess
import re


stop_words = stopwords.words('english')

def my_preprocessor(doc):
	doc_lower = doc.lower().replace("amp", "")
	doc_lower = re.sub(r'http\S+', '', doc_lower)
	remove_digits = str.maketrans('', '', string.digits)
	translator = str.maketrans('', '', string.punctuation)
	doc_no_punctuation = doc_lower.translate(translator).translate(remove_digits)
	return(doc_no_punctuation)

import spacy
spacy.load('en_core_web_sm')
lemmatizer = spacy.lang.en.English()

def my_tokenizer1(doc):
	tokens = lemmatizer(doc)
	return([token.lemma_ for token in tokens])

def my_tokenizer2(doc):
	if len(doc) == 0:
		return('')
	else:
		doc_tokens =gensim.utils.simple_preprocess(doc, deacc=True)
		ps = PorterStemmer()
		doc_filtered = [w for w in doc_tokens if not w in stop_words]
		doc_stem = [ps.stem(w) for w in doc_filtered]
		# return(doc_stem)
		return (" ".join(doc_stem))

docs1 = pd.read_csv("data/forest_landscape_restoration_country_tab.tsv", sep ='\t')
print(len(docs1))
docs1['text'] = docs1['text'].apply(my_preprocessor)
docs1['text'].replace('', np.nan, inplace=True)
docs1.dropna(subset=['text'], inplace=True)
docs1['text'] = docs1['text'].apply(my_tokenizer2)
docs1['text'].replace('', np.nan, inplace=True)
docs1.dropna(subset=['text'], inplace=True)
print(len(docs1))
docs1.head()

data1 = docs1['text'].tolist()
year_month1 = docs1['year_month'].tolist()

# docs2 = pd.read_csv("data/landscape_restoration_country_tab.tsv", sep ='\t')
# print(len(docs2))
# docs2['text'] = docs2['text'].apply(my_preprocessor)
# docs2['text'].replace('', np.nan, inplace=True)
# docs2.dropna(subset=['text'], inplace=True)
# docs2['text'] = docs2['text'].apply(my_tokenizer2)
# docs2['text'].replace('', np.nan, inplace=True)
# docs2.dropna(subset=['text'], inplace=True)
# print(len(docs2))

# data2 = docs2['text'].tolist()
# year_month2 = docs2['year_month'].tolist()

# docs3 = pd.read_csv("data/ecosystem_restoration_country_tab.tsv", sep ='\t')
# print(len(docs3))
# docs3['text'] = docs3['text'].apply(my_preprocessor)
# docs3['text'] = docs3['text'].apply(my_tokenizer2)
# docs3['text'].replace('', np.nan, inplace=True)
# docs3.dropna(subset=['text'], inplace=True)
# print(len(docs3))

# data3 = docs3['text'].tolist()
# year_month3 = docs3['year_month'].tolist()

# docs4 = pd.read_csv("data/ecological_restoration_country_tab.tsv", sep ='\t')
# print(len(docs4))
# docs4['text'] = docs4['text'].apply(my_preprocessor)
# docs4['text'] = docs4['text'].apply(my_tokenizer2)
# docs4['text'].replace('', np.nan, inplace=True)
# docs4.dropna(subset=['text'], inplace=True)
# print(len(docs4))

# data4 = docs4['text'].tolist()
# year_month4 = docs4['year_month'].tolist()

#import hdbscan
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
# import umap
from umap import UMAP

# Prepare custom models
# min_sample : large number leads to more outliers
#cluster_selection_epsilon
hdbscan_model = HDBSCAN(min_cluster_size=20,min_samples=5, metric='euclidean',
                        cluster_selection_method='eom', prediction_data=True)

# random state for reproducibility
#n_neighbors is the numer of neighboring sample points used when making the manifold approximation.
#Increasing this value typically results in a more global view of the embedding structure whilst smaller values result in a more local view.
#Increasing this value often results in larger clusters being created.
umap_model = UMAP(n_neighbors=15, n_components=10,random_state=120,min_dist=0.0, metric='cosine')
vectorizer_model = CountVectorizer(ngram_range=(2, 3),max_df=0.95, min_df=0.05)
#vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")

from bertopic import BERTopic

topic_model = BERTopic(umap_model=umap_model,
                       hdbscan_model=hdbscan_model,
                       vectorizer_model=vectorizer_model)

topics, probs = topic_model.fit_transform(data1)

print(topic_model.get_topic_info().count())
print(topic_model.get_topic_info())
fig = topic_model.visualize_barchart(top_n_topics=12, width=512, height=450, n_words=20)
fig.write_image("FR_topic_1.png")
xx = topic_model.reduce_topics(data1, nr_topics="auto")
fig2 = xx.visualize_barchart(top_n_topics=12, width=512, height=450, n_words=20)
fig2.write_image("FR_topic_2.png")
print(xx.get_topic_info().count())
xx.get_topic_info().to_csv("fr.csv")


# topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, vectorizer_model=vectorizer_model)

# topics, probs = topic_model.fit_transform(data2)

# print(topic_model.get_topic_info().count())
# print(topic_model.get_topic_info())
# topic_model.visualize_barchart(top_n_topics=12, width=512, height=450, n_words=20)
# topic_model.reduce_topics(data2, nr_topics="auto")
# print(topic_model.get_topic_info().count())
# topic_model.get_topic_info().to_csv("lr.csv")


# topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, vectorizer_model=vectorizer_model)
# topics, probs = topic_model.fit_transform(data3)
# print(topic_model.get_topic_info().count())
# print(topic_model.get_topic_info())
# # topic_model.visualize_barchart(top_n_topics=12, width=512, height=450, n_words=20)
# topic_model.reduce_topics(data3, nr_topics="auto")
# fig = topic_model.visualize_barchart(top_n_topics=12, width=512, height=450, n_words=20)
# fig.write_image("EcosR_topic.png")
# print(topic_model.get_topic_info().count())
# topic_model.get_topic_info().to_csv("EcosR.csv")


# topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, vectorizer_model=vectorizer_model)
# topics, probs = topic_model.fit_transform(data4)
# print(topic_model.get_topic_info().count())
# print(topic_model.get_topic_info())
# # topic_model.visualize_barchart(top_n_topics=12, width=512, height=450, n_words=20)
# topic_model.reduce_topics(data4, nr_topics="auto")
# fig = topic_model.visualize_barchart(top_n_topics=12, width=512, height=450, n_words=20)
# fig.write_image("EcolR_topic.png")
# print(topic_model.get_topic_info().count())
# topic_model.get_topic_info().to_csv("EcolR.csv")