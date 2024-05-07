import sys
##Install nltk package for tweets clean ups and stop words removal
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
import pandas as pd
from typing import List, Union
import plotly.graph_objects as go
from sklearn.preprocessing import normalize
from plotly.validators.scatter.marker import SymbolValidator



def visualize_topics_over_time(topic_model,
                               topics_over_time: pd.DataFrame,
                               top_n_topics: int = None,
                               topics: List[int] = None,
                               normalize_frequency: bool = False,
                               custom_labels: Union[bool, str] = False,
                               title: str = "<b>Topics over Time</b>",
                               width: int = 1300,
                               height: int = 450) -> go.Figure:
    """ Visualize topics over time
    Arguments:
        topic_model: A fitted BERTopic instance.
        topics_over_time: The topics you would like to be visualized with the
                          corresponding topic representation
        top_n_topics: To visualize the most frequent topics instead of all
        topics: Select which topics you would like to be visualized
        normalize_frequency: Whether to normalize each topic's frequency individually
        custom_labels: If bool, whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
                       If `str`, it uses labels from other aspects, e.g., "Aspect1".
        title: Title of the plot.
        width: The width of the figure.
        height: The height of the figure.
    Returns:
        A plotly.graph_objects.Figure including all traces
    Examples:
    To visualize the topics over time, simply run:
    ```python
    topics_over_time = topic_model.topics_over_time(docs, timestamps)
    topic_model.visualize_topics_over_time(topics_over_time)
    ```
    Or if you want to save the resulting figure:
    ```python
    fig = topic_model.visualize_topics_over_time(topics_over_time)
    fig.write_html("path/to/file.html")
    ```
    <iframe src="../../getting_started/visualization/trump.html"
    style="width:1000px; height: 680px; border: 0px;""></iframe>
    """
    # colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#D55E00", "#0072B2", "#CC79A7"]
    colors= ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a", "#ffff99", "#b15928"]
    # raw_symbols = SymbolValidator().values
    # namestems = []
    # namevariants = []
    # symbols = []
    # for i in range(0,len(raw_symbols),3):
    #   name = raw_symbols[i+2]
    #   symbols.append(raw_symbols[i])
    #   namestems.append(name.replace("-open", "").replace("-dot", ""))
    #   namevariants.append(name[len(namestems[-1]):])
    # print(symbols)
    symbols = [0, 1, 2, 3, 101, 102, 103,114,115,117,126,116]

    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        selected_topics = list(topics)
    elif top_n_topics is not None:
        selected_topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        selected_topics = sorted(freq_df.Topic.to_list())

    # Prepare data
    if isinstance(custom_labels, str):
        topic_names = [[[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic] for topic in topics]
        topic_names = ["_".join([label[0] for label in labels[:4]]) for labels in topic_names]
        topic_names = [label if len(label) < 30 else label[:27] + "..." for label in topic_names]
        topic_names = {key: topic_names[index] for index, key in enumerate(topic_model.topic_labels_.keys())}
    elif topic_model.custom_labels_ is not None and custom_labels:
        topic_names = {key: topic_model.custom_labels_[key + topic_model._outliers] for key, _ in topic_model.topic_labels_.items()}
    else:
        topic_names = {key: value[:40] + "..." if len(value) > 40 else value
                       for key, value in topic_model.topic_labels_.items()}
    topics_over_time["Name"] = topics_over_time.Topic.map(topic_names)
    data = topics_over_time.loc[topics_over_time.Topic.isin(selected_topics), :].sort_values(["Topic", "Timestamp"])

    # Add traces
    fig = go.Figure()
    for index, topic in enumerate(data.Topic.unique()):
        trace_data = data.loc[data.Topic == topic, :]
        topic_name = trace_data.Name.values[0]
        words = trace_data.Words.values
        if normalize_frequency:
            y = normalize(trace_data.Frequency.values.reshape(1, -1))[0]
        else:
            y = trace_data.Frequency
        fig.add_trace(go.Scatter(x=trace_data.Timestamp, y=y,
                                 marker_symbol=symbols[index % 12],
                                 mode='lines+markers',
                                 marker_color=colors[index % 12],
                                 hoverinfo="text",
                                 name=topic_name,
                                 hovertext=[f'<b>Topic {topic}</b><br>Words: {word}' for word in words]))

    # Styling of the visualization
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.update_layout(
        yaxis_title="Normalized Frequency" if normalize_frequency else "Frequency",
        title={
            'text': f"{title}",
            'y': .95,
            'x': 0.40,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        template="simple_white",
        width=width,
        height=height,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
        legend=dict(
            title="<b>Topics",
        )
    )
    return fig

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

docs2 = pd.read_csv("data/landscape_restoration_country_tab.tsv", sep ='\t')
print(len(docs2))
docs2['text'] = docs2['text'].apply(my_preprocessor)
docs2['text'].replace('', np.nan, inplace=True)
docs2.dropna(subset=['text'], inplace=True)
docs2['text'] = docs2['text'].apply(my_tokenizer2)
docs2['text'].replace('', np.nan, inplace=True)
docs2.dropna(subset=['text'], inplace=True)
# print(len(docs2))
data2 = docs2['text'].tolist()
year_month2 = docs2['year_month'].tolist()

docs3 = pd.read_csv("data/ecosystem_restoration_country_tab.tsv", sep ='\t')
print(len(docs3))
docs3['text'] = docs3['text'].apply(my_preprocessor)
docs3['text'] = docs3['text'].apply(my_tokenizer2)
docs3['text'].replace('', np.nan, inplace=True)
docs3.dropna(subset=['text'], inplace=True)
# print(len(docs3))
data3 = docs3['text'].tolist()
year_month3 = docs3['year_month'].tolist()

docs4 = pd.read_csv("data/ecological_restoration_country_tab.tsv", sep ='\t')
print(len(docs4))
docs4['text'] = docs4['text'].apply(my_preprocessor)
docs4['text'] = docs4['text'].apply(my_tokenizer2)
docs4['text'].replace('', np.nan, inplace=True)
docs4.dropna(subset=['text'], inplace=True)
# print(len(docs4))
data4 = docs4['text'].tolist()
year_month4 = docs4['year_month'].tolist()

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

# topic_model = BERTopic(umap_model=umap_model,
#                        hdbscan_model=hdbscan_model,
#                        vectorizer_model=vectorizer_model)

# topics, probs = topic_model.fit_transform(data1)

# print(topic_model.get_topic_info().count())
# print(topic_model.get_topic_info())
# fig = topic_model.visualize_barchart(top_n_topics=12, width=512, height=450, n_words=20)
# fig.write_image("FR_topic_1.png")
# xx = topic_model.reduce_topics(data1, nr_topics="auto")
# fig2 = xx.visualize_barchart(top_n_topics=12, width=512, height=450, n_words=20)
# fig2.write_image("FR_topic_2.png")
# print(xx.get_topic_info().count())
# xx.get_topic_info().to_csv("fr.csv")

# topic_model.reduce_topics(data1, nr_topics="auto")
# topic_model.set_topic_labels({0: "Topic 0", 1: "Topic 1", 2: "Topic 2", 3: "Topic 3", 4: "Topic 4", 5: "Topic 5", 6: "Topic 6", 7: "Topic 7", 8: "Topic 8", 9: "Topic 9", 10: "Topic 10", 11: "Topic 11"})
# topics_over_time = topic_model.topics_over_time(data1, year_month1, datetime_format="%Y-%M", nr_bins=96)
# # fig = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=12, normalize_frequency=True, custom_labels=True)
# # fig.write_image("ts_topic_overtime_FLR.png")
# fig = visualize_topics_over_time(topic_model,topics_over_time, top_n_topics=12,normalize_frequency=True,custom_labels=True)
# fig.write_image("FLR_overtime.png")




topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, vectorizer_model=vectorizer_model)

topics, probs = topic_model.fit_transform(data2)

topic_model.reduce_topics(data2, nr_topics="auto")
topic_model.set_topic_labels({0: "Topic 0", 1: "Topic 1", 2: "Topic 2", 3: "Topic 3", 4: "Topic 4", 5: "Topic 5", 6: "Topic 6", 7: "Topic 7", 8: "Topic 8", 9: "Topic 9", 10: "Topic 10", 11: "Topic 11"})
topics_over_time = topic_model.topics_over_time(data2, year_month2, datetime_format="%Y-%M", nr_bins=96)
# fig2 = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=12, normalize_frequency=True, custom_labels=True)
# fig2.write_image("ts_topic_overtime_LR.png")
fig2 = visualize_topics_over_time(topic_model,topics_over_time, top_n_topics=12,normalize_frequency=True,custom_labels=True)
fig2.write_image("LR_overtime.png")



topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, vectorizer_model=vectorizer_model)
topics, probs = topic_model.fit_transform(data3)
topic_model.reduce_topics(data3, nr_topics="auto")
topic_model.set_topic_labels({0: "Topic 0", 1: "Topic 1", 2: "Topic 2", 3: "Topic 3", 4: "Topic 4", 5: "Topic 5", 6: "Topic 6", 7: "Topic 7", 8: "Topic 8", 9: "Topic 9", 10: "Topic 10", 11: "Topic 11"})
topics_over_time = topic_model.topics_over_time(data3, year_month3, datetime_format="%Y-%M", nr_bins=96)
# fig3 = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=12, normalize_frequency=True, custom_labels=True)
# fig3.write_image("ts_topic_overtime_EcosR.png")
fig3 = visualize_topics_over_time(topic_model,topics_over_time, top_n_topics=12,normalize_frequency=True,custom_labels=True)
fig3.write_image("EcosR_overtime.png")



topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, vectorizer_model=vectorizer_model)
topics, probs = topic_model.fit_transform(data4)
topic_model.reduce_topics(data4, nr_topics="auto")
topic_model.set_topic_labels({0: "Topic 0", 1: "Topic 1", 2: "Topic 2", 3: "Topic 3", 4: "Topic 4", 5: "Topic 5", 6: "Topic 6", 7: "Topic 7", 8: "Topic 8", 9: "Topic 9", 10: "Topic 10", 11: "Topic 11"})
topics_over_time = topic_model.topics_over_time(data4, year_month4, datetime_format="%Y-%M", nr_bins=96)
# fig4 = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=12, normalize_frequency=True, custom_labels=True)
# fig4.write_image("ts_topic_overtime_EcolR.png")
fig4 = visualize_topics_over_time(topic_model,topics_over_time, top_n_topics=12,normalize_frequency=True,custom_labels=True)
fig4.write_image("EcoLR_overtime.png")
