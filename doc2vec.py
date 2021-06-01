import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import plotly.express as px
import matplotlib.pyplot as plt

from gensim import utils
from gensim.models.doc2vec import LabeledSentence, TaggedDocument
from gensim.models import Doc2Vec
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score

## change

def build_d2v_model(data, vector_size, alpha, max_iterations, epochs,
                    min_count=1, window=10, dm=1, pretrained_emb=None):
    """
    Train a Doc2Vec model either from scratch (pretrained_emb = None)
    or using a pretrained embedding.
    """

    sentences = [str(line).lower().split() for line in data]
    tagged_data = [TaggedDocument(words=sentence, tags=[str(i)]) for i, sentence in enumerate(sentences)]

    model = Doc2Vec(vector_size=vector_size,
                    alpha=alpha,
                    min_count=min_count,
                    window=window,
                    dm=dm,
                    )  # 0 = dbow; 1 = dmpv

    if pretrained_emb is not None:
        model.pretrained_emb = pretrained_emb

    model.build_vocab(tagged_data)

    for epoch in range(max_iterations):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=epochs)
        # decrease the learning rate
        model.alpha -= alpha * 0.1
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    return model


def get_most_similar_products(app_name, topn=10):
    """
    Returns a graph containg the most similar applications name and rating.
    """

    v = model.infer_vector(app_name.lower().split())
    most_sim = model.docvecs.most_similar(positive=[v], topn=topn)
    table = {
        'name': [df.name.iloc[int(most_sim[i][0])] for i in range(len(most_sim))],
        'cos_sim': [most_sim[i][1] for i in range(len(most_sim))],
        'reviews': [df.reviews.iloc[int(most_sim[i][0])] for i in range(len(most_sim))],
        'rating': [round(df.rating.iloc[int(most_sim[i][0])], 2) for i in range(len(most_sim))]
    }
    table = pd.DataFrame.from_dict(table)
    table.sort_values(by='cos_sim', inplace=True)

    fig = px.bar(table, x='rating', y='name', color='cos_sim', orientation='h',
                 hover_data=['name', 'rating', 'reviews'],
                 color_continuous_scale='sunset',
                 height=400
                 )
    return fig