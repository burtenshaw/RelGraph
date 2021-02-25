# %%
import sklearn
from sklearn import preprocessing
# KMeans clustering a kind of clustering.
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import sys
import umap
import umap.plot
import hdbscan
import os
from random import sample 
from spacy.lang.en import English
nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)
import matplotlib.pyplot as plt

from string import punctuation
from spacy.lang.en.stop_words import STOP_WORDS
#%%

n_characters = 20
from_each_book = 100

data_dir = os.path.join('data')

df = pd.read_json(os.path.join(data_dir,'PAIRS.json'))

junk = set([' '] + list(punctuation) + list(STOP_WORDS)) 

df = df.loc[df.chunk.apply(lambda x : {str(t) for t in tokenizer(x)})\
                    .apply(lambda  x : not x.issubset(junk))]

frequent = df.source_base.value_counts()[:n_characters].index

df = df.loc[df.source_base.map(lambda x : x in frequent) \
            & \
            df.target_base.map(lambda x : x in frequent) 
            ]

df['chunk_len'] = df.chunk.apply(lambda c : len(c.split(' ')))
df.loc[df.groupby('book').chunk_len.apply(lambda ser : ser.sort_values(ascending=False)[:200]).index.droplevel(0)]
# df = df.groupby('book').sample(n=from_each_book)
ENTITY_CHUNKS = df.train_data.apply(lambda x : ''.join(x)).to_list()
#%%

def Kmeans_clusters(text,c):
    vec = TfidfVectorizer(tokenizer=tokenizer, 
    stop_words='english', use_idf=True)
    matrix = vec.fit_transform(text)
    km = KMeans(n_clusters=c)
    km.fit(matrix)
    return km.labels_

def umap_hdbscan(ENTITY_CHUNKS, model, n_neighbors = 8, n_components = 2, min_cluster_size = 100, min_samples = 10):

    BERT_embeddings = model.encode(ENTITY_CHUNKS)

    umap_embeddings = umap.UMAP(n_neighbors=n_neighbors, 
                        n_components=n_components, 
                        metric='cosine').fit_transform(BERT_embeddings)
    
    cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                            metric='euclidean',                      
                            cluster_selection_method='eom',
                            min_samples = min_samples,
                            gen_min_span_tree=True)

    predictions = cluster.fit_predict(umap_embeddings)

    return predictions


# FAST CLUSTERING
def community_detection(ENTITY_CHUNKS, model, threshold=0.75, min_community_size=10, init_max_size=1000):

    embeddings = model.encode(ENTITY_CHUNKS)

    # Compute cosine similarity scores
    cos_scores = util.pytorch_cos_sim(embeddings, embeddings)

    # Minimum size for a community
    top_k_values, _ = cos_scores.topk(k=min_community_size, largest=True)

    # Filter for rows >= min_threshold
    extracted_communities = []
    for i in range(len(top_k_values)):
        if top_k_values[i][-1] >= threshold:
            new_cluster = []

            # Only check top k most similar entries
            top_val_large, top_idx_large = cos_scores[i].topk(k=init_max_size, largest=True)
            top_idx_large = top_idx_large.tolist()
            top_val_large = top_val_large.tolist()

            if top_val_large[-1] < threshold:
                for idx, val in zip(top_idx_large, top_val_large):
                    if val < threshold:
                        break

                    new_cluster.append(idx)
            else:
                # Iterate over all entries (slow)
                for idx, val in enumerate(cos_scores[i].tolist()):
                    if val >= threshold:
                        new_cluster.append(idx)

            extracted_communities.append(new_cluster)

    # Largest cluster first
    extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)

    # Step 2) Remove overlapping communities
    unique_communities = []
    extracted_ids = set()

    for community in extracted_communities:
        add_cluster = True
        for idx in community:
            if idx in extracted_ids:
                add_cluster = False
                break

        if add_cluster:
            unique_communities.append(community)
            for idx in community:
                extracted_ids.add(idx)
    
    return unique_communities

def align_predictions(unique_communities, s, top_k = None):

    cluster_sentences = {}

    for i, cluster in enumerate(unique_communities):
        print("\nCluster {}, #{} Elements ".format(i+1, len(cluster)))
        cluster = sample(cluster[:top_k], s)
        for sentence_id in cluster:
            cluster_sentences[sentence_id] = i 

    return cluster_sentences
#%%
# model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# unique_communities = community_detection(ENTITY_CHUNKS, 
#                             model, 
#                             min_community_size=30, 
#                             threshold=0.88)

# print(len(unique_communities))
# cluster_sentences = align_predictions(unique_communities, 10, top_k=20)
# df['cluster'] = pd.Series(cluster_sentences)

# anno = df.loc[df.cluster > 0].sort_values('cluster')

# with pd.ExcelWriter(os.path.join(data_dir, 'clusters.xlsx')) as writer:
#     for cluster_id, _df in anno.groupby('cluster'):
#         _df.to_excel(writer, 'cluster_%s' % cluster_id)

# anno.to_json('data/cluster_sample.json')
#%%
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
BERT_embeddings = model.encode(ENTITY_CHUNKS)

n_neighbors = 8
n_components = 2
min_dist = .99

print(n_neighbors, n_components, min_dist)

mapper = umap.UMAP(n_neighbors=n_neighbors, 
            n_components=n_components,
            min_dist=min_dist,
            metric='euclidean').fit(BERT_embeddings)


umap.plot.connectivity(mapper, show_points=True)

#%%
mapper = umap.UMAP(n_neighbors=n_neighbors, 
            n_components=n_components,
            min_dist=min_dist,
            metric='euclidean').fit_transform(BERT_embeddings)

min_samples = 6
min_cluster_size = 15

cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                        metric='euclidean',                      
                        cluster_selection_method='eom',
                        min_samples = min_samples,
                        gen_min_span_tree=True)

df['umap_predictions'] =  cluster.fit_predict(mapper)
print(df.umap_predictions.max())

anno = df.loc[df.umap_predictions > 0].sort_values('umap_predictions')
with pd.ExcelWriter(os.path.join(data_dir, 'clusters.xlsx')) as writer:
    for cluster_id, _df in anno.groupby('umap_predictions'):
        _df.to_excel(writer, 'cluster_%s' % cluster_id)

#%%
# n_neighbors = [2,4,8,12]
# n_components = [2,3]
# min_dist= [.8,.9,.99]

# for n in n_neighbors:
#     for c in n_components:
#         for dis in min_dist:

#             print(n, c, dis)

#             u = umap.UMAP(n_neighbors=n, 
#                         n_components=c,
#                         min_dist=dis,
#                         metric='euclidean').fit_transform(BERT_embeddings)

#             fig = plt.figure()

#             if n_components == 1:
#                 ax = fig.add_subplot(111)
#                 ax.scatter(u[:,0], range(len(u)), s=0.1, cmap='Spectral')
#             if n_components == 2:
#                 ax = fig.add_subplot(111)
#                 ax.scatter(u[:,0], u[:,1], s=0.1, cmap='Spectral')
#             if n_components == 3:
#                 ax = fig.add_subplot(111, projection='3d')
#                 ax.scatter(u[:,0], u[:,1], u[:,2], s=100, cmap='Spectral')

#             plt.savefig('data/cluster_plots/%s_%s_%s.png' % (n,c,dis))


# #%%

# if 'kmeans' in sys.argv:
#     df['K_Clusters'] = Kmeans_clusters(ENTITY_CHUNKS, 20) 

# if 'umap' in sys.argv:
#     model = SentenceTransformer('distilbert-base-nli-mean-tokens')
#     df['umap_predictions'] = umap_hdbscan(ENTITY_CHUNKS, model)

# if 'fast' in sys.argv:
#     print('doing fast clusters')
#     model = SentenceTransformer('distilbert-base-nli-mean-tokens')
#     unique_communities = community_detection(ENTITY_CHUNKS, 
#                                 model, 
#                                 min_community_size=60, 
#                                 threshold=0.7)
#     df = align_predictions(ENTITY_CHUNKS, unique_communities, df)
# #%%

# # %%
