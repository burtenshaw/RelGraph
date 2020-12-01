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
import hdbscan

from spacy.lang.en import English
nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)


df = pd.read_pickle('/home/burtenshaw/now/potter_kg/data/chunk_relations_26_11_2020.bin')

entity_chunk_df = df[['source','chunk','target']]\
                    .dropna()\
                        .apply( lambda row : ' '.join(row.to_list())\
                                 if pd.notna(row.chunk) and len(row.chunk) > 2\
                                     else np.nan, axis = 1).dropna().drop_duplicates()

ENTITY_CHUNKS = entity_chunk_df.to_list()

results = pd.DataFrame()
results['sentences'] = entity_chunk_df


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

def align_predictions(ENTITY_CHUNKS, unique_communities, df):

    cluster_sentences = {}

    for i, cluster in enumerate(unique_communities):
        print("\nCluster {}, #{} Elements ".format(i+1, len(cluster)))
        cluster_sentences[i] = ENTITY_CHUNKS[cluster[0]] 
        for sentence_id in cluster:
            df.at[sentence_id,'BERT_cos_cluster'] = i
            df.at[sentence_id,'BERT_top_sentence'] = ENTITY_CHUNKS[cluster[0]] 
            print("\t", ENTITY_CHUNKS[sentence_id])

    return df
#%%
# sys.argv.append('fast')
#%%
if 'kmeans' in sys.argv:
    df['K_Clusters'] = Kmeans_clusters(ENTITY_CHUNKS, 20) 

if 'umap' in sys.argv:
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    df['umap_predictions'] = umap_hdbscan(ENTITY_CHUNKS, model)

if 'fast' in sys.argv:
    print('doing fast clusters')
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    unique_communities = community_detection(ENTITY_CHUNKS, 
                                model, 
                                min_community_size=60, 
                                threshold=0.7)
    df = align_predictions(ENTITY_CHUNKS, unique_communities, df)
#%%
df.to_pickle('/home/burtenshaw/now/potter_kg/data/cluster_sent_df.bin')
# %%
