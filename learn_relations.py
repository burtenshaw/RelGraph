# %%
import sklearn
from sklearn import preprocessing
# KMeans clustering a kind of clustering.
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import seaborn as sns
from bokeh.plotting import figure, output_file, show, output_notebook

import umap
import hdbscan
# import umap.plot
import matplotlib.pyplot as plt

from spacy.lang.en import English
nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)

output_notebook()

df = pd.read_pickle('/home/burtenshaw/now/potter_kg/data/chunk_relations_26_11_2020.bin')
ENTITY_CHUNKS = df[['source','chunk','target']]\
                    .dropna()\
                        .apply( lambda row : ' '.join(row.to_list())\
                                 if pd.notna(row.chunk) and len(row.chunk) > 2\
                                     else np.nan, axis = 1).dropna().drop_duplicates()\
                                         .to_list()

#%%

def Kmeans_clusters(text,c):
    vec = TfidfVectorizer(tokenizer=tokenizer, 
    stop_words='english', use_idf=True)
    matrix = vec.fit_transform(text)
    km = KMeans(n_clusters=c)
    km.fit(matrix)
    return km.labels_

df['K_Clusters'] = Kmeans_clusters(ENTITY_CHUNKS, 20) 
# %%
# do bert embeddings
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
BERT_embeddings = model.encode(ENTITY_CHUNKS)
#%%

def build_clusters(umap_embeddings, n_neighbors, n_components, min_cluster_size):

    # umap clusters of bert embeddings

    cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                            metric='euclidean',                      
                            cluster_selection_method='eom').fit(umap_embeddings)

    color_palette = sns.color_palette("Spectral", n_colors=cluster.labels_.max()+1)

    cluster_colors = [color_palette[x] if x >= 0
                    else (0.5, 0.5, 0.5)
                    for x in cluster.labels_]

    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                            zip(cluster_colors, cluster.probabilities_)]
    
    plt.scatter(*umap_embeddings.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)

    plt.show()

    cluster.condensed_tree_.plot()
    plt.show()


# for n_components in [2]:
#     print(' Number of components %s' % n_components)

#     for n_neighbors in [4,6,8]:
#         print(' Nearest neighbors %s ' % n_neighbors)
#         umap_embeddings = umap.UMAP(n_neighbors=n_neighbors, 
#                             n_components=n_components, 
#                             metric='cosine').fit_transform(BERT_embeddings)
    
#         for min_cluster_size in [15,30,45]:
#             print('Minimum cluster size %s' % min_cluster_size)
#             build_clusters(umap_embeddings, n_neighbors, n_components, min_cluster_size)

n_neighbors = 8
n_components = 2
min_cluster_size = 100
min_samples = 10

umap_embeddings = umap.UMAP(n_neighbors=n_neighbors, 
                    n_components=n_components, 
                    metric='cosine').fit_transform(BERT_embeddings)
#%%
cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                        metric='euclidean',                      
                        cluster_selection_method='eom',
                        min_samples = min_samples,
                        gen_min_span_tree=True)

# %%
x = umap_embeddings[:1000,[0]]
y = umap_embeddings[:1000,[1]]

big_palette = palettes.Turbo256 * 2

cluster_colors = [big_palette[col]
                  if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
                  zip(cluster.labels_, cluster.probabilities_)][:1000]
# %%
p = figure(plot_width=800, plot_height=800)

# add a circle renderer with a size, color, and alpha
p.scatter(x.flatten(),y.flatten(), color = cluster_colors, size=10, alpha=0.5)

# show the results
show(p)



#%%
df['BERT_UMAP_cluster'] = cluster.fit_predict(umap_embeddings)

#%%

# FAST CLUSTERING


def community_detection(embeddings, threshold=0.75, min_community_size=10, init_max_size=1000):
    """
    Function for Fast Community Detection
    Finds in the embeddings all communities, i.e. embeddings that are close (closer than threshold).
    Returns only communities that are larger than min_community_size. The communities are returned
    in decreasing order. The first element in each list is the central point in the community.
    """

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
# %%
clusters = community_detection(BERT_embeddings, min_community_size=25, threshold=0.75)
# %%

#Print all cluster / communities

cluster_sentences = {}

for i, cluster in enumerate(clusters):
    print("\nCluster {}, #{} Elements ".format(i+1, len(cluster)))
    cluster_sentences[i] = ENTITY_CHUNKS[cluster[0]] 
    for sentence_id in cluster:
        df.at[sentence_id,'BERT_cos_cluster'] = i
        print("\t", ENTITY_CHUNKS[sentence_id])

# %%
cluster_sent_df = pd.DataFrame(cluster_sentences)
# %%
# df.to_pickle('/home/burtenshaw/now/potter_kg/data/clustered_26_11_2020.bin')
cluster_sent_df.to_pickle('/home/burtenshaw/now/potter_kg/data/cluster_sent_df.bin')
# %%
