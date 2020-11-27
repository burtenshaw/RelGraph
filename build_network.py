# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd

import numpy as np
import networkx as nx

from collections import defaultdict
import matplotlib.pyplot as plt
from bokeh import palettes
import json

cluster_sent_df = pd.read_pickle('/home/burtenshaw/now/potter_kg/data/cluster_sent_df.bin')
df = pd.read_pickle('/home/burtenshaw/now/potter_kg/data/clustered_26_11_2020.bin')

age_stages =[
            'infant',
            'child',
            'latechild',
            'adolescent',
            'adult',
            'earlyadult',
            'middleadult',
            'oldadult']
#%%
# mega graph for react app
nodes = pd.read_csv('/home/burtenshaw/now/potter_kg/data/characters.csv', index_col = 0)\
            .applymap(lambda x : x if x != 'none' else np.nan)

nodes['binned_lifestage'] = pd.cut(nodes.age,len(age_stages), labels=age_stages)
nodes['life stage'] = nodes.apply(lambda row : row['life stage'] if row['life stage'] != np.nan else row.binnned_lifestage, axis=1)

nodes.columns = ['title', 'individual_group', 'species', 'gender', 'ethnicity',
       'relation_to_protagonist', 'relation_to_protagonist_2',
       'age', 'life_stage', 'certainty', 'notes', 'binned_lifestage']

nodes['id'] = nodes.index

out = df.dropna(subset=['source','BERT_cos_cluster','target'])\
          .rename(columns={'BERT_cos_cluster':'edge'})\
          .drop_duplicates(subset=['source','edge','target'])

edges = out[['source','edge','target']]
edges.edge = edges.edge.apply(lambda x : str(int(x)))

edges = edges[edges.source.map(lambda x : x in nodes.index)]\
            [edges.target.map(lambda x : x in nodes.index)]

G=nx.from_pandas_edgelist(edges, "source", "target", 
                          edge_attr=True)

pos = pd.DataFrame(nx.spring_layout(G)).T
pos.columns = ['x', 'y']

nodes = pos.merge(nodes, how='left', left_index=True, right_index=True)

nodes = nodes[['x','y','title', 'life_stage']].rename(columns = {'life_stage':'type'})

nodes = nodes.dropna()
nodes['id'] = nodes.index

edges['type'] = edges.edge 
normo = lambda col : col-col.mean()/col.std()
outliers = lambda col : (col < col.quantile(.9)) | (col > col.quantile(0.1))
nodes.x = normo(nodes.x) * 10000
nodes.y = normo(nodes.y) * 10000
nodes.x = nodes[outliers(nodes.x)].x
nodes.y = nodes[outliers(nodes.y)].y
#%%

output_dict = {
    'nodes' : nodes.to_dict(orient='records'),
    'edges' : edges.to_dict(orient='records')
}

with open('/home/burtenshaw/now/potter_kg/app/src/data.json', 'w') as f:
    d = json.dumps(output_dict)

    json.dump(d, f)
# %%
# making node types

_life_stages = nodes.type.drop_duplicates().to_list()
node_map = pd.DataFrame(index=_life_stages)
node_map['typeText'] = _life_stages
node_map['shapeId'] = node_map.typeText.apply(lambda x : '#%s' % x)
node_map['color'] = palettes.magma(len(node_map))
# nodes = nodes.merge(node_map, how='left', left_on='type', right_index=True)
# %%
#  making edge type

_clusters = edges.edge.drop_duplicates().to_list()
edge_map = pd.DataFrame(index=_clusters)
edge_map['typeText'] = [str(int(x)) for x in _clusters]
edge_map['shapeId'] = edge_map.typeText.apply(lambda x : '#%s' % x)
edge_map['color'] = palettes.viridis(len(edge_map))
# edges = edges.merge(edge_map, how='left', left_on='edge', right_index=True)
# edge_map.index = edge_map['shapeId']
# %%

config_dict = {
    'NodeTypes' : node_map.to_dict(orient='records'),
    'NodeSubtypes': {},
    'EdgeTypes' : edge_map.to_dict(orient='records')
}

with open('/home/burtenshaw/now/potter_kg/app/src/config.json', 'w') as f:
    d = json.dumps(config_dict)
    json.dump(d, f)
    # %%
