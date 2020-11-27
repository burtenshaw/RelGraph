# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd

import numpy as np
import networkx as nx
from bokeh.io import output_file, show, output_notebook
from bokeh.plotting import figure, from_networkx
from bokeh.models import (BoxZoomTool, Circle, HoverTool,
                          MultiLine, Plot, Range1d, ResetTool,)

from collections import defaultdict
import matplotlib.pyplot as plt
from bokeh.plotting import show, ColumnDataSource
from bokeh.models import LabelSet
from bokeh import palettes

output_notebook()
cluster_sent_df = pd.read_pickle('/home/burtenshaw/now/potter_kg/data/cluster_sent_df.bin')

#%%
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
df['binned_lifestage_target'] = pd.cut(df.age_target,len(age_stages), labels=age_stages)
df['binned_lifestage_source'] = pd.cut(df.age_source,len(age_stages), labels=age_stages)
df['lifestage_target'] = df.apply(lambda row : row['life stage_target'] if row['life stage_target'] != np.nan else row.binnned_lifestage_target, axis=1)
df['lifestage_source'] = df.apply(lambda row : row['life stage_source'] if row['life stage_source'] != np.nan else row.binnned_lifestage_source, axis=1)

#%%

# mini lifestage cluster graph

clusters = df.BERT_cos_cluster.drop_duplicates().to_list()

sample = df[['lifestage_source','BERT_cos_cluster','lifestage_target']].dropna()\
          .rename(columns={'BERT_cos_cluster':'edge'})

G=nx.from_pandas_edgelist(sample, "lifestage_source", "lifestage_target", 
                          edge_attr=True)

G.remove_nodes_from(list(nx.isolates(G)))
plt.figure(figsize=(12,12))
pos = nx.spring_layout(G, k = 0.5) # k regulates the distance between nodes
nx.draw(G, with_labels=True, node_color='lightgreen', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
nx.draw_networkx_edges(G, pos=pos, arrowsize=50, min_target_margin=20)
labels = {k:v for k,v in sample.apply(lambda row : [(row.lifestage_source,row.lifestage_target), cluster_sent_df.iloc[int(row.edge)]], axis = 1 ).to_list()}
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels,font_color='red')
plt.show()

# plot = figure(x_range=(-1.1,1.1), 
#                 y_range=(-1.1,1.1),
#               tools="")

# graph = from_networkx(G, nx.spring_layout, center=(0,0))

# plot.renderers.append(graph)
# node_hover_tool = HoverTool(tooltips=[("index", "@index"),("edge", "@edge")])
# plot.add_tools(node_hover_tool,BoxZoomTool(), ResetTool())

# labels = LabelSet(x=transform('start', xcoord),
#                   y=transform('start', ycoord),
#                   text='names', text_font_size="12px",
#                   x_offset=5, y_offset=5,
#                   source=source, render_mode='canvas')

# p.add_layout(labels)

# show(plot)

# %%

# mega graph for react app

out = df.dropna(subset=['source','BERT_cos_cluster','target'])\
          .rename(columns={'BERT_cos_cluster':'edge'})\
          .drop_duplicates(subset=['source','edge','target'])

edges = out[['source','edge','target']]
edges.edge = edges.edge.apply(lambda x : str(int(x)))
#%%
G=nx.from_pandas_edgelist(edges, "source", "target", 
                          edge_attr=True)

pos = pd.DataFrame(nx.spring_layout(G)).T

pos.columns = ['x', 'y']
#%%
nodes = pd.read_csv('/home/burtenshaw/now/potter_kg/data/characters.csv', index_col = 0)\
            .applymap(lambda x : x if x != 'none' else np.nan)
nodes['binned_lifestage'] = pd.cut(nodes.age,len(age_stages), labels=age_stages)
nodes['life stage'] = nodes.apply(lambda row : row['life stage'] if row['life stage'] != np.nan else row.binnned_lifestage, axis=1)

nodes.columns = ['title', 'individual_group', 'species', 'gender', 'ethnicity',
       'relation_to_protagonist', 'relation_to_protagonist_2',
       'age', 'life_stage', 'certainty', 'notes', 'binned_lifestage']

nodes['id'] = nodes.index
nodes = pos.merge(nodes, how='left', left_index=True, right_index=True)
nodes = nodes[['x','y','title', 'life_stage']].rename({'life_stage':'sample'})

nodes = nodes.dropna().drop(columns=['x','y'])
nodes['id'] = nodes.index

# \ solve missing annotation
# nodes = nodes.append(pd.concat([edges.target.loc[edges.target.apply(\
#                 lambda x : True if x in nodes.id else False)],\
#                     edges.source.loc[edges.source.apply(\
#                         lambda x : True if x in nodes.id else False)]])\
#                         .drop_duplicates().to_frame().set_index(0))
# nodes = nodes.applymap(lambda x : x if type(x) == str or x == 'NaN' else 'empty')

edges = edges[edges.source.map(lambda x : x in nodes.index)]\
            [edges.target.map(lambda x : x in nodes.index)]
#%%
# making node types

_life_stages = nodes.life_stage.drop_duplicates().to_list()
node_map = pd.DataFrame(index=_life_stages)
node_map['class'] = _life_stages
node_map['color'] = palettes.magma(len(node_map))
nodes = nodes.merge(node_map, how='left', left_on='life_stage', right_index=True)
# %%
#  making edge type

_clusters = edges.edge.drop_duplicates().to_list()
edge_map = pd.DataFrame(index=_clusters)
edge_map['class'] = [str(int(x)) for x in _clusters]
edge_map['color'] = palettes.viridis(len(node_map))
edges = edges.merge(edge_map, how='left', left_on='edge', right_index=True)
# edge_map.index = edge_map['shapeId']
# %%
output_dict = {
    'nodes' : nodes.to_dict(orient='records'),
    'edges' : edges.to_dict(orient='records')
}

import json

with open('/home/burtenshaw/now/potter_kg/app/src/data.json', 'w') as f:
    d = json.dumps(output_dict)
    json.dump(d, f)
# %%

# config_dict = {
#     'NodeTypes' : node_map.to_dict(orient='index'),
#     'NodeSubtypes': {},
#     'EdgeTypes' : edge_map.to_dict(orient='index')
# }

# with open('/home/burtenshaw/now/potter_kg/app/src/config.json', 'w') as f:
#     d = json.dumps(config_dict)
#     json.dump(d, f)
#     # %%
