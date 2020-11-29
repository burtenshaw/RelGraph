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

data_dir = '/home/ben/now/potter_graph/'

cluster_top_sentences = pd.read_pickle(data_dir + 'data/cluster_sent_df.bin')
df = pd.read_pickle(data_dir + 'data/clustered_26_11_2020.bin')

df['entity_chunk'] = df[['source','chunk','target']]\
                    .dropna()\
                        .apply( lambda row : ' '.join(row.to_list())\
                                 if pd.notna(row.chunk) and len(row.chunk) > 2\
                                     else np.nan, axis = 1).dropna().drop_duplicates()

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
nodes = pd.read_csv(data_dir + 'data/characters.csv', index_col = 0)\
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

pos = pd.DataFrame(nx.circular_layout(G)).T
pos.columns = ['x', 'y']

nodes = pos.merge(nodes, how='left', left_index=True, right_index=True)

nodes = nodes[['x','y','title', 'life_stage']].rename(columns = {'life_stage':'type'})

nodes = nodes.dropna()
nodes['id'] = nodes.index

clu = lambda x : 'cluster%s' % int(x)
edges['type'] = edges.edge.apply(clu)
normo = lambda col : col-col.mean()/col.std()
outliers = lambda col : (col < col.quantile(.9)) | (col > col.quantile(0.1))
nodes.x = normo(nodes.x) * 1000
nodes.y = normo(nodes.y) * 1000
nodes.x = nodes[outliers(nodes.x)].x
nodes.y = nodes[outliers(nodes.y)].y

# %%
# making node types

idm = lambda x : '#%s' % x
_life_stages = nodes.type.drop_duplicates().to_list()
node_map = pd.DataFrame(index=_life_stages)
node_map['typeText'] = _life_stages
node_map['shapeId'] = node_map.typeText.apply(idm)
node_map['color'] = palettes.magma(len(node_map))
# nodes = nodes.merge(node_map, how='left', left_on='type', right_index=True)
# %%
#  making edge type

_clusters = edges.edge.drop_duplicates().to_list()
edge_map = pd.DataFrame(index=_clusters)
edge_map['typeText'] = [clu(x) for x in _clusters]
edge_map['shapeId'] = edge_map.typeText.apply(idm)
edge_map['color'] = palettes.viridis(len(edge_map))
# edges = edges.merge(edge_map, how='left', left_on='edge', right_index=True)
edge_map.index = edge_map['typeText']
# %%

def write_data(nodes, edges, node_map, edge_map, clusters = []):

    if clusters != []:
        print(  clusters)
        edges = edges[edges.type.map(lambda x : x in clusters)]
        nodes = nodes.loc[edges.target].drop_duplicates()\
                    .dropna()\
                    .append(nodes.loc[edges.source].drop_duplicates()\
                        .dropna())
        
    output_dict = {
        'nodes' : nodes.to_dict(orient='records'),
        'edges' : edges.to_dict(orient='records')
    }


    config_dict = {
        'NodeTypes' : node_map.to_dict(orient='records'),
        'NodeSubtypes': {},
        'EdgeTypes' : edge_map.to_dict(orient='records')
    }


    print(edges.index)

    return {'data' : output_dict, 'config' : config_dict}
    # %%

packet = write_data(nodes, edges, node_map, edge_map)

with open('/home/ben/now/potter_graph/app/src/data.json', 'w') as f:
    d = json.dumps(packet['data'])
    json.dump(d, f)

with open('/home/ben/now/potter_graph/app/src/config.json', 'w') as f:
    d = json.dumps(packet['config'])
    json.dump(d, f)


from flask import Flask
from flask import request, jsonify
from flask_cors import CORS

app = Flask(__name__)
app.config["DEBUG"] = True

@app.route('/cluster', methods=['GET','POST'])
def home():
    clusters = request.json
    packet = write_data(nodes, edges, node_map, edge_map, clusters)
    packet = jsonify(packet)
    return packet


CORS(app)

app.run(host='localhost', port=5001)



# %%
