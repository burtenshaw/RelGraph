# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd

import numpy as np
import networkx as nx
from spacy.lang.en.stop_words import STOP_WORDS
import string
from collections import defaultdict
import matplotlib.pyplot as plt
from bokeh import palettes
import json

data_dir = '/home/ben/now/potter_graph/'
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

edges = out[['source','edge','target', 'entity_chunk']].dropna()
edges.edge = edges.edge.apply(lambda x : str(int(x)))

edges = edges[edges.source.map(lambda x : x in nodes.index)]\
            [edges.target.map(lambda x : x in nodes.index)]

# creat graph

def create_positions_from_edges(edges, nodes):
        
    G=nx.from_pandas_edgelist(edges, "source", "target", 
                            edge_attr=True)
    pos = nx.circular_layout(G)  

    # creat fictive circle positions for clustering
    life_stages_map = {l : n for n,l in enumerate(set(nodes.life_stage.to_list()))}
    angs = np.linspace(0, 2*np.pi, 1+len(life_stages_map))
    rad = 3000
    repos = [np.array([rad*np.cos(ea), rad*np.sin(ea)]) for ea in angs]
    circ = nodes.apply(lambda x : repos[life_stages_map[x.life_stage]], axis = 1, result_type='expand')
    circ.columns = ['x','y']

    # geometrically group real valuse 
    pos = pd.DataFrame(pos).T
    pos.columns = ['x', 'y']
    normo = lambda col : col-col.mean()/col.std()
    outliers = lambda col : (col < col.quantile(.9)) | (col > col.quantile(0.1))
    pos.x = normo(pos.x)
    pos.y = normo(pos.y)
    pos.x = pos[outliers(pos.x)].x
    pos.y = pos[outliers(pos.y)].y

    pos['x'], pos['y'] = (pos.x * circ.x), (pos.y * circ.y)

    return pos

pos = create_positions_from_edges(edges, nodes)

nodes = pos.merge(nodes, how='left', left_index=True, right_index=True)

#%%
nodes = nodes[['x','y','title', 'life_stage']].rename(columns = {'life_stage':'type'})

nodes = nodes.dropna()
nodes['id'] = nodes.index

clu = lambda x : 'cluster%s' % int(x)
edges['type'] = edges.edge.apply(clu)


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
edge_map = pd.DataFrame(index=edges.edge.drop_duplicates())
edge_map['cluster'] = edge_map.index
edge_map['typeText'] = edge_map.cluster.apply(clu)
edge_map['shapeId'] = edge_map.typeText.apply(idm)
edge_map['color'] = palettes.viridis(len(edge_map))
# edges = edges.merge(edge_map, how='left', left_on='edge', right_index=True)
edge_map.index = edge_map['typeText']
# %%

clu = pd.read_pickle(data_dir + 'data/cluster_sent_df.bin')

name_list = list(set([w.lower() for n in nodes.title.str.split(' ').to_list() for w in n]))

clu.columns = ['top_sentence']

clu['cluster'] = clu.index

clu['entity_chunk'] = clu.cluster.astype(str).apply(lambda x : \
                        edges.loc[edges.edge == x]\
                        .entity_chunk.dropna().to_list())

clu['source'] = clu.cluster.astype(str).apply(lambda x : \
                        edges.loc[edges.edge == x]\
                        .source.dropna().to_list())

clu['target'] = clu.cluster.astype(str).apply(lambda x : \
                        edges.loc[edges.edge == x]\
                        .target.dropna().to_list())

clu['main_index'] = clu.cluster.astype(str).apply(lambda x : \
                        edges.loc[edges.edge == x]\
                        .target.dropna().index.to_list())


clu['paragraphs'] = clu.main_index.apply(lambda x : \
                        df.loc[x].paragraph.to_list())


clu['chunk'] = clu.main_index.apply(lambda x : \
                        df.loc[x].chunk.to_list())

kw = lambda w : w not in STOP_WORDS and len(w) > 2 and not w in string.punctuation and w.lower() not in name_list

clu['key_words'] = clu.paragraphs.apply( lambda x : ' '.join(x))\
                    .str.replace('[^\w\s]','').apply( lambda x : \
                        [w for w in x.split(' ') if kw(w) ][:20])

clu['size'] = clu.paragraphs.apply(len)

#%%
if clu.shape == clu.loc[clu[['entity_chunk','source','target']]\
    .apply( lambda r : len(r.entity_chunk) == \
                        len(r.source) == \
                        len(r.target),axis = 1)].shape:
                        print('clean clusters')
else:
    print('dirty clusters')

clu['relations'] = clu[['entity_chunk','source','target', 'paragraphs', 'chunk']].apply(lambda row : \
                        [{'s': s, 'c' : c, 't' : t, 'p': p, 'rc' : rc} for s,c,t,p,rc in \
                        list(zip(row.source,row.entity_chunk, row.target, row.paragraphs, row.chunk))], axis=1)


clu.index = clu.cluster
clu = clu[['relations','top_sentence', 'size', 'key_words']]
relations_dict = clu.to_dict(orient='index')

# %%

def write_data(nodes, edges, node_map, edge_map, clusters = []):

    if clusters != []:
        print(clusters)
        # edges = edges[edges.type.map(lambda x : x in clusters)]
        # nodes = nodes.loc[edges.target].drop_duplicates()\
        #             .dropna()\
        #             .append(nodes.loc[edges.source].drop_duplicates()\
        #             .dropna())

        # G=nx.from_pandas_edgelist(edges, "source", "target", 
        #                   edge_attr=True)

        # pos = pd.DataFrame(nx.spring_layout(G)).T
        # pos.columns = ['x', 'y']

        # nodes['x'] = pos.x
        # nodes['y'] = pos.y
        # nodes.x = normo(nodes.x) * 10000
        # nodes.y = normo(nodes.y) * 10000
        # nodes.x = nodes[outliers(nodes.x)].x
        # nodes.y = nodes[outliers(nodes.y)].y

        # print(' nodes : %s \n edges : %s' % (len(nodes), len(edges)))
        
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

with open('/home/ben/now/potter_graph/app/src/relations.json', 'w') as f:
    d = json.dumps(relations_dict)
    json.dump(d, f)
#%%
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

app.run(host='localhost', port=5000)



# %%
