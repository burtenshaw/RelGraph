# %%
import pandas as pd
import collections
import numpy as np
import networkx as nx
from spacy.lang.en.stop_words import STOP_WORDS
import string
from collections import defaultdict
import matplotlib.pyplot as plt
from bokeh import palettes
import json
import argparse
import re


if 'IPyKernelApp' not in get_ipython().config:
    class DumArgs:
        data = '/home/burtenshaw/now/potter_kg/data/'
        out = '/home/burtenshaw/now/potter_kg/app/src/'
    args = DumArgs()
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='source of input files', default = '/home/burtenshaw/now/potter_kg/data/')
    parser.add_argument('--out', help='target of output files', default= '/home/burtenshaw/now/potter_kg/app/src/')
    args = parser.parse_args()


df = pd.read_pickle(args.data + 'cluster_sent_df.bin')
characters = pd.read_csv(args.data + 'characters.csv', index_col = 0)      
CHARACTER_DICT = characters['life stage'].to_dict()

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

def add_entity_chunk(df):

    ''' add a [source - chunk - target]  field to the main df . 
        Should not be used for presentation . '''

    p = re.compile(r"[\d{}]+$".format(re.escape(string.punctuation)))
    pattern = lambda item : False if p.match(item) else True

    df['entity_chunk'] = df[['source','chunk','target']]\
                    .dropna()\
                    .apply( lambda row : ' '.join(row.to_list())\
                    if pd.notna(row.chunk) and len(row.chunk) > 2\
                                            and pattern(row.chunk)\
                    else np.nan, axis = 1).dropna().drop_duplicates()
    return df

df = add_entity_chunk(df)
#%%
# mega graph for react app

def make_nodes_df(nodes):

    ''' convert character df into nodes '''
    nodes = nodes.applymap(lambda x : x if x != 'none' else np.nan)
    nodes['binned_lifestage'] = pd.cut(nodes.age,len(age_stages), labels=age_stages)
    nodes['life stage'] = nodes.apply(lambda row : row['life stage'] if row['life stage'] != np.nan else row.binnned_lifestage, axis=1)

    nodes.columns = ['title', 'individual_group', 'species', 'gender', 'ethnicity',
        'relation_to_protagonist', 'relation_to_protagonist_2',
        'age', 'life_stage', 'certainty', 'notes', 'binned_lifestage']

    nodes['id'] = nodes.index

    return nodes 

nodes = make_nodes_df(characters)
#%%
def make_edges_df(df, nodes_index):

    ''' make network edges from the annotation df '''

    edges = df.dropna(subset=['source','BERT_cos_cluster','target'])\
            .rename(columns={'BERT_cos_cluster':'edge'})\
            .drop_duplicates(subset=['source','edge','target'])\
                [['source','edge','target', 'entity_chunk']]

    edges.edge = edges.edge.apply(lambda x : str(int(x)))

    # align with nodes df

    edges = edges[edges.source.map(lambda x : x in nodes_index)]\
                [edges.target.map(lambda x : x in nodes_index)]

    edges['type'] = edges.edge.apply(lambda x : 'cluster%s' % int(x))

    return edges

edges = make_edges_df(df, nodes.index)
#%%
# creat graph

def create_positions_from_edges(edges, nodes, big_circle = False):
        
    G=nx.from_pandas_edgelist(edges, "source", "target", 
                            edge_attr=True)
    normo = lambda col : col-col.mean()/col.std()
    outliers = lambda col : (col < col.quantile(.9)) | (col > col.quantile(0.1))
    if big_circle:            
        pos = nx.circular_layout(G) 

        # creat fictive circle positions for clustering
        life_stages_map = {l : n for n,l in enumerate(set(nodes.life_stage.to_list()))}
        angs = np.linspace(0, 2*np.pi, 1+len(life_stages_map))
        rad = 30
        repos = [np.array([rad*np.cos(ea), rad*np.sin(ea)]) for ea in angs]
        circ = nodes.apply(lambda x : repos[life_stages_map[x.life_stage]], axis = 1, result_type='expand')
        circ.columns = ['x','y']

        # geometrically group real valuse 
        pos = pd.DataFrame(pos).T
        pos.columns = ['x', 'y']
        normo = lambda col : col-col.mean()/col.std()
        outliers = lambda col : (col < col.quantile(.9)) | (col > col.quantile(0.1))
        pos.x, pos.y = normo(pos.x), normo(pos.y)
        pos.x, pos.y = pos[outliers(pos.x)].x, pos[outliers(pos.y)].y
        pos['x'], pos['y'] = (pos.x * circ.x), (pos.y * circ.y)

    else:
        pos = nx.spring_layout(G)
        pos = pd.DataFrame(pos).T
        pos.columns=['x','y']
        pos.x , pos.y = (pos.x * 10000) , (pos.y * 10000)
        pos.x, pos.y = pos[outliers(pos.x)].x, pos[outliers(pos.y)].y

    return pos

pos = create_positions_from_edges(edges, nodes)

nodes = pos.merge(nodes, how='left', left_index=True, right_index=True)\
            [['x','y','title', 'life_stage']]\
            .rename(columns = {'life_stage':'type'}).dropna()

# %%
def create_nodes_edge_config(nodes, edges):

    ''' make the config file for d3 visualisations '''

    # making node types
    idm = lambda x : '#%s' % x
    clu = lambda x : 'cluster%s' % int(x)

    _life_stages = nodes.type.drop_duplicates().to_list()
    node_map = pd.DataFrame(index=_life_stages)
    node_map['typeText'] = _life_stages
    node_map['shapeId'] = node_map.typeText.apply(idm)
    node_map['color'] = palettes.magma(len(node_map))
    # nodes = nodes.merge(node_map, how='left', left_on='type', right_index=True)
    #  making edge type
    edge_map = pd.DataFrame(index=edges.edge.drop_duplicates())
    edge_map['cluster'] = edge_map.index
    edge_map['typeText'] = edge_map.cluster.apply(clu)
    edge_map['shapeId'] = edge_map.typeText.apply(idm)
    edge_map['color'] = palettes.viridis(len(edge_map))
    # edges = edges.merge(edge_map, how='left', left_on='edge', right_index=True)
    edge_map.index = edge_map['typeText']

    return node_map, edge_map

node_map, edge_map = create_nodes_edge_config(nodes, edges)
# %%

def wrangle_clusters(df, nodes, edges):

    ''' prepare the parsed clusters for visualisation '''

    name_list = list(set([w.lower() for n in nodes.title.str.split(' ').to_list() for w in n]))

    df['book_info'] = df[['book','chapter','page']].to_dict(orient='records')
    _columns = ['paragraph', 'source', 'target', 'chunk', 'book_info']
    df.dropna(subset=_columns, inplace=True)
    clu = df.groupby('BERT_cos_cluster').first()[_columns]

    for col in _columns:
        clu[col] = df[_columns + ['BERT_cos_cluster']].groupby('BERT_cos_cluster')[col].apply(list)
    
    kw = lambda w : w not in STOP_WORDS \
                    and len(w) > 2 and not w in \
                        string.punctuation and w.lower() \
                            not in name_list

    clu['key_words'] = clu.paragraph.apply( lambda x : ' '.join(x))\
                        .str.replace('[^\w\s]','').apply( lambda x : \
                            [w for w in x.split(' ') if kw(w) ][:20])

    clu['size'] = clu.paragraph.apply(len)


    clu['relations'] = clu[['book_info','source','target', 'paragraph', 'chunk']].apply(lambda row : \
                            [{'s': s, 't' : t, 'p': p, 'rc' : rc, 'bi': bi} for s,t,p,rc, bi in \
                            list(zip(row.source, row.target, row.paragraph, row.chunk, row.book_info))], axis=1)

    
    relations_dict = clu.to_dict(orient='records')

    return relations_dict


relations_dict = wrangle_clusters(df, nodes, edges)
#%%

def make_age_resource(nodes, edges):
    nodes['characters'] = nodes.index

    ages = nodes[['characters', 'type']].rename(columns={'type':'age'})\
            .merge(edges.rename(columns={'type':'relation'}), \
                                how='left', 
                                left_index=True, 
                                right_on='source')\
            .fillna('')\
            .groupby('age').agg(list)

    ages['relations'] = ages[['source','target', 'relation', 'entity_chunk']].apply(lambda row : \
                        [{'s': s, 't' : t, 'r': r, 'rc' : rc} for s,t,r,rc in \
                        list(zip(row.source, row.target, row.relation, row.entity_chunk))], axis=1)
    
    ages['relation_frequency'] = ages.relation.apply(collections.Counter).apply(dict)

    ages['frequency'] = ages.relations.apply(len)

    ages['characters'] = ages.characters.apply(set).apply(list)

    ages['n_characters'] = ages.characters.apply(len)

    ages['id'] = ages.index

    return ages[['id','characters', 'relations', 'relation_frequency', 'n_characters', 'frequency']]

ages_dict = make_age_resource(nodes, edges).to_dict(orient='records')

#%%
def grouped_data_by_age(nodes, edges):

    ''' take the big df's and group them by age to make dfs of just life stage '''

    edges['source'] = edges.source.apply(lambda x : CHARACTER_DICT[x])
    edges['target'] = edges.target.apply(lambda x : CHARACTER_DICT[x])
    nodes = nodes.groupby('type').first()
    nodes['title'] = nodes['id'] = nodes.index

    pos = create_positions_from_edges(edges, nodes[['title', 'id']])

    nodes = pos.merge(nodes[['title', 'id']], how='left', left_index=True, right_index=True)

    nodes = nodes.loc[[i for i in nodes.index if edges[edges.source == i].shape[0] > 5]]
    edges = edges[edges.source.map(lambda x : x in nodes.index) | edges.target.map(lambda x : x in nodes.index)]
    return nodes.dropna(), edges.dropna()

nodes, edges = grouped_data_by_age(nodes, edges)

#%%
def make_connection_resource(conn):

    conn['connection'] = conn.apply(lambda row : '%s_%s' % (row.source, row.target), axis = 1)
    conn = conn.groupby('connection').agg(list)
    conn['relation_frequency'] = conn.type.apply(collections.Counter).apply(dict)

    conn['relations'] = conn[['source','target', 'type', 'entity_chunk']].apply(lambda row : \
                    [{'s': s, 't' : t, 'r': r, 'rc' : rc} for s,t,r,rc in \
                    list(zip(row.source, row.target, row.type, row.entity_chunk))], axis=1)
    conn['id'] = conn.index
    conn['frequency'] = conn.relations.apply(len)
    return conn[['relation_frequency', 'relations', 'id']]

conn = make_connection_resource(edges).to_dict(orient='records')

# %%

# %%

def make_d3_data(nodes, edges, node_map, edge_map, clusters = []):

    ''' create a packet of data and config files for d3 vis '''

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


packet = make_d3_data(nodes, edges, node_map, edge_map)


#%%

def write_output(data_dir, data, config, relations, ages, conn):
    
    with open(data_dir + 'data.json', 'w') as f:
        d = json.dumps(data)
        json.dump(d, f)

    with open(data_dir + 'config.json', 'w') as f:
        d = json.dumps(config)
        json.dump(d, f)

    with open(data_dir + 'relations.json', 'w') as f:
        d = json.dumps(relations)
        json.dump(d, f)

    with open(data_dir + 'ages.json', 'w') as f:
        d = json.dumps(ages)
        json.dump(d, f)

    with open(data_dir + 'connections.json', 'w') as f:
        d = json.dumps(conn)
        json.dump(d, f)


#%%
write_output(args.out, packet['data'], packet['config'], relations_dict, ages_dict, conn)


# %%

# %%
