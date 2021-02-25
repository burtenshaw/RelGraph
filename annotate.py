#%%
import pandas as pd
import os
import json
data_dir = os.path.join('data')

clusters = pd.read_json('data/cluster_sample.json')

df = pd.read_json(os.path.join(data_dir,'DATA_1_2_21.json'))
_pairs = df.explode('pairs').pairs.apply(pd.Series)
_df = pd.concat([df, _pairs], axis = 1)
# %%
clusters.iloc[0]
# %%
def get_base_data(row):
    _c = clusters.loc[
        (clusters.book == row.book) &
        (clusters.chapter == row.chapter) &
        (clusters.page == row.page) &
        (clusters.source == row.source) &
        (clusters.chunk == row.chunk) &
        (clusters.target == row.target)
    ]
    if _c.shape[0] > 0:
        return _c.cluster.iloc[0]
# %%
_df['cluster'] = _df.apply(get_base_data, axis = 1)
# %%
subset = _df.dropna(subset=['cluster'])
subset
#%%
def make_cluster_ents(row):
    sorted_ents = sorted(row.entities + [[row.cluster, row.chunk_ent]], key = lambda x : x[1][0])
    source = []
    target = []
    chunk = None 
    for label, loc in sorted_ents:
        if label == row.cluster:
            chunk = ['cluster_%s'%int(label), loc]
        elif label == row.source and not chunk:
            source = [label, loc]
        elif chunk and label == row.target:
            target = [label, loc]
    
    return [source, chunk, target]
    
subset['all_entities'] = subset.apply(make_cluster_ents, axis = 1)



# %%
def to_doccano(entities, text):
    ''' 
    {
    "text": "EU rejects German call to boycott British lamb.", 
    "labels": [ [0, 2, "ORG"], [11, 17, "MISC"], ... ]
    }      
    '''

    entities = [[start, end, label.upper()] for label, (start, end) in entities]
    return { "text" : str(text), "labels": entities}

# %%
output = subset.apply(lambda row : to_doccano(row.all_entities, row.paragraph), axis = 1).to_list()

output_path = os.path.join('data/doccano')

with open(os.path.join(output_path, 'subset_0.jsonl'), 'w') as f:
    for line in output:
        f.write('%s\n' % json.dumps(line))
# %%
