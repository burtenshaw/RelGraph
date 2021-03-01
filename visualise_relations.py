#%%
import pandas as pd
import numpy as np
import os
import networkx as nx
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from lit_world import annotate

potter = pd.read_json('data/potter/test_preds.json')
pullman = pd.read_json('data/pullman/all_preds.json')

labels = to_keep = ['friend',
           'adversary',
           'teacher',
           'helper',
           'other',
           'family',
           'none',
           'ambivalent']
label_names = { n: v for n, v in enumerate(labels)}
potter['pred_relation'] = potter.y_pred.apply(lambda x : label_names[x])
pullman['pred_relation'] = pullman.pred_y.apply(lambda x : label_names[x])
#
#  %%

def single_graph_plot(test, characters = []):
    c_map = {'friend': '#a6cee3',
            'adversary': '#e31a1c',
            'teacher': '#b2df8a',
            'helper': '#33a02c',
            'other': '#ffffff',
            'family': '#1f78b4',
            'none': '#ffffff',
            'ambivalent': '#999999'}

    fig=plt.figure(figsize=(12,12))
    ax = plt.gca()
    if characters == []:
        characters = test.source_base.drop_duplicates().to_list()
    
    test = test.loc[(test.source_base.isin(characters)) & \
                     (test.target_base.isin(characters))] 

    
    edges = test[['source_base', 'target_base', 'pred_relation']]
    edges.columns = ['source', 'target', 'relation']
    edges['color'] = edges.relation.apply(lambda x : c_map[x])

    G = nx.from_pandas_edgelist(
        edges,
        edge_key="relation",
        edge_attr=True,
        create_using=nx.MultiGraph(),
    )

    pos = nx.spring_layout(G)
    colors = nx.get_edge_attributes(G,'color').values()

    nx.draw_networkx(G, pos, edge_color=colors, ax = ax)

    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in c_map.values()]
    plt.legend(markers, c_map.keys(), numpoints=1)

    return fig

def graph_plot(test, characters = []):
    c_map = {'friend': '#a6cee3',
            'adversary': '#e31a1c',
            'teacher': '#b2df8a',
            'helper': '#33a02c',
            'other': '#ffffff',
            'family': '#1f78b4',
            'none': '#ffffff',
            'ambivalent': '#999999'}

    books = test.book.drop_duplicates().to_list()

    fig, axs = plt.subplots(2,4, figsize=(40,25))
    ax = axs.flatten()

    if characters == []:
        characters = test.source_base.drop_duplicates().to_list()
    
    test = test.loc[(test.source_base.map(lambda x : x in characters)) & \
                     (test.target_base.map(lambda x : x in characters))] 

    for i, b in enumerate(books):
        fig.suptitle('Vertically stacked subplots')
        edges = test.loc[test.book == b][['source_base', 'target_base', 'pred_relation']]
        edges.columns = ['source', 'target', 'relation']
        edges['color'] = edges.relation.apply(lambda x : c_map[x])

        G = nx.from_pandas_edgelist(
            edges,
            edge_key="relation",
            edge_attr=True,
            create_using=nx.MultiGraph(),
        )

        pos = nx.spring_layout(G)
        colors = nx.get_edge_attributes(G,'color').values()

        nx.draw_networkx(G, pos, edge_color=colors, ax = ax[i])

        ax[i].set_axis_off()

    fig.delaxes(ax[-1])

    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in c_map.values()]
    plt.legend(markers, c_map.keys(), numpoints=1)

    plt.show()

def radar_plot(rel_df, src, trg, _cols = ['adversary', 'friend', 'teacher']):

    ''' harry to snape over the series '''


    scaler = MinMaxScaler()
    h_s = rel_df.loc[:,src,trg,:].reset_index()
    _cols = h_s.pred_relation.drop_duplicates().values
    h_s = h_s.pivot_table(index='book', values='value',columns='pred_relation')\
            [_cols]\
            .apply(lambda x : pd.Series(scaler.fit_transform(x.values.reshape(-1,1)).flatten()), axis=1)

    h_s.columns = _cols
    # h_s.loc[:] = scaler.fit_transform(h_s.values)

    # fig, axs = plt.subplots(2,4, figsize=(40,25))
    # axes = axs.flatten()
    fig=plt.figure(figsize=(20,10))
    # rect = [0.05, 0.05, 0.95, 0.95]
    # axes = [fig.add_axes(rect, projection="polar", label="axes%d" % i) for i in range(len(books))]
    books = h_s.index.values

    for i, b in enumerate(books):

        labels = h_s.columns.values
        angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        stats = h_s.loc[b].fillna(0.01).values
        # stats=np.concatenate((stats,[stats[0]]))
        # angles=np.concatenate((angles,[angles[0]]))

        # fig=plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111, polar=True)

        ax.plot(angles, stats, 'o-', linewidth=2)
        ax.fill(angles, stats, alpha=0.25)
        ax.set_thetagrids(angles * 180/np.pi, labels)
        ax.set_title('Book : %s : %s & %s' % (b, src, trg))
        ax.grid(True)

    return fig




# multiple characters


def plot_para(rel_df, source, targets, n_books=7, scaled = True):

    fig, axs = plt.subplots(3, figsize=(20,30))
    cols = []
    scaler = MinMaxScaler()
    cols = ['adversary', 'friend']

    for i, p in enumerate(targets):
        df = rel_df.loc[:,source,p,:].reset_index()
        df = df.pivot_table(index='book', values='value',columns='pred_relation')
        ax = axs[i]

        df = df[cols].apply(lambda x : pd.Series(scaler.fit_transform(x.values.reshape(-1,1)).flatten()), axis=1)
        df.columns = cols

        ax.set_title('%s to %s' % (source, p))

        df.reset_index(inplace = True)

    # ax = plt.figure()

        pd.plotting.parallel_coordinates(

            df[cols].reset_index(), 'index' , ax = ax, color= sns.color_palette("viridis", n_books)

        )
    
    plt.show()

#%%
pred_dir = os.path.join('data','predictions')

prdf = pd.DataFrame()

for _f in os.listdir(pred_dir):
    f = os.path.join(pred_dir, _f)
    _df = pd.read_json(f)
    prdf = pd.concat([prdf, _df])


potter['train'] = 'potter'
potter['test'] = 'potter'

prdf = pd.concat([prdf,potter])

prdf = prdf.loc[(prdf.label < 6) & (prdf.y_pred < 6)]
#%%
# from sklearn.metrics import precision_recall_fscore_support

# macro_clas_report = prdf.groupby(['train', 'test']).apply(lambda df : pd.Series(precision_recall_fscore_support(df.label, df.y_pred, average = 'macro')))
# macro_clas_report.columns = ['precision', 'recall', 'f1', 'support']
# macro_clas_report['support'] = prdf.groupby(['train', 'test']).count().level_0
# clas_report = prdf.groupby(['train', 'test']).apply(lambda df : pd.Series(precision_recall_fscore_support(df.label, df.y_pred)))

with open('paper/figures/macro_classification_report.md', 'w') as f:
    f.write(macro_clas_report.reset_index().to_markdown(index = False))
#%%
''' potter graph plots '''

graph_plot(potter)

#%%

for b in potter.book.drop_duplicates():
    fig = single_graph_plot(potter.loc[potter.book == b])
    fig.savefig('paper/figures/potter_relations_graph_%s.png' % b)

#%%
rel_df = potter.groupby(['book', 'source_base', 'target_base']).pred_relation.value_counts().to_frame()
rel_df.columns = ['value']

for src in ['harry']:
    for trg in ['snape', 'dumbledore', 'voldemort', 'dumbledore']:
        fig = radar_plot(rel_df, src, trg)
        fig.savefig('paper/figures/potter_relations_radar_%s_%s.png' % (src, trg))

#%%
source = 'harry'
targets = ['snape', 'dumbledore', 'voldemort']

plot_para(rel_df, source, targets)


# %%
''' pullman '''

# graph plots

for b in pullman.book.drop_duplicates():
    fig = single_graph_plot(pullman.loc[pullman.book == b],\
         characters=pullman.source_base.value_counts()[:5].index.values)
    fig.savefig('paper/figures/pullman_relations_graph_%s.png' % b)

#%%
#%%
rel_df = pullman.groupby(['book', 'source_base', 'target_base']).pred_relation.value_counts().to_frame()
rel_df.columns = ['value']

for src in ['lyra']:
    for trg in ['will', 'pan', 'asriel', 'mrscoulter']:
        fig = radar_plot(rel_df, src, trg)
        fig.savefig('paper/figures/pullman_relations_radar_%s_%s.png' % (src, trg))
#%%
source = 'lyra'
targets = pullman.source_base.value_counts().iloc[:5].index.values

plot_para(rel_df, source, targets)

# %%

''' ages '''

cdf = pd.read_json(os.path.join('data/potter','CHARACTERS_1_2_21.json'))
cdf = cdf.set_index('id')
cdf = cdf[~cdf.duplicated(keep='first')]
# %%
# potter = pd.merge(left=potter, 
#                   right=cdf[['life stage']].add_suffix('_source'), 
#                   left_on='source', 
#                   right_index=True)

# potter = pd.merge(left=potter, 
#                   right=cdf[['life stage']].add_suffix('_target'), 
#                   left_on='target', 
#                   right_index=True)

# potter = potter.rename(columns = {'life stage_source' : 'source_life_stage',
#                         'life stage_target' : 'target_life_stage'})
# potter = potter.dropna(subset=['source_life_stage', 'target_life_stage'])

def get_stage(a):
    
    if a < 3:
        return 'infant'
    elif a < 6:
        return 'child'
    elif a < 13:
        return 'latechild'
    elif a < 18:
        return 'adolescent'
    elif a < 30:
        return 'earlyadult'
    elif a < 50:
        return 'middleadult'
    else:
        return 'oldadult'
    
def stage_age(row):
    
    if not row.age and not row['life stage']:
        return np.nan
    elif not row.age:
        return row['life stage']
    elif not row['life stage']:
        return get_stage(row.age)
    else:
        return np.nan


age_dict = cdf.apply(stage_age, axis = 1).dropna().to_dict()

def lookup_age(id_):
    try:
        return age_dict[id_]
    except KeyError:
        try:
            return cdf.loc[id_]['life stage'].dropna().values[0]
        except:
            return np.nan

# %%
potter['source_life_stage'] = potter.source.apply(lookup_age)
potter['target_life_stage'] = potter.target.apply(lookup_age)

#%%
age_rel_df = potter.groupby(['book', 'source_life_stage', 'target_life_stage' ]).pred_relation.value_counts()
age_rel_df=age_rel_df.to_frame()
age_rel_df.columns = ['value']
age_rel_df = age_rel_df.reset_index()
age_rel_df['age_rel'] = age_rel_df.apply(lambda row : '%s_%s' % (row.source_life_stage,row.target_life_stage), axis = 1)
age_rel_df = age_rel_df.pivot_table(index=['book','age_rel'], columns='pred_relation', values='value')

# %%
source = 'adolescent'
targets = ['child', 'latechild', 'adult']

fig, axs = plt.subplots(7, figsize=(20,30))
# cols = ['adversary', 'friend']
scaler = MinMaxScaler()

for n in range(1,8):
    b = 'b%s' % n
    i = n-1

    fig=plt.figure(figsize=(20,10))

    ax = plt.gca()

    # for i, p in enumerate(targets):
    df = age_rel_df.loc[b].fillna(0)
    cols=df.columns
    # ax = axs[i]
    df = df.apply(lambda x : pd.Series(scaler.fit_transform(x.values.reshape(-1,1)).flatten()), axis=1)
    df.columns = cols

    ax.set_title('%s' % b)

    df.reset_index(inplace = True)

    # ax = plt.figure()

    pd.plotting.parallel_coordinates(

        df, 'age_rel' , ax = ax, color= sns.color_palette("viridis", 14)

    )

    fig.savefig('paper/figures/potter_age_relations_book_%s.png' % n)


# %%

age_rel_df = potter.groupby(['book', 'source_life_stage', 'target_life_stage' ]).pred_relation.value_counts()
age_rel_df=age_rel_df.to_frame()
age_rel_df.columns = ['value']
age_rel_df = age_rel_df.reset_index()
age_rel_df['age_rel'] = age_rel_df.apply(lambda row : '%s_%s' % (row.source_life_stage,row.target_life_stage), axis = 1)

#%%
focus_rels = ['adolescent_middleadult',
              'adolescent_adolescent',
              'adolescent_oldadult',
              'adolescent_latechild']

age_rel_df = age_rel_df.loc[age_rel_df.age_rel.isin(focus_rels)]\
            .pivot_table(index=['book','age_rel'], 
                         columns='pred_relation', 
                         values='value')


age_relations = age_rel_df.index.get_level_values(1).drop_duplicates().values
colors = sns.color_palette("Paired", age_relations.shape[0])

c_dict = dict(zip(age_relations, colors))

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=c, lw=4) for c in c_dict.values()]

for n in range(1,8):
    b = 'b%s' % n
    i = n-1

    fig=plt.figure(figsize=(20,10))

    ax = plt.gca()

    # for i, p in enumerate(targets):
    df = age_rel_df.loc[b].fillna(0)
    cols=df.columns
    # ax = axs[i]
    df = df.apply(lambda x : pd.Series(scaler.fit_transform(x.values.reshape(-1,1)).flatten()), axis=1)
    df.columns = cols
    fig.suptitle('Book : %s' % b)

    for idx,row in df.iterrows():
        ax = fig.add_subplot(111, polar=True)
        labels = df.columns.values
        angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        stats = row

        ax.plot(angles, stats, 'o-', linewidth=2, color=c_dict[idx])
        ax.fill(angles, stats, alpha=0.25, color=c_dict[idx])
        ax.set_thetagrids(angles * 180/np.pi, labels)
    
        # ax.legend()
    ax.legend(custom_lines, c_dict.keys())
    
    fig.savefig('paper/figures/potter_age_relations_book_%s.png' % n)


# %%
rels = pd.read_excel(os.path.join('data/potter', 'relations_annotated.xlsx'), index_col=0)
rels = rels[['harry','voldemort','hermione']].loc[['harry','voldemort','hermione']]

# lines = ax.plot(data)
# %%

with open('paper/figures/mock_matrix.md', 'w') as f:
    f.write(rels.to_markdown())

# %%
