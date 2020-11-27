# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
from bs4 import BeautifulSoup
from bs4 import element
import re
import logging;
import numpy as np

import spacy
from spacy.matcher import Matcher 
from spacy.tokens import Span 
import os
from tqdm import tqdm

nlp = spacy.load("en")

data_dir = '/home/burtenshaw/now/potter_kg/data/'
books_dir = '/home/burtenshaw/now/potter_kg/data/books'

excel_path = data_dir + 'ROWLING_harrypotterandthephilosophersstone_1997.xlsx'
xml = data_dir + 'ROWLING_harrypotterandthephilosophersstone_1997.xml'
characters = pd.read_excel(excel_path).set_index('id')

# charcters = pd.read_csv(data_dir + 'characters.csv', index_col=0).set_index('id')

c_list = characters.index.to_list()

# characters.to_csv('data/character.csv')
# %%
def parse_paragraphs(soup):
    paragraphs = soup.find_all('p')
    _collected = []

    for p in tqdm(paragraphs[1:]):

        packet = dict(zip(['book','chapter','page'], p.attrs['n'].split('-')))
        _p = p.text

        entities = []

        for rs in p.find_all('rs'):
            i = _p.find(rs.text)
            j = i + len(rs.text)
            _i = p.text.find(rs.text)
            _j = _i + len(rs.text)
            refs = rs['ref'].split(' ')
            entities.extend([[r,[_i,_j]] for r in refs])
            refs = ' & '.join(refs)
            _p = _p[:i] + '# %s #' % (refs) + _p[j:]

        packet.update({'paragraph': p.text})
        packet.update({'ref_paragraph': _p})
        packet.update({'entities': entities})
        packet.update({'sentences' : p.text.split(' ')})
        packet.update({'ref_sentences': _p.split('.')})

        _collected.append(packet)

    return _collected


def get_relation(sent):

    doc = nlp(sent)

    # Matcher class object 
    matcher = Matcher(nlp.vocab)

    #define the pattern 
    pattern = [{'DEP':'ROOT'}, 
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},  
            {'POS':'ADJ','OP':"?"}] 

    matcher.add("ref", None, pattern) 

    matches = matcher(doc)
    spans = [' '.join(doc[i:j].text) for _,i,j in matches]
    # ents.extend(spans)
    # ents = sorted(ents, key = lambda key : key[1][0])
    return spans

is_m = lambda y : True if not y.isspace() and len(y) > 0 and '.' not in y else False

def parse_relations(ents, sent):
    ents = sorted(ents, key = lambda key : key[1][0])
    relations_str = []
    for i, j in zip(ents, ents[1:]):
        x,y,z = i[0], sent[i[1][1]: j[1][0]], j[0]
        if is_m(y):
            relations_str.append({'source':z,'chunk':y,'target':x}) 
    return relations_str

def parse_books(books_dir):
    paths = os.listdir(books_dir)
    _dfs = []

    for p in paths:
        print(p)
        with open(os.path.join(books_dir,p),'r',encoding='utf8') as f:
            soup = BeautifulSoup(f)
        _dfs.append(pd.DataFrame(parse_paragraphs(soup)))
    
    df = pd.concat(_dfs).reset_index()

    df['relation_chunks'] = df.apply(lambda row : parse_relations(row.entities, row.paragraph), axis=1)
    
    df = pd.concat([df,df.relation_chunks.explode().apply(pd.Series)], axis=1)
    # df['src_chu_trg'] = df[['source','chunk','target']]\
    #                     .dropna()\
    #                         .apply( lambda row : ' '.join(row.to_list()), axis = 1)
    
    return df.reset_index()

df = parse_books(books_dir)

#%%
# df['linguistic_relation'] = df.relation_chunks.explode().dropna().apply(get_relation)

_df = pd.merge(df,characters.add_suffix('_source'), how='left', left_on='source', right_on='id')
df = pd.merge(_df,characters.add_suffix('_target'), how='left', left_on='target', right_on='id')

#%%
df.to_csv(data_dir + 'chunk_relations_26_11_2020.csv')
df.to_pickle(data_dir + 'chunk_relations_26_11_2020.bin')