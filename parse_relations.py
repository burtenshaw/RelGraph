# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from bs4 import element
import re
import logging
import os
from tqdm import tqdm

import spacy
from spacy.tokenizer import Tokenizer
from spacy.matcher import Matcher 
from spacy.lang.en import English
from spacy.tokens import Span 

nlp = spacy.load("en_core_web_sm")
_nlp = English()
tokenizer = Tokenizer(_nlp.vocab)

# load resources
# data_dir = os.path.join('data')
# books_dir = os.path.join(data_dir, 'books')
# characters_dir = os.path.join(data_dir, 'characters')

# %%
def parse_paragraphs(soup, split_on = 'p', book = 0):
    paragraphs = soup.find_all(split_on)
    collected = []

    for p in tqdm(paragraphs[1:]):

        if split_on == 'p':
            packet = dict(zip(['book','chapter','page'], p.attrs['n'].split('-')))
        else:
            packet = dict(book = book, chapter = 0, page = 0)

        entities = []

        for rs in p.find_all('rs'):
            begin_char = p.text.index(rs.text)
            end_char = len(rs.text) + begin_char
            refs = rs['ref'].split(' ')
            text = p.text[begin_char:end_char]
            entities.extend([[r,[begin_char,end_char], text] for r in refs])

        packet.update({'paragraph': p.text})
        packet.update({'entities': entities})

        if split_on == 'p':
            sentences =  p.text.split('.')
        else:
            sentences = [p.text]

        packet.update({'sentences': sentences})
        packet.update({'tokens' : [[str(w) for w in tokenizer(sent)] for sent in sentences]})
        
        collected.append(packet)

    return collected

def parse_chapters(soup, book = 0):
    chapters = soup.find_all('div')
    collected = []
    paragraph = 0

    for p in tqdm(chapters):
        packet = dict(chapter = p.attrs['n'],
                      book = book)
        entities = []

        for rs in p.find_all('rs'):
            begin_char = p.text.index(rs.text)
            end_char = len(rs.text) + begin_char
            refs = rs['ref'].split(' ')
            text = p.text[begin_char:end_char]
            entities.extend([[r,[begin_char,end_char], text] for r in refs])

        packet.update({'paragraph': p.text})
        packet.update({'entities': entities})

        sentences =  p.text.split('.')

        packet.update({'sentences': sentences})
        packet.update({'tokens' : [[str(w) for w in tokenizer(sent)] for sent in sentences]})
        
        collected.append(packet)

    return collected

def linguistic_relations(sent):

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

is_good = lambda chunk : not chunk.isspace() and \
                         len(chunk) > 2 and \
                         '.' not in chunk and \
                         len(chunk.split(' ')) > 1 and \
                         chunk.split(' ')

def parse_chunks(ents, sent):

    ents = sorted(ents, key = lambda key : key[1][0])
    relations_str = []
    chunk_ents = []

    for _source, _target in zip(ents, ents[1:]):
        source,chunk,target = _source[0], sent[_source[1][1]: _target[1][0]], _target[0]
        source_txt, target_txt = _source[2], _target[2]

        if is_good(chunk) and source != target:
            relations_str.append({'source':source,
                                  'chunk':chunk,
                                  'target':target,
                                  'chunk_ent' : [_source[1][1], _target[1][0]],
                                  'source_txt' : source_txt,
                                  'target_txt' : target_txt})
    return relations_str

def parse_simple_pairs(row):
    paragraph_entities = [e[0] for e in row.entities]
    pairs = []

    for sentence in row.ref_tokens:
        ents = [word for word in sentence if word in paragraph_entities]
        for source, target in zip(ents, ents[1:]):
            _pair = {
                'chunk' : sentence,
                'source' : source,
                'target' : target
            }
            pairs.append(_pair)
            
    return pairs


def parse_books(books_dir, 
                do_simple = False, 
                do_ling = False, 
                do_chunks = False, 
                split_on = 'p'):

    paths = os.listdir(books_dir)
    _dfs = []

    for b, p in enumerate(paths):
        print(p)
        with open(os.path.join(books_dir,p),'r',encoding='utf8') as f:
            soup = BeautifulSoup(f)

        _df = pd.DataFrame(parse_paragraphs(soup, split_on=split_on, book=b))
        _df['file_name'] = p
        _dfs.append(_df)
    
    df = pd.concat(_dfs).sort_values('book').reset_index()

    if do_chunks:
        df['pairs'] = df.apply(lambda row : parse_chunks(row.entities, row.paragraph), axis=1)
        
    elif do_ling:
        _linguistic_relations = df.relation_chunks.explode().dropna().apply(linguistic_relations)
        df = pd.concat([df,_linguistic_relations.apply(pd.Series)], axis=1)

    elif do_simple:
        df['pairs'] = df.apply(parse_simple_pairs, axis = 1)

        
    return df

def build_character_data(characters_dir):
    paths = os.listdir(characters_dir)
    _dfs = []

    for p in paths:
        print(p)

        _df = pd.read_excel(os.path.join(characters_dir, p))
        _df['file_name'] = p
        _dfs.append(_df)
    
    df = pd.concat(_dfs).reset_index()

    df['book'] = df.file_name.apply(lambda f : BOOK_MAP[f])

    return df

def get_character_info(cdf, book_path, character_entity):
    return cdf.loc[(cdf.file_name == book_path) & (cdf.id == character_entity)]

   
def make_train_data(df):

    pairs = df.explode('pairs').dropna().pairs.apply(pd.Series)
    pairs['book'] = df.book
    pairs['chapter'] = df.chapter
    pairs['page'] = df.page
    pairs['train_data'] = pairs.apply(lambda row :[row.source_txt] + [row.chunk] + [row.target_txt], axis = 1)
    pairs['source_base'] = pairs.source.str.strip('0123456789')
    pairs['target_base'] = pairs.target.str.strip('0123456789')
    
    return pairs

# %%
# df = parse_books(
#     books_dir,
#     do_chunks=True
# )
# pairs = make_train_data(df)

# BOOK_MAP =  book_map = {v : k for k,v in df.groupby('book')\
#             .first().file_name.str.split('.')\
#             .apply(lambda x : '%s.xlsx' % x[0])\
#             .to_dict().items()}

# cdf = build_character_data(characters_dir)
# pairs.reset_index().to_json(os.path.join(data_dir,'PAIRS.json'))
#%%

def make_rel_matrix(cdf, pairs, n_characters = 10):
    frequent = pairs.source_base.value_counts()[:n_characters].index

    cdf['id_base'] = cdf.id.str.strip('0123456789')
    rdf = cdf.loc[cdf.id_base.map(lambda x : x in frequent)]

    rdf = rdf.groupby(['book', 'id_base']).first()[['relation to protagonist']]\
            .rename(columns={'relation to protagonist' : 'harry'})\
            .reindex(columns = frequent)

    books = rdf.index.get_level_values(0).drop_duplicates()

    _rels = [
    'friend',
    'family',
    'helper',
    'teacher'
    'ambivalent',
    'adversary',
    'protagonist',
    'none',
    ]


    with pd.ExcelWriter(os.path.join(data_dir, 'relations_empty.xlsx')) as writer:
        for book in books:
            sheet_id = 'book_%s' % book
            rdf.loc[(book)].reindex(frequent).to_excel(writer, sheet_id)
            
            # worksheet = writer.book._sheets[-1]
            # for n, rel in enumerate(_rels):
            #     n = n+1
            #     worksheet['M%s' % n] = n
            #     worksheet['N%s' % n] = rel

#%%
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ling", action= 'store_true', default=False)
    parser.add_argument("--chunks", action= 'store_true', default=False)
    parser.add_argument("--simple", action= 'store_true', default=False)
    parser.add_argument('--pairs_json', action='store_true', default=False)
    parser.add_argument('--split', default='p')
    parser.add_argument('--data_dir')
    args = parser.parse_args()

    data_dir = os.path.join(args.data_dir)
    books_dir = os.path.join(data_dir, 'books')
    characters_dir = os.path.join(data_dir, 'characters')

    df = parse_books(books_dir, 
                     do_simple=args.simple, 
                     do_ling=args.ling,
                     do_chunks=args.chunks,
                     split_on=args.split)

    BOOK_MAP =  book_map = {v : k for k,v in df.groupby('book')\
                .first().file_name.str.split('.')\
                .apply(lambda x : '%s.xlsx' % x[0])\
                .to_dict().items()}

    pairs = make_train_data(df)

    cdf = build_character_data(characters_dir)

    pairs.reset_index().to_json(os.path.join(data_dir,'PAIRS.json'))
    df.to_json(os.path.join(data_dir,'DATA.json'))
    cdf.to_json(os.path.join(data_dir,'CHARACTERS.json'))
    
    make_rel_matrix(cdf, pairs)
