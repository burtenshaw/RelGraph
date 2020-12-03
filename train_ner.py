#%%
from __future__ import unicode_literals, print_function

import plac
import random
import warnings
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import pandas as pd
import numpy as np
## training data

## Spacy example

data_dir = '/home/ben/now/potter_graph/data/'
df = pd.read_pickle(data_dir + 'chunk_relations_3_12_2020.bin')

#%%
def make_spacy_ents(df):
    """
    convert to spacy format entity tuples, remove duplicates, and non entity text

    remove overlapping entities

    TRAIN_DATA = [
        ("Who is Shaka Khan?", {"entities": [(7, 17, "PERSON")]}),
        ("I like London and Berlin.", {"entities": [(7, 13, "LOC"), (18, 24, "LOC")]}),
    ]
    
    """
    
    df = df.drop_duplicates(subset=['paragraph'])[['paragraph', 'entities']]
    df['entities'] = df.entities.dropna()\
                .apply(lambda x : {'entities' : [(start, end, ent) for n, (start, end, ent) \
                    in enumerate(x) if start > x[n-1][1]]})

    return df.dropna().apply(lambda x : (x.paragraph, x.entities), axis = 1).to_list()


TRAIN_DATA = make_spacy_ents(df)
#%%

results = []

@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, output_dir=None, n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    # only train NER
    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print(itn)
            print("Losses", losses)

    # test the trained model
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        e = [(ent.text, ent.label_) for ent in doc.ents]
        t = [(t.text, t.ent_type_, t.ent_iob) for t in doc]
        results.append({'text' : t, 'entities': e})


    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)



if __name__ == "__main__":
    plac.call(main)
# %%
