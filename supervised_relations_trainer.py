# %%
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import sys
import os
import torch
from tqdm.notebook import tqdm

from transformers import BertTokenizer
from torch.utils.data import TensorDataset

from transformers import BertForSequenceClassification
from tqdm import *
from tensorflow.keras.utils import to_categorical

import transformers
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import ElectraForTokenClassification, ElectraTokenizerFast
from datasets import Dataset, DatasetDict
from datasets import load_metric

from lit_world import annotate

df = pd.DataFrame()

series_list = ['potter', 'pullman']

for series in series_list:
    data_dir = os.path.join('data', series)
    _df = pd.read_json(os.path.join(data_dir,'PAIRS.json'))

    rel_dict = {}

    for b in _df.book.drop_duplicates().values:
        rels = pd.read_excel(os.path.join(data_dir, 'relations_annotated.xlsx'), 
                sheet_name='book_%s' % b, index_col = 0)
        rel_dict[b] = rels.to_dict()

    _df['y_relation'] = _df.apply(annotate.get_rel, rel_dict = rel_dict, axis = 1)
    _df.dropna(subset=['y_relation'], inplace = True)

    _df['series'] = series

    df = pd.concat([df,_df])
#%% most active
to_keep = ['friend',
           'adversary',
           'teacher',
           'helper',
           'other',
           'family',
           'none',
           'ambivalent']

label_map = { v: n for n, v in enumerate(to_keep)}

df = df.loc[df.y_relation.map(lambda x : x in to_keep)]

df['label'] = df.y_relation.apply(lambda x : label_map[x])

df['sentence'] = df.train_data.apply(lambda x : ''.join(x))
#%%


#%%
metric = load_metric('glue', "mnli")
model_checkpoint = 'albert-base-v2'
batch_size = 16
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
num_labels = df.label.max() + 1

# %%
def preprocess_function(examples):
    return tokenizer(examples['sentence'], truncation=True, padding=True)

# %%
# metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
metric_name = 'accuracy'
args = TrainingArguments(
    "test-glue",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=4,
    weight_decay=0.03,
    metric_for_best_model=metric_name,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)
# %%
experiments = [
    {'train' : ['potter','pullman'],'test' : ['potter','pullman']},
    {'train' : ['potter','pullman'],'test' : ['pullman']},
    {'train' : ['potter','pullman'],'test' : ['potter']},
    {'train' : ['potter'],'test' : ['pullman']},
    {'train' : ['pullman'],'test' : ['pullman']},
    # {'test' : 'train' :},
]

for exp in experiments:

    run_df = df.loc[df.series.isin(exp['train'] + exp['test'])]

    train_index, test_index = train_test_split(
                                        run_df.index, 
                                        test_size=0.33, 
                                        random_state=42)

    
    s_test = df.series.isin(exp['test']).index

    test_index = s_test.intersection(test_index)

    print ('Training on %s . %s samples' % (exp['train'], train_index.shape[0]) )
    print ('Testing on %s . %s samples' % (exp['test'], test_index.shape[0]) )

    file_name = 'train_' + '_'.join(exp['train']) + \
                '-test_'  + '_'.join(exp['test'])

    print(file_name)    

    datasets = DatasetDict()
    datasets['train'] = Dataset.from_pandas(run_df.loc[train_index][['sentence', 'label']])
    datasets['test'] = Dataset.from_pandas(run_df.loc[test_index][['sentence', 'label']])
    encoded_dataset = datasets.map(preprocess_function, batched=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset['test'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    trainer.evaluate()

    predictions, labels, _ = trainer.predict(encoded_dataset['test'])
    predictions = np.argmax(predictions, axis=1)

    test = run_df.loc[test_index]
    test['y_pred'] = predictions
    test['train'] = '_'.join(exp['train'])
    test['test'] = '_'.join(exp['test'])

    test.reset_index().to_json('data/predictions/%s.json' % file_name)



#%%

# anno = test.sort_values('cluster')
# with pd.ExcelWriter(os.path.join(data_dir, 'clusters.xlsx')) as writer:
#     for cluster_id, _df in anno.groupby('cluster'):
#         _df.to_excel(writer, label_names[cluster_id])

# # %%
# def get_prediction(sentence):
#     d = Dataset.from_pandas(pd.DataFrame([{'sentence' : sentence, 'label' : 0}]))
#     d = d.map(preprocess_function, batched=False)
#     return np.argmax(trainer.predict(d).predictions)
# # %%


# pd.DataFrame(classification_report(test.cluster, test.label, output_dict=True))
# # %%
# pullman_dir = os.path.join('data/pullman')
# pudf = pd.read_json(os.path.join(pullman_dir,'PAIRS.json'))
# pudf['label'] = 0
# pudf['sentence'] = pudf.train_data.apply(lambda x : ''.join(x))
# pudf = pudf.dropna(subset=['sentence', 'label'])
# datasets['eval'] = Dataset.from_pandas(pudf[['sentence', 'label']])
# encoded_eval = datasets['eval'].map(preprocess_function, batched=False)
# e_predictions, e_labels, _ = trainer.predict(encoded_eval)
# pudf['pred_y'] = np.argmax(e_predictions, -1)

# #%%
# pudf.to_json('data/pullman/all_preds.json')

# %%

# %%
