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
data_dir = os.path.join('data')
df = pd.read_json(os.path.join(data_dir,'PAIRS.json'))
cdf = pd.read_json(os.path.join(data_dir,'CHARACTERS_1_2_21.json'))

#%% Add relation info
# cdf = cdf.set_index(['book', 'id'])

# df = df.merge(cdf[['relation to protagonist']]\
#         .add_prefix('source_'), 
#         left_on=['book', 'source'],
#         right_index=True)

# df = df.merge(cdf[['relation to protagonist']]\
#         .add_prefix('target_'), 
#         left_on=['book', 'target'], 
#         right_index=True)

# df = df.loc[df.source_base.map(lambda x : 'harry' in x) \
#             | \
#             df.target_base.map(lambda x : 'harry' in x) 
#             ]

# df['source_rel'] = df['source_relation to protagonist']
# df['target_rel'] = df['target_relation to protagonist']

# df = df.reset_index()

# df['y_relation'] = df.dropna(subset=['source_rel', 'target_rel'])\
#                     .apply(lambda row : (row.source_rel+row.target_rel)\
#                     .replace('protagonist', '') , axis = 1)
#%%
rel_dict = {}

for b in df.book.drop_duplicates().values:
    rels = pd.read_excel(os.path.join(data_dir, 'relations_annotated.xlsx'), 
            sheet_name='book_%s' % b, index_col = 0)
    rel_dict[b] = rels.to_dict()

def get_rel(row):

    try:
        return rel_dict[row.book][row.source_base][row.target_base]
    except KeyError:
        return np.nan

df['y_relation'] = df.apply(get_rel, axis = 1)
df.dropna(subset=['y_relation'], inplace = True)
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
train_index, test_index = train_test_split(
                                        df.index, 
                                        test_size=0.33, 
                                        random_state=42)

#%%
metric = load_metric('glue', "mnli")
model_checkpoint = 'albert-base-v2'
batch_size = 16
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

datasets = DatasetDict()

datasets['train'] = Dataset.from_pandas(df.loc[train_index][['sentence', 'label']])
datasets['test'] = Dataset.from_pandas(df.loc[test_index][['sentence', 'label']])
# %%
def preprocess_function(examples):
    return tokenizer(examples['sentence'], truncation=True)

encoded_dataset = datasets.map(preprocess_function, batched=False)

num_labels = df.label.max() + 1

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)


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
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# %%
trainer.train()
# %%
trainer.evaluate()
#%%

predictions, labels, _ = trainer.predict(encoded_dataset['test'])
predictions = np.argmax(predictions, axis=1)

#%%
label_names = { n: v for n, v in enumerate(to_keep)}

test = df.loc[test_index]
test['cluster'] = predictions

#%%

y_pred = test.groupby('source').cluster.agg(lambda x:x.value_counts().index[0]).to_list() + \
            test.groupby('target').cluster.agg(lambda x:x.value_counts().index[0]).to_list() 

y_true = test.groupby('source').label.first().to_list() + \
            test.groupby('target').label.first().to_list()

pd.DataFrame(classification_report(y_pred, y_true, output_dict=True))

#%%

anno = test.sort_values('cluster')
with pd.ExcelWriter(os.path.join(data_dir, 'clusters.xlsx')) as writer:
    for cluster_id, _df in anno.groupby('cluster'):
        _df.to_excel(writer, label_names[cluster_id])



# %%
def get_prediction(sentence):
    d = Dataset.from_pandas(pd.DataFrame([{'sentence' : sentence, 'label' : 0}]))
    d = d.map(preprocess_function, batched=False)
    return np.argmax(trainer.predict(d).predictions)
# %%
from sklearn.metrics import classification_report

pd.DataFrame(classification_report(test.cluster, test.label, output_dict=True))
# %%
