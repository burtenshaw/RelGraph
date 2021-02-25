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
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

import torch
from transformers import AdamW
device = torch.device('cuda')

data_dir = os.path.join('data')
df = pd.read_json(os.path.join(data_dir,'PAIRS.json'))
cdf = pd.read_json(os.path.join(data_dir,'CHARACTERS_1_2_21.json'))

#%% Add relation info
rel_matrix = True
to_keep = ['friend',
        'adversary',
        'teacher',
        'helper',
        'other',
        'family',
        'none',
        'ambivalent']

if not rel_matrix:
    cdf = cdf.set_index(['book', 'id'])

    df = df.merge(cdf[['relation to protagonist']]\
            .add_prefix('source_'), 
            left_on=['book', 'source'],
            right_index=True)

    df = df.merge(cdf[['relation to protagonist']]\
            .add_prefix('target_'), 
            left_on=['book', 'target'], 
            right_index=True)

    df = df.loc[df.source_base.map(lambda x : 'harry' in x) \
                | \
                df.target_base.map(lambda x : 'harry' in x) 
                ]

    df['source_rel'] = df['source_relation to protagonist']
    df['target_rel'] = df['target_relation to protagonist']

    df = df.reset_index()

    df['y_relation'] = df.dropna(subset=['source_rel', 'target_rel'])\
                        .apply(lambda row : (row.source_rel+row.target_rel)\
                        .replace('protagonist', '') , axis = 1)

    df = df.loc[df.y_relation.map(lambda x : x in to_keep)]
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

label_map = { v: n for n, v in enumerate(df.y_relation)}
df['label'] = df.y_relation.apply(lambda x : label_map[x])
df['sentence'] = df.train_data.apply(lambda x : ''.join(x))
#%%
complete_index = df.index
# complete_index = df.index[:100]
train_index, test_index = train_test_split(
                                        complete_index, 
                                        test_size=0.2, 
                                        random_state=42)

train_index, val_index = train_test_split(
                                        train_index ,
                                        test_size=0.1, 
                                        random_state=42)

#%%
model_checkpoint = 'albert-base-v2'
num_labels = df.label.max() + 1

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

X_train_text = df.loc[train_index].sentence.to_list()
X_val_text = df.loc[val_index].sentence.to_list()
X_test_text = df.loc[test_index].sentence.to_list()


# %%
X_train = tokenizer(X_train_text, return_tensors='pt', padding=True, truncation=True)
y_train = torch.tensor(df.loc[train_index].label.values)

X_val = tokenizer(X_val_text, return_tensors='pt', padding=True, truncation=True)
y_val = torch.tensor(df.loc[val_index].label.values)

X_test = tokenizer(X_test_text, return_tensors='pt', padding=True, truncation=True)
y_test = torch.tensor(df.loc[test_index].label.values)
#%%
import torch.nn as nn
import torch.nn.functional as F

class RelationMatcher(nn.Module):
    def __init__(self, model_checkpoint, num_class, batch_size):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, 
                                                                        num_labels=num_class
                                                                        )
        self.model.train()
        self.lin1 = nn.Linear(batch_size, num_class)
        self.dropout = nn.Dropout(0.2)
        self.lin2 = nn.Linear(batch_size, num_class)
        self.sm = nn.Softmax()

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask=attention_mask)
        linear_output = self.lin1(output.logits)
        linear_output = self.dropout(linear_output)
        linear_output = self.lin2(linear_output)
        # probs = self.sm(linear_output)
        return linear_output

def run(model, X, y, train= True):

    # Train the model
    epoch_loss = 0
    acc = 0

    input_ids, attention_mask, labels = \
        X['input_ids'].cuda(), X['attention_mask'].cuda(), y.cuda()
    
    num_samples = X['input_ids'].shape[0]

    permutation = torch.randperm(num_samples)

    for i in tqdm(range(0,num_samples, batch_size)):

        indices = permutation[i:i+batch_size]

        batch_input_ids, batch_attention_mask, batch_labels = \
            input_ids[indices], attention_mask[indices], labels[indices]

        if train:
            optimizer.zero_grad()
            outputs = model.forward(batch_input_ids, batch_attention_mask)
            loss = criterion(outputs, batch_labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Adjust the learning rate
            # scheduler.step()

        else:
            with torch.no_grad():
                outputs = model(batch_input_ids, batch_attention_mask)
                loss = criterion(outputs, batch_labels)
                epoch_loss += loss.item()

        acc += (outputs.argmax(1) == batch_labels).sum().item()

    return epoch_loss / num_samples, acc / num_samples

#%%
batch_size = 8
num_epochs = 4
num_samples = X_train['input_ids'].shape[0]
seq_len = X_train['input_ids'].shape[1]

model = RelationMatcher(model_checkpoint, num_labels, batch_size).to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss().to(device)

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

for epoch in range(num_epochs):

    train_loss = 0
    train_acc = 0

    print('EPOCH %s' % (epoch + 1))

    train_loss, train_acc = run(model, X_train, y_train)
    valid_loss, valid_acc = run(model, X_val, y_val, train = False)  

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/test', valid_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/test', valid_acc, epoch)

    print('Epoch: %d' %(epoch + 1))
    print(f'\ttrain loss: {train_loss:.4f}(train)\t|\ttrain acc: {train_acc * 100:.1f}%(train)')
    print(f'\tvalid loss: {valid_loss:.4f}(valid)\t|\tvalid acc: {valid_acc * 100:.1f}%(valid)')

#%%
def predict(model, X):

    input_ids, attention_mask = \
        X['input_ids'].cuda(), X['attention_mask'].cuda()

    num_samples = X['input_ids'].shape[0]

    permutation = torch.randperm(num_samples)

    preds = []

    for i in tqdm(range(0,num_samples, batch_size)):

        indices = permutation[i:i+batch_size]

        batch_input_ids, batch_attention_mask= \
            input_ids[indices], attention_mask[indices]


        with torch.no_grad():
            outputs = model(batch_input_ids, batch_attention_mask)
            preds.append(outputs.argmax(1).cpu().numpy())

    preds = np.concatenate(preds)
    return preds

predictions = predict(model, X_test)

#%%
label_names = { n: v for n, v in enumerate(to_keep)}

test = df.loc[test_index]
test['cluster'] = predictions
anno = test.sort_values('cluster')
#%%
pd.DataFrame(classification_report(test.cluster, test.label, output_dict=True))
#%%

y_pred = test.groupby('source').cluster.agg(lambda x:x.value_counts().index[0]).to_list() + \
            test.groupby('target').cluster.agg(lambda x:x.value_counts().index[0]).to_list() 

y_true = test.groupby('source').label.first().to_list() + \
            test.groupby('target').label.first().to_list()

pd.DataFrame(classification_report(y_pred, y_true, output_dict=True))


# %%
with pd.ExcelWriter(os.path.join(data_dir, 'clusters.xlsx')) as writer:
    for cluster_id, _df in anno.groupby('cluster'):
        _df.to_excel(writer, label_names[cluster_id])

def get_prediction(sentence):
    d = Dataset.from_pandas(pd.DataFrame([{'sentence' : sentence, 'label' : 0}]))
    d = d.map(preprocess_function, batched=False)
    return np.argmax(trainer.predict(d).predictions)
# %%



# %%

''' predict user relation '''

test.groupby('source').cluster.value_counts()