#%%
import numpy as np
import pandas as pd
import random
import os
from ast import literal_eval
import argparse

from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast
from transformers import TrainingArguments 
from transformers import Trainer 
from transformers import DataCollatorForTokenClassification 
from transformers import ElectraTokenizerFast 
from transformers import ElectraForTokenClassification

from datasets import load_dataset
from datasets import load_metric 
from datasets import ClassLabel 
from datasets import Sequence
from datasets import Dataset 
from datasets import DatasetDict

dev_path = 'chunk_relations_26_11_2020.bin'
dev_output_path = 'potter_NER.json'
# parser = argparse.ArgumentParser(description='Add Name Entity Recognition to dataset using FineTuned Transformer model')

# parser.add_argument('--data', 
#                     metavar='--d', 
#                     type=str, 
#                     help='path to preprocessed data',
#                     defaulbt=dev_path)

# parser.add_argument('--output', 
#                     metavar='--d', 
#                     type=str, 
#                     help='path to output data with NER predictions',
#                     default=dev_output_path)

# args = parser.parse_args()

SOURCE_PATH = dev_path
OUTPUT_PATH = dev_output_path

potter = pd.read_pickle(SOURCE_PATH)
task = "ner" # Should be one of "ner", "pos" or "chunk"
model_checkpoint = 'google/electra-small-discriminator'
batch_size = 16
datasets = load_dataset("conll2003")
label_list = datasets["train"].features[f"{task}_tags"].feature.names

tokenizer = ElectraTokenizerFast.from_pretrained(model_checkpoint)
assert isinstance(tokenizer, PreTrainedTokenizerFast)

label_all_tokens = True
#%%

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p):

    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# def tokenize_eval(example):

#     tokenized_inputs = tokenizer(example["paragraph"], 
#                                  truncation=True, 
#                                  is_split_into_words=False)

#     tokenized_inputs["word_ids"] = tokenized_inputs.word_ids()    

#     return tokenized_inputs


# %%
potter['tokens'] = potter.sentences
potter[f"{task}_tags"] = potter.tokens.apply(lambda x : [0] * len(x))
datasets['eval'] = Dataset.from_pandas(potter[['tokens', f"{task}_tags"]])
# tokenized_datasets['eval'] = datasets['eval'].map(tokenize_eval, batched=False)
tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer)
metric = load_metric("seqeval")
#%% Train the model
model = ElectraForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

args = TrainingArguments(
    f"test-{task}",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=4,
    weight_decay=0.01,
)

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

#%% Evaluate the model
trainer.evaluate()
# %%
_predictions, labels, _ = trainer.predict(tokenized_datasets['eval'])
predictions = np.argmax(_predictions, axis=2)
# %%  Remove ignored special tokens
potter['ner'] = [
    [label_list[p] for p in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

# %%
potter.to_json(OUTPUT_PATH)