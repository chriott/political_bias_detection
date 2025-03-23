import os
import json
import time
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
import itertools

seed = 100
torch.manual_seed(seed)
np.random.seed(seed)

# 1. PREPROCESSING ------------------------------------------------------------------

# Choose which split type you want to use: 'random' or 'media'
split_type = 'media'

json_dir = 'data/jsons/'
splits_base_dir = f'data/splits/{split_type}/'

train_split_path = os.path.join(splits_base_dir, 'train.tsv')
val_split_path = os.path.join(splits_base_dir, 'valid.tsv')
test_split_path = os.path.join(splits_base_dir, 'test.tsv')

# 2. LOAD SPLIT FILES ----------------------------------------------------------------

def load_split_file(filepath):
    return pd.read_csv(filepath, sep='\t')

print("Loading split files...")
train_df = load_split_file(train_split_path)
val_df = load_split_file(val_split_path)
test_df = load_split_file(test_split_path)

print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")

# 3. PARSE JSON CONTENTS --------------------------------------------------------------

def load_article_text(article_id, json_folder=json_dir):
    json_path = os.path.join(json_folder, f"{article_id}.json")
    
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return ""
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('content', '')

def load_dataset(df):
    texts = []
    labels = []
    
    total = len(df)
    print(f"Loading {total} samples...")

    start_time = time.time()

    for idx, row in df.iterrows():
        article_id = row['ID']
        label = row['bias']
        
        text = load_article_text(article_id)
        
        if not text or text.strip() == "":
            continue
        
        texts.append(text)
        labels.append(label)

    print("Finished loading data.")
    return texts, labels


print("Loading datasets (this might take a while)...")
train_texts, train_labels = load_dataset(train_df)
val_texts, val_labels = load_dataset(val_df)
test_texts, test_labels = load_dataset(test_df)

print(f"Loaded {len(train_texts)} train, {len(val_texts)} validation, {len(test_texts)} test samples.")

# 4. TOKENIZE TEXT -------------------------------------------------------------------

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print("Tokenizing data...")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

# 5. CREATE PYTORCH DATASETS --------------------------------------------------------

class BiasDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = BiasDataset(train_encodings, train_labels)
val_dataset = BiasDataset(val_encodings, val_labels)
test_dataset = BiasDataset(test_encodings, test_labels)

# 7. DEFINE TRAINING ARGUMENTS ------------------------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    mae = mean_absolute_error(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'mae': mae
    }

# 8. GRID SEARCH OVER HYPERPARAMETERS + EARLY STOPPING -------------------------------

best_f1 = 0
best_params = None

search_space = {
    "learning_rate": [1e-5, 5e-5],
    "num_train_epochs": [5, 8],
    "per_device_train_batch_size": [8,16],
    "weight_decay": [0.0, 0.01]
}

param_combinations = list(itertools.product(*search_space.values()))

for param_values in param_combinations:
    current_params = dict(zip(search_space.keys(), param_values))
    print(f"Trying hyperparameter combination: {current_params}")

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=current_params["learning_rate"],
        per_device_train_batch_size=current_params["per_device_train_batch_size"],
        num_train_epochs=current_params["num_train_epochs"],
        weight_decay=current_params["weight_decay"],
        logging_dir='./logs',
        logging_strategy="epoch",
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    eval_metrics = trainer.evaluate(val_dataset)
    if eval_metrics['eval_f1'] > best_f1:
        best_f1 = eval_metrics['eval_f1']
        best_params = current_params

print(f"Best params found: {best_params} with F1: {best_f1}")

# 9. FINAL TRAINING WITH BEST HYPERPARAMETERS ---------------------------------------

print(f"\nTraining with best hyperparameters and seed {seed}...")

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=best_params["learning_rate"],
    per_device_train_batch_size=best_params["per_device_train_batch_size"],
    num_train_epochs=best_params["num_train_epochs"],
    weight_decay=best_params["weight_decay"],
    logging_dir='./logs',
    logging_strategy="epoch",
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()
test_metrics = trainer.evaluate(test_dataset)

# Get predictions for confusion matrix and accuracy
predictions_output = trainer.predict(test_dataset)
predictions = np.argmax(predictions_output.predictions, axis=-1)
true_labels = predictions_output.label_ids

cm = confusion_matrix(true_labels, predictions)
acc = accuracy_score(true_labels, predictions)

print(f"\nConfusion Matrix (seed {seed}):\n{cm}")
print(f"Accuracy (seed {seed}): {acc:.4f}")

mae = test_metrics.get('eval_mae', None)
if mae is not None:
    print(f"Test set MAE (seed {seed}): {mae:.4f}")

final_macro_f1 = test_metrics['eval_f1']
final_mae = mae
final_acc = acc

print("\nFinal Macro F1 score:", final_macro_f1)
print("\nFinal MAE score:", final_mae)
print("\nFinal Accuracy score:", final_acc)

# 10. SAVE FINAL RESULTS -------------------------------------------------------------

results_dir = './results/bert_base_uncased'
os.makedirs(results_dir, exist_ok=True)

final_results = {
    'seed': seed,
    'macro_f1_score': final_macro_f1,
    'mae_score': final_mae,
    'accuracy_score': final_acc
}

filename = f'bert_base_eval_results_{split_type}.json'
with open(os.path.join(results_dir, filename), 'w') as f:
    json.dump(final_results, f, indent=4)

print(f"\nSaved final evaluation metrics to {os.path.join(results_dir, filename)}")