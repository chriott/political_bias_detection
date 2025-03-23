import os
import json
import time
import itertools
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error
from sklearn.metrics import accuracy_score, f1_score

seed = 333
torch.manual_seed(seed)
np.random.seed(seed)

# 1. CONFIGURATION ------------------------------------------------------------------
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

# 3. LOAD JSON CONTENTS --------------------------------------------------------------
def load_article_text(article_id, json_folder=json_dir):
    json_path = os.path.join(json_folder, f"{article_id}.json")
    
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return ""
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Choose 'content' or 'content_original'
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

    print("Finished loading dataset")
    return texts, labels

print("Loading dataset...")
train_texts, train_labels = load_dataset(train_df)
val_texts, val_labels = load_dataset(val_df)
test_texts, test_labels = load_dataset(test_df)

print(f"Loaded {len(train_texts)} train, {len(val_texts)} validation, {len(test_texts)} test samples.")

# 4. TOKENIZE TEXT -------------------------------------------------------------------
model_id = "answerdotai/ModernBERT-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Tokenizing data...")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=4096)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=4096)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=4096)

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

# 6. DEFINE HYPERPARAMETER SEARCH SPACE ----------------------------------------------
search_space = {
    "learning_rate": [5e-5, 1e-5],
    "num_train_epochs": [4, 8],
    "per_device_train_batch_size": [4,8]
}

keys, values = zip(*search_space.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

best_f1 = 0
best_params = None

print(f"Running grid search over {len(combinations)} combinations...")

# 7. GRID SEARCH OVER HYPERPARAMETERS + EARLY STOPPING -------------------------------
for idx, params in enumerate(combinations):
    print(f"\nRunning combination {idx + 1}/{len(combinations)}: {params}")
    
    # Reinitialize the model here
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=3, torch_dtype=torch.bfloat16
    )
    model.to('cuda')
    
    # Update training arguments with new hyperparameters
    trial_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        learning_rate=params["learning_rate"],
        per_device_train_batch_size=params["per_device_train_batch_size"],
        per_device_eval_batch_size=4,
        num_train_epochs=params["num_train_epochs"],
        weight_decay=0.01,
        logging_dir='./logs',
        logging_strategy="epoch",
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        bf16=True,
        fp16=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Initialize a new trainer with updated args
    trainer = Trainer(
        model=model,
        args=trial_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    trainer.train()
    eval_metrics = trainer.evaluate()
    
    print(f"Validation F1: {eval_metrics['eval_f1']}")
    
    if eval_metrics['eval_f1'] > best_f1:
        best_f1 = eval_metrics['eval_f1']
        best_params = params

print("\nGrid search complete.")
print(f"Best F1: {best_f1}")
print(f"Best hyperparameters: {best_params}")

# 8. FINAL TRAINING AND EVALUATION ---------------------------------------------------
print(f"\nStarting final training with best hyperparameters...")

final_model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=3, torch_dtype=torch.bfloat16)
final_model.to('cuda')

final_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="no",
    load_best_model_at_end=False,
    learning_rate=best_params["learning_rate"],
    per_device_train_batch_size=best_params["per_device_train_batch_size"],
    per_device_eval_batch_size=4,
    num_train_epochs=best_params["num_train_epochs"],
    weight_decay=0.01,
    logging_dir='./logs',
    logging_strategy="epoch",
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    bf16=True,
    fp16=False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

final_trainer = Trainer(
    model=final_model,
    args=final_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

final_trainer.train()

print("Evaluating on test set...")
test_eval_metrics = final_trainer.evaluate(test_dataset)
print(f"Test set metrics: {test_eval_metrics}")

mae = test_eval_metrics.get('eval_mae', None)
if mae is not None:
    print(f"Test set MAE: {mae:.4f}")

final_macro_f1 = test_eval_metrics['eval_f1']
final_mae = mae
final_acc = test_eval_metrics['eval_accuracy']

print("\nFinal Macro F1 score:", final_macro_f1)
print("\nFinal MAE score:", final_mae)
print("\nFinal Accuracy score:", final_acc)

# 9. SAVE FINAL RESULTS --------------------------------------------------------------
results_dir = './results/mbert_large_tune'
os.makedirs(results_dir, exist_ok=True)

final_results = {
    'seed': seed,
    'macro_f1_score': final_macro_f1,
    'mae_score': final_mae,
    'accuracy_score': final_acc
}

filename = f'bert_large_eval_results_{split_type}.json'
with open(os.path.join(results_dir, filename), 'w') as f:
    json.dump(final_results, f, indent=4)

print(f"\nSaved final evaluation metrics to {os.path.join(results_dir, filename)}")