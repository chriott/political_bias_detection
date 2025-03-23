import os
import json
import pandas as pd
import random
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error
)

# 1. CONFIGURATION ------------------------------------------------------------------

# Choose split type
split_type = 'media'  # or 'random'

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

train_labels = train_df['bias'].tolist()
val_labels = val_df['bias'].tolist()
test_labels = test_df['bias'].tolist()

# 3. EVALUATION METRICS -------------------------------------------------------------

def evaluate_baseline(true_labels, pred_labels):
    acc = accuracy_score(true_labels, pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
    mae = mean_absolute_error(true_labels, pred_labels)
    
    print(f"Accuracy: {acc}")
    print(f"F1 macro: {f1_macro}")
    print(f"MAE: {mae}")

# 4. MAJORITY CLASS BASELINE --------------------------------------------------------

def majority_class_baseline(true_labels):
    label_counts = Counter(true_labels)
    most_common_label, count = label_counts.most_common(1)[0]
    
    pred_labels = [most_common_label] * len(true_labels)
    
    evaluate_baseline(true_labels, pred_labels)
    
    return most_common_label

# 5. RANDOM BASELINE -----------------------------------------------------------------

def random_baseline(true_labels):
    unique_classes = list(set(true_labels))
    total = len(true_labels)
    
    pred_labels = [random.choice(unique_classes) for _ in range(total)]
    acc = accuracy_score(true_labels, pred_labels)
    
    print(f"\nRandom baseline accuracy: {acc}")
    evaluate_baseline(true_labels, pred_labels)

# 6. RUN BASELINES -------------------------------------------------------------------

# Majority Class Baseline
print("\nEvaluating majority class baseline:")
majority_class_baseline(test_labels)

# Random Baseline
print("\nEvaluating random baseline:")
random_baseline(test_labels)
