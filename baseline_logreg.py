import os
import json
import pandas as pd
import random
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, mean_absolute_error
from tqdm import tqdm

seed = 100
random.seed(seed)
np.random.seed(seed)

# 1. CONFIGURATION -------------------------------------------------------------------
split_type = 'media'

json_dir = 'data/jsons/'
splits_base_dir = f'data/splits/{split_type}/'

train_split_path = os.path.join(splits_base_dir, 'train.tsv')
val_split_path = os.path.join(splits_base_dir, 'valid.tsv')
test_split_path = os.path.join(splits_base_dir, 'test.tsv')

# 2. LOAD DATA -----------------------------------------------------------------------
def load_split_file(filepath):
    return pd.read_csv(filepath, sep='\t')

train_df = load_split_file(train_split_path)
val_df = load_split_file(val_split_path)
test_df = load_split_file(test_split_path)

def load_article_text(article_id, json_folder=json_dir):
    json_path = os.path.join(json_folder, f"{article_id}.json")
    if not os.path.exists(json_path):
        return ""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('content', '')

def load_dataset(df):
    texts = []
    labels = []
    for idx, row in df.iterrows():
        article_id = row['ID']
        label = row['bias']
        text = load_article_text(article_id)
        if not text or text.strip() == "":
            continue
        texts.append(text)
        labels.append(label)
    return texts, labels

train_texts, train_labels = load_dataset(train_df)
val_texts, val_labels = load_dataset(val_df)
test_texts, test_labels = load_dataset(test_df)

# 3. VECTORIZATION ------------------------------------------------------------------
def simple_tokenizer(text):
    return text.lower().split()

print("Vectorizing with TF-IDF...")
vectorizer = TfidfVectorizer(
    tokenizer=simple_tokenizer,
    max_features=5000,        
    ngram_range=(1, 2),
)

X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)
X_test = vectorizer.transform(test_texts)

# 4. TRAIN CLASSIFIER ----------------------------------------------------------------
print("Training Logistic Regression baseline...")

clf = LogisticRegression(max_iter=1000, solver='liblinear', multi_class='auto', random_state=seed)
clf.fit(X_train, train_labels)

# 5. EVALUATION ----------------------------------------------------------------------
def evaluate_baseline(model, X, y_true, split_name="Validation"):
    preds = model.predict(X)

    acc = accuracy_score(y_true, preds)
    f1_macro = f1_score(y_true, preds, average='macro')
    mae = mean_absolute_error(y_true, preds)
    report = classification_report(y_true, preds, digits=4)

    print(f"\n{split_name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Classification Report:\n{report}")

    return acc, f1_macro, mae, report

# Evaluate on validation set
val_acc, val_f1, val_mae, val_report = evaluate_baseline(clf, X_val, val_labels, split_name="Validation")

# Evaluate on test set
test_acc, test_f1, test_mae, test_report = evaluate_baseline(clf, X_test, test_labels, split_name="Test")

# 6. SAVE FINAL RESULTS --------------------------------------------------------------
results_dir = './results/logreg'
os.makedirs(results_dir, exist_ok=True)

baseline_results = {
    'test_accuracy': test_acc,
    'test_macro_f1': test_f1,
    'test_mae': test_mae
}

results_output_filename = f'logreg_baseline_results_{split_type}.json'
results_output_path = os.path.join(results_dir, results_output_filename)
with open(results_output_path, 'w') as f:
    json.dump(baseline_results, f, indent=4)

print(f"\nBaseline results saved to {results_output_path}")