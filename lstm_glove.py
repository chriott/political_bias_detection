import os
import json
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_absolute_error
from itertools import product
from tqdm import tqdm
import random

seed = 100
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 1. DEVICE SETUP -------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. CONFIGURATION ------------------------------------------------------------------
# Choose which split type you want to use: 'random' or 'media'
split_type = 'media'

json_dir = 'data/jsons/'
splits_base_dir = f'data/splits/{split_type}/'

train_split_path = os.path.join(splits_base_dir, 'train.tsv')
val_split_path = os.path.join(splits_base_dir, 'valid.tsv')
test_split_path = os.path.join(splits_base_dir, 'test.tsv')

# 3. LOAD DATA ----------------------------------------------------------------------
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

# 4. TOKENIZATION & VOCAB BUILDING --------------------------------------------------
def simple_tokenizer(text):
    return text.lower().split()

# Build vocab from train texts
def build_vocab(texts, tokenizer, min_freq=5):
    counter = Counter()
    for text in texts:
        tokens = tokenizer(text)
        counter.update(tokens)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def load_glove_embeddings(glove_file_path, vocab, embed_dim=300):
    embeddings_index = {}
    print(f"Loading GloVe embeddings from {glove_file_path}...")
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector

    # Initialize embedding
    embedding_matrix = np.random.uniform(-0.05, 0.05, (len(vocab), embed_dim)).astype('float32')

    # Populate embedding matrix with --> GloVe vectors
    for word, idx in vocab.items():
        if word in embeddings_index:
            embedding_matrix[idx] = embeddings_index[word]

    return torch.tensor(embedding_matrix)

vocab = build_vocab(train_texts, simple_tokenizer)
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")

glove_path = 'glove.6B.300d.txt'
embed_dim = 300  
embedding_matrix = load_glove_embeddings(glove_path, vocab, embed_dim)

num_classes = len(set(train_labels))

def encode_text(text, vocab, tokenizer, max_len=512):
    tokens = tokenizer(text)
    ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    # Pad or truncate
    if len(ids) < max_len:
        ids += [vocab['<PAD>']] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

# 5. DATASET & DATALOADER -----------------------------------------------------------
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        input_ids = encode_text(text, self.vocab, self.tokenizer, self.max_len)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

batch_size = 32

train_dataset = TextDataset(train_texts, train_labels, vocab, simple_tokenizer)
val_dataset = TextDataset(val_texts, val_labels, vocab, simple_tokenizer)
test_dataset = TextDataset(test_texts, test_labels, vocab, simple_tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 6. LSTM MODEL ---------------------------------------------------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=1, dropout=0.5, pretrained_embeddings=None, freeze_embeddings=False):
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(pretrained_embeddings)
            self.embedding.weight.requires_grad = not freeze_embeddings

        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_out, (hidden, _) = self.lstm(embedded)
        out = self.fc(self.dropout(hidden[-1]))
        return out

def compute_loss_on_validation(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            outputs = model(input_ids)
            predictions = torch.argmax(outputs, dim=1)

            preds.extend(predictions.cpu().tolist())
            trues.extend(labels.cpu().tolist())

    acc = accuracy_score(trues, preds)
    f1_macro = f1_score(trues, preds, average='macro')
    mae = mean_absolute_error(trues, preds)
    report = classification_report(trues, preds, digits=4)
    return acc, f1_macro, mae, report

# 7. HYPERPARAMETER TUNING ------------------------------------------------------------------

def grid_search(train_loader, val_loader, vocab_size, num_classes):
    hidden_dims = [64, 128, 256]
    learning_rates = [1e-3, 1e-4]
    num_layers_list = [1, 2]
    num_epochs_list = [3, 5, 8]
    best_val_f1 = 0
    best_params = {}

    for hidden_dim, lr, num_layers, num_epochs in product(hidden_dims, learning_rates, num_layers_list, num_epochs_list):
        print(f"\nTesting config: hidden_dim={hidden_dim}, lr={lr}, num_layers={num_layers}, num_epochs={num_epochs}")

        model = LSTMClassifier(vocab_size, embed_dim, hidden_dim, num_classes, num_layers=num_layers, dropout=0.5, pretrained_embeddings=embedding_matrix, freeze_embeddings=False).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        patience = 2
        best_inner_val_f1 = 0
        epochs_no_improve = 0
        early_stop = False

        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion)
            val_acc, val_f1_macro, val_mae, val_report = evaluate(model, val_loader)

            if val_f1_macro > best_inner_val_f1:
                best_inner_val_f1 = val_f1_macro
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1} in grid search")
                early_stop = True
                break

        if best_inner_val_f1 > best_val_f1:
            best_val_f1 = best_inner_val_f1
            best_params = {
                'hidden_dim': hidden_dim,
                'learning_rate': lr,
                'num_layers': num_layers,
                'num_epochs': num_epochs
            }

        print(f"Validation F1 Macro: {val_f1_macro:.4f}")

    print("\nBest Hyperparameters found:")
    print(best_params)
    return best_params


best_params = grid_search(train_loader, val_loader, vocab_size, num_classes)
hidden_dim = best_params['hidden_dim']
learning_rate = best_params['learning_rate']
num_layers = best_params['num_layers']
num_epochs = best_params['num_epochs']

model = LSTMClassifier(vocab_size, embed_dim, hidden_dim, num_classes, num_layers=num_layers, dropout=0.5, pretrained_embeddings=embedding_matrix, freeze_embeddings=False).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 8. FINAL TRAINING -----------------------------------------------------------------
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_acc, val_f1_macro, val_mae, val_report = evaluate(model, val_loader)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation F1 Macro: {val_f1_macro:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation Report:\n{val_report}")

# 9. EVALUATE TEST SET --------------------------------------------------------------
test_acc, test_f1_macro, test_mae, test_report = evaluate(model, test_loader)
print(f"Test F1 Macro: {test_f1_macro:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Classification Report:\n{test_report}")

# 10. SAVE FINAL RESULTS -------------------------------------------------------------
results_dir = './results/lstmmin5'
os.makedirs(results_dir, exist_ok=True)

results = {
    'test_accuracy': test_acc,
    'test_macro_f1': test_f1_macro,
    'test_mae': test_mae
}

results_output_filename = f'lstm_experiment_{split_type}.json'
results_output_path = os.path.join(results_dir, results_output_filename)
with open(results_output_path, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {results_output_path}")