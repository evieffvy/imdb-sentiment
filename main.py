import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'aclImdb')

# Reduced for CPU training
NUM_STEPS   = 200   # sequence length (original: 500)
BATCH_SIZE  = 64
EPOCHS      = 5     # (original: 10)
EMBED_SIZE  = 100
NUM_HIDDENS = 100
NUM_LAYERS  = 1
LR          = 0.001


# ── 1. Data Loading ───────────────────────────────────────────────────
def read_imdb(data_dir, is_train):
    """Load IMDB reviews and binary labels (1=positive, 0=negative)"""
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder = os.path.join(data_dir, 'train' if is_train else 'test', label)
        if not os.path.exists(folder):
            print(f"Error: Path not found {folder}")
            return [], []
        for file in os.listdir(folder):
            with open(os.path.join(folder, file), 'rb') as f:
                data.append(f.read().decode('utf-8').replace('\n', ''))
                labels.append(1 if label == 'pos' else 0)
    return data, labels


print("Loading data...")
train_data, train_labels_list = read_imdb(DATA_DIR, is_train=True)
test_data,  test_labels_list  = read_imdb(DATA_DIR, is_train=False)

if not train_data:
    print("Dataset not found. Please place aclImdb under ./data/")
    exit()

print(f"Train: {len(train_data)} samples | Test: {len(test_data)} samples")


# ── 2. Tokenization & Vocabulary ─────────────────────────────────────
tokenizer = get_tokenizer("basic_english")


def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)


print("Building vocabulary (min_freq=5)...")
vocab = build_vocab_from_iterator(
    yield_tokens(train_data),
    min_freq=5,
    specials=["<unk>", "<pad>"]
)
vocab.set_default_index(vocab["<unk>"])
print(f"Vocabulary size: {len(vocab)}")


# ── 3. Padding & Truncating ───────────────────────────────────────────
def process_text(text_list, num_steps=NUM_STEPS):
    """Tokenize, convert to indices, then pad or truncate to fixed length"""
    processed = []
    for text in text_list:
        indices = vocab(tokenizer(text))
        if len(indices) > num_steps:
            indices = indices[:num_steps]
        else:
            indices = indices + [vocab["<pad>"]] * (num_steps - len(indices))
        processed.append(indices)
    return torch.tensor(processed, dtype=torch.int64)


print(f"Processing sequences (length={NUM_STEPS})...")
train_features = process_text(train_data)
test_features  = process_text(test_data)

train_labels_t = torch.tensor(train_labels_list, dtype=torch.float32)
test_labels_t  = torch.tensor(test_labels_list,  dtype=torch.float32)

train_iter = DataLoader(TensorDataset(train_features, train_labels_t), batch_size=BATCH_SIZE, shuffle=True)
test_iter  = DataLoader(TensorDataset(test_features,  test_labels_t),  batch_size=BATCH_SIZE, shuffle=False)


# ── 4. BiRNN Model ────────────────────────────────────────────────────
class BiRNN(nn.Module):
    """Bidirectional LSTM for binary sentiment classification"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # bidirectional=True doubles the hidden size
        self.encoder = nn.LSTM(embed_size, num_hiddens,
                               num_layers=num_layers, bidirectional=True)
        # concatenate first and last hidden states from both directions: 4 * num_hiddens
        self.decoder = nn.Linear(4 * num_hiddens, 1)

    def forward(self, inputs):
        inputs = inputs.permute(1, 0)           # (Batch, Seq) -> (Seq, Batch)
        embeddings = self.embedding(inputs)
        outputs, _ = self.encoder(embeddings)
        # Use first and last timestep to capture full context
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        return self.decoder(encoding)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

model     = BiRNN(len(vocab), EMBED_SIZE, NUM_HIDDENS, NUM_LAYERS).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# ── 5. Training Loop ──────────────────────────────────────────────────
def evaluate(model, data_iter, device):
    """Return accuracy, and all predictions + true labels for the dataset"""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for text_batch, label_batch in data_iter:
            text_batch = text_batch.to(device)
            preds = (torch.sigmoid(model(text_batch).squeeze(1)) > 0.5).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(label_batch.tolist())
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return acc, all_preds, all_labels


train_losses, train_accs, test_accs = [], [], []

print(f"Training BiRNN ({EPOCHS} epochs)...")
for epoch in range(EPOCHS):
    t0 = time.time()
    model.train()
    total_loss, correct, total = 0, 0, 0

    for text_batch, label_batch in train_iter:
        text_batch, label_batch = text_batch.to(device), label_batch.to(device)
        optimizer.zero_grad()
        predictions = model(text_batch).squeeze(1)
        loss = criterion(predictions, label_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += ((torch.sigmoid(predictions) > 0.5) == label_batch).sum().item()
        total   += label_batch.size(0)

    epoch_loss = total_loss / len(train_iter)
    epoch_acc  = correct / total
    test_acc, _, _ = evaluate(model, test_iter, device)

    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    test_accs.append(test_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Loss: {epoch_loss:.4f} | "
          f"Train Acc: {epoch_acc:.4f} | "
          f"Test Acc: {test_acc:.4f} | "
          f"Time: {time.time()-t0:.1f}s")


# ── 6. Final Evaluation (BiRNN) ───────────────────────────────────────
_, birnn_preds, true_labels = evaluate(model, test_iter, device)

birnn_acc  = accuracy_score(true_labels,  birnn_preds)
birnn_pre  = precision_score(true_labels, birnn_preds)
birnn_rec  = recall_score(true_labels,    birnn_preds)
birnn_f1   = f1_score(true_labels,        birnn_preds)
birnn_cm   = confusion_matrix(true_labels, birnn_preds)

print("\n========== BiRNN (from scratch) ==========")
print(f"  Accuracy  : {birnn_acc:.4f}")
print(f"  Precision : {birnn_pre:.4f}")
print(f"  Recall    : {birnn_rec:.4f}")
print(f"  F1 Score  : {birnn_f1:.4f}")


# ── 7. Baseline — Logistic Regression + TF-IDF ───────────────────────
print("\nTraining baseline (Logistic Regression + TF-IDF)...")
tfidf = TfidfVectorizer(max_features=10000)
X_train_tfidf = tfidf.fit_transform(train_data)
X_test_tfidf  = tfidf.transform(test_data)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, train_labels_list)
lr_preds = lr_model.predict(X_test_tfidf)

lr_acc = accuracy_score(test_labels_list,  lr_preds)
lr_pre = precision_score(test_labels_list, lr_preds)
lr_rec = recall_score(test_labels_list,    lr_preds)
lr_f1  = f1_score(test_labels_list,        lr_preds)
lr_cm  = confusion_matrix(test_labels_list, lr_preds)

print("\n========== Baseline (Logistic Regression + TF-IDF) ==========")
print(f"  Accuracy  : {lr_acc:.4f}")
print(f"  Precision : {lr_pre:.4f}")
print(f"  Recall    : {lr_rec:.4f}")
print(f"  F1 Score  : {lr_f1:.4f}")


# ── 8. Plots ──────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, title, ax):
    """Plot a labeled confusion matrix"""
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticklabels(['Negative', 'Positive'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(title, fontweight='bold')
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black', fontsize=13)


# Training curves
_, axes = plt.subplots(1, 3, figsize=(15, 4))
epochs_x = range(1, EPOCHS + 1)

axes[0].plot(epochs_x, train_losses, 'o-', color='steelblue')
axes[0].set_title('Training Loss', fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')

axes[1].plot(epochs_x, train_accs, 'o-', color='steelblue', label='Train')
axes[1].plot(epochs_x, test_accs,  'o-', color='coral',     label='Test')
axes[1].set_title('Accuracy per Epoch', fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].legend()

# Metrics comparison bar chart
metrics   = ['Accuracy', 'Precision', 'Recall', 'F1']
birnn_scores = [birnn_acc, birnn_pre, birnn_rec, birnn_f1]
lr_scores    = [lr_acc,    lr_pre,    lr_rec,    lr_f1]
x = np.arange(len(metrics)); w = 0.35
axes[2].bar(x - w/2, birnn_scores, w, label='BiRNN',        color='steelblue')
axes[2].bar(x + w/2, lr_scores,    w, label='LR + TF-IDF',  color='coral')
axes[2].set_xticks(x); axes[2].set_xticklabels(metrics)
axes[2].set_ylim(0, 1.1); axes[2].set_ylabel('Score')
axes[2].set_title('BiRNN vs Baseline', fontweight='bold')
axes[2].legend()
for bar in axes[2].patches:
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{bar.get_height():.2f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'training_result.png'), dpi=150)
plt.close()

# Confusion matrices
_, axes = plt.subplots(1, 2, figsize=(10, 4))
plot_confusion_matrix(birnn_cm, 'BiRNN',               axes[0])
plot_confusion_matrix(lr_cm,    'Logistic Regression', axes[1])
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'confusion_matrices.png'), dpi=150)
plt.close()

print("\nGraphs saved: training_result.png, confusion_matrices.png")
