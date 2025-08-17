#pip install transformers torch spacy matplotlib numpy pandas seaborn nltk umap-learn sentence-transformers scikit-learn tqdm
#python -m spacy download en_core_web_sm

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import spacy #pip install spacy and python -m spacy download en_core_web_sm
import matplotlib.pyplot as plt #pip install matplotlib
import numpy as np #pip install numpy
import pandas as pd #pip install pandas
import seaborn as sns #pip install seaborn
from nltk.stem.snowball import SnowballStemmer #pip install nltk
from umap import UMAP #pip install umap-learn
from sentence_transformers import SentenceTransformer #pip install sentence-transformers
from sklearn.model_selection import train_test_split #pip install sklearn
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
 
# --- Initial Setup and Data Loading (from your original code) ---

# --- Hyperparameters for Training ---
MAX_LEN = 32
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.002
TOTAL_DATA = 100000

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
stemmer = SnowballStemmer("english")

# Load dataset
try:
    df = pd.read_csv('data/go_emotions_dataset.csv')
except FileNotFoundError:
    print("Error: 'data/go_emotions_dataset.csv' not found in 'data'")
    exit() # Exit if dataset is not found

df = df.head(TOTAL_DATA) # Limiting for testing purposes as per your request

# Define emotion labels
labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Create a mapping from label name to its index
label_to_id = {label: i for i, label in enumerate(labels)}
id_to_label = {i: label for i, label in enumerate(labels)}

# Convert to multi-hot encoder
emotion_columns_in_df = [col for col in df.columns if col in labels]
if not emotion_columns_in_df:
    print("Error: Emotion columns not found in DataFrame. Please check your CSV column names.")
    print("Expected columns like:", labels[:5]) # Print first 5 expected labels for example
    exit()

# Convert boolean columns to integer (0 or 1)
df[emotion_columns_in_df] = df[emotion_columns_in_df].astype(int)
df['multi_hot_labels'] = df[labels].values.tolist()


# --- Data Splitting ---
# Split the dataset into training, validation, and test sets
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

# --- PyTorch Dataset and DataLoader ---
class GoEmotionsDataset(Dataset):
    def __init__(self, texts, multi_hot_labels, tokenizer, max_len):
        self.texts = texts
        self.multi_hot_labels = multi_hot_labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        # Convert list of ints to float tensor for BCEWithLogitsLoss
        labels = torch.tensor(self.multi_hot_labels[idx], dtype=torch.float)

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt', # Return PyTorch tensors
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }

# --- Model and Tokenizer Initialization for Training ---
# We'll use a BERT-based model for classification
MODEL_NAME = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# The classification model will be fine-tuned on the GoEmotions dataset
# problem_type="multi_label_classification" is crucial for handling multiple labels
model_for_training = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    problem_type="multi_label_classification",
    id2label=id_to_label,
    label2id=label_to_id
)

# Set device for training (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_for_training.to(device)

# Create Dataset instances
train_dataset = GoEmotionsDataset(
    texts=train_df['text'].tolist(),
    multi_hot_labels=train_df['multi_hot_labels'].tolist(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)
val_dataset = GoEmotionsDataset(
    texts=val_df['text'].tolist(),
    multi_hot_labels=val_df['multi_hot_labels'].tolist(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)
test_dataset = GoEmotionsDataset(
    texts=test_df['text'].tolist(),
    multi_hot_labels=test_df['multi_hot_labels'].tolist(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

# Create DataLoader instances
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# --- Training Components ---
optimizer = torch.optim.AdamW(model_for_training.parameters(), lr=LEARNING_RATE)
# BCEWithLogitsLoss combines sigmoid and Binary Cross Entropy,
# which is numerically more stable for multi-label classification.
criterion = torch.nn.BCEWithLogitsLoss()

# --- Training and Evaluation Functions ---
def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, data in enumerate(tqdm(data_loader, desc="Training")):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits # Get raw logits from the model output

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating"):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Apply sigmoid to get probabilities, then threshold to get binary predictions
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).cpu().numpy() # Threshold at 0.5
            all_predictions.extend(predictions)
            all_true_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)

    # Calculate metrics
    # For multi-label, macro F1 is often preferred for imbalanced datasets
    macro_f1 = f1_score(all_true_labels, all_predictions, average='macro', zero_division=0)
    micro_f1 = f1_score(all_true_labels, all_predictions, average='micro', zero_division=0)
    accuracy = accuracy_score(all_true_labels, all_predictions) # Exact match accuracy

    return avg_loss, macro_f1, micro_f1, accuracy

# --- Main Training Loop ---
print("\n--- Starting Model Training ---")
best_val_f1 = -1
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    train_loss = train_epoch(model_for_training, train_dataloader, criterion, optimizer, device)
    val_loss, val_macro_f1, val_micro_f1, val_accuracy = evaluate_model(model_for_training, val_dataloader, criterion, device)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Macro F1: {val_macro_f1:.4f}, Micro F1: {val_micro_f1:.4f}, Accuracy: {val_accuracy:.4f}")

    # Save the best model based on validation macro F1
    if val_macro_f1 > best_val_f1:
        best_val_f1 = val_macro_f1
        torch.save(model_for_training.state_dict(), 'best_goemotions_model2.pt')
        print("Saved best model!")

print("\n--- Training Complete ---")

# --- Evaluate on Test Set ---
print("\n--- Evaluating on Test Set ---")
model_for_training.load_state_dict(torch.load('best_goemotions_model.pt')) # Load best model
test_loss, test_macro_f1, test_micro_f1, test_accuracy = evaluate_model(model_for_training, test_dataloader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Macro F1: {test_macro_f1:.4f}, Test Micro F1: {test_micro_f1:.4f}, Test Accuracy: {test_accuracy:.4f}")