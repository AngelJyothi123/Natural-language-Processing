import re
import wikipedia
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
import torch
import torch.nn as nn

# ---------------------------
# LSTM Model (placeholder)
# ---------------------------
class ChunkerModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(out[:, -1, :]))


# ---------------------------
# Fetch Wikipedia page
# ---------------------------
def fetch_text(title):
    """Fetch Wikipedia page text by title"""
    try:
        page = wikipedia.page(title)
        text = page.content
        return text
    except Exception as e:
        print(f"Error fetching page: {e}")
        return None


# ---------------------------
# Preprocess text
# ---------------------------
def preprocess(text):
    """Tokenize and pad text for model input"""
    tok = Tokenizer(num_words=5000)
    tok.fit_on_texts([text])
    seq = tok.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=100, padding='post')
    return seq, tok


# ---------------------------
# Segment text into chunks
# ---------------------------
def segment_text(title):
    text = fetch_text(title)
    if not text:
        return ["No text found for this Wikipedia page."]
    
    # Preprocess (optional for model input)
    seq, tok = preprocess(text)
    
    # Split text into sentences using period or newline
    chunks = re.split(r'\. |\n', text)
    
    return chunks[:5]  # Return first 5 chunks


# ---------------------------
# Run example
# ---------------------------
if __name__ == "__main__":
    title = "Natural language processing"
    print("Extracted Chunks:\n")
    for chunk in segment_text(title):
        print(chunk)
