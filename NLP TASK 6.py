import nltk
from nltk.util import ngrams
from nltk.lm import Laplace
from nltk.tokenize import word_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline

# Download punkt tokenizer
nltk.download('punkt')

# Function to train N-gram model with Laplace smoothing
def ngram_smoothing(sentence, n):
    tokens = word_tokenize(sentence.lower())  # tokenize
    # For NLTK LM, training data should be a list of lists of tokens
    train_data, padded_sents = padded_everygram_pipeline(n, [tokens])
    model = Laplace(n)
    model.fit(train_data, padded_sents)
    return model, tokens

# Input
sentence = input("Enter a sentence: ")
n = int(input("Enter the value of N for N-grams: "))

# Train model
model, tokens = ngram_smoothing(sentence, n)

# Get context (last n-1 words)
if n > 1:
    context = tuple(tokens[-n+1:])
else:
    context = ()

# Generate 3 next words
next_words = model.generate(3, text_seed=context)

print("Next words:", ' '.join(next_words))
