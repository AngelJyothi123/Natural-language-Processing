import nltk
from nltk.corpus import treebank
from nltk.tag import hmm

# Download necessary NLTK resources
nltk.download('treebank')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('maxent_treebank_pos_tagger')

# Load corpus
corpus = list(treebank.tagged_sents())

# Train-test split
train_data = corpus[:int(0.8 * len(corpus))]
test_data = corpus[int(0.8 * len(corpus)):]

# Train HMM Tagger
hmm_trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = hmm_trainer.train(train_data)

# Evaluate
hmm_accuracy = hmm_tagger.accuracy(test_data)
print(f"HMM Tagger Accuracy: {hmm_accuracy:.4f}")

