import nltk
from nltk.tag.hmm import HiddenMarkovModelTrainer
from nltk.corpus import treebank

# Download required resources
nltk.download('treebank')
nltk.download('punkt')

# Load dataset
corpus = list(treebank.tagged_sents())
train_data = corpus[:3000]
test_data = corpus[3000:]

# Train HMM tagger
trainer = HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)

# Input a sentence from user
sentence = input("Enter a sentence: ")
tokens = nltk.word_tokenize(sentence)

# POS tagging
tagged_sentence = hmm_tagger.tag(tokens)
print("Tagged Sentence:", tagged_sentence)
