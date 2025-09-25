import nltk
from nltk.corpus import treebank
from nltk.tag.hmm import HiddenMarkovModelTrainer
from nltk.tag import PerceptronTagger
from nltk.classify import MaxentClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ===============================
# Download required resources
# ===============================
nltk.download('treebank')
nltk.download('punkt')
nltk.download('punkt_tab')  # fix for new NLTK
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# ===============================
# Load dataset
# ===============================
corpus = list(treebank.tagged_sents())
train_data = corpus[:int(0.8 * len(corpus))]
test_data = corpus[int(0.8 * len(corpus)):]

# ===============================
# HMM Tagger
# ===============================
print("\nTraining HMM Tagger...")
hmm_trainer = HiddenMarkovModelTrainer()
hmm_tagger = hmm_trainer.train(train_data)

hmm_preds, hmm_gold = [], []
for sent in test_data:
    words = [w for w, t in sent]
    gold_tags = [t for w, t in sent]
    pred_tags = [t for w, t in hmm_tagger.tag(words)]
    hmm_preds.extend(pred_tags)
    hmm_gold.extend(gold_tags)
hmm_acc = accuracy_score(hmm_gold, hmm_preds)
print("HMM Accuracy:", hmm_acc)

# ===============================
# Perceptron Tagger
# ===============================
print("\nEvaluating Perceptron Tagger...")
perc_tagger = PerceptronTagger()
perc_preds, perc_gold = [], []
for sent in test_data:
    words = [w for w, t in sent]
    gold_tags = [t for w, t in sent]
    pred_tags = [t for w, t in perc_tagger.tag(words)]
    perc_preds.extend(pred_tags)
    perc_gold.extend(gold_tags)
perc_acc = accuracy_score(perc_gold, perc_preds)
print("Perceptron Accuracy:", perc_acc)

# ===============================
# MaxEnt (Log-linear) Tagger
# ===============================
print("\nTraining MaxEnt Tagger (small subset for speed)...")

def pos_features(sentence, i):
    word = sentence[i]
    return {
        "word": word,
        "is_first": i == 0,
        "is_last": i == len(sentence) - 1,
        "prev_word": "" if i == 0 else sentence[i - 1],
        "next_word": "" if i == len(sentence) - 1 else sentence[i + 1],
        "is_capitalized": word[0].upper() == word[0],
        "suffix-1": word[-1:],
        "suffix-2": word[-2:],
    }

# Train on smaller subset for speed
train_feats = []
for sent in train_data[:2000]:  # limit size
    words, tags = zip(*sent)
    for i in range(len(words)):
        train_feats.append((pos_features(words, i), tags[i]))

maxent_classifier = MaxentClassifier.train(train_feats, algorithm="iis", max_iter=10, trace=0)

# Evaluate on smaller test set
maxent_preds, maxent_gold = [], []
for sent in test_data[:500]:
    words, tags = zip(*sent)
    for i in range(len(words)):
        pred = maxent_classifier.classify(pos_features(words, i))
        maxent_preds.append(pred)
        maxent_gold.append(tags[i])
maxent_acc = accuracy_score(maxent_gold, maxent_preds)
print("MaxEnt Accuracy:", maxent_acc)

# ===============================
# Final Comparison
# ===============================
print("\n===== Final Results =====")
print(f"HMM Accuracy        : {hmm_acc:.4f}")
print(f"Perceptron Accuracy : {perc_acc:.4f}")
print(f"MaxEnt Accuracy     : {maxent_acc:.4f}")

# ===============================
# Confusion Matrix Heatmap
# ===============================
print("\nGenerating confusion matrix for Perceptron vs Gold...")
labels = sorted(list(set(perc_gold)))
cm = confusion_matrix(perc_gold, perc_preds, labels=labels)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, xticklabels=labels, yticklabels=labels, cmap="Blues", cbar=False)
plt.title("Confusion Matrix: Perceptron POS Tagger")
plt.xlabel("Predicted")
plt.ylabel("Gold Standard")
plt.show()
