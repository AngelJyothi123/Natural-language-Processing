import string
import random
import nltk
from nltk.corpus import stopwords, reuters
from collections import Counter, defaultdict
from nltk import FreqDist, ngrams

# Downloads
nltk.download('punkt')
nltk.download('punkt_tab')   # 🔹 Add this line
nltk.download('stopwords')
nltk.download('reuters')

sents = reuters.sents()
stop_word = set(stopwords.words('english'))
string.punctuation = string.punctuation + ' " ' + ' " ' + ' - ' + ' _ '
removal_list = list(stop_word) + list(string.punctuation) + ['\t', 'rt']

unigram = []
bigram = []
trigram = []
tokenized_text = []

for sentence in sents:
    sentence = list(map(lambda x: x.lower(), sentence))
    for word in sentence:
        if word == '.':
            sentence.remove(word)
        else:
            unigram.append(word)
            tokenized_text.append(word)

    bigram.extend(list(ngrams(sentence, 2, pad_left=True, pad_right=True)))
    trigram.extend(list(ngrams(sentence, 3, pad_left=True, pad_right=True)))

def remove_stopwords(x):
    y = []
    for pair in x:
        count = 0
        for word in pair if isinstance(pair, tuple) else [pair]:
            if word in removal_list:
                count = count or 0
            else:
                count = count or 1
        if count == 1:
            y.append(pair)
    return y

unigram = remove_stopwords(unigram)
bigram = remove_stopwords(bigram)
trigram = remove_stopwords(trigram)

freq_uni = FreqDist(unigram)
freq_bi = FreqDist(bigram)
freq_tri = FreqDist(trigram)

d = defaultdict(Counter)
for a, b, c in freq_tri:
    if (a is not None) and (b is not None) and (c is not None):
        d[a, b][c] += freq_tri[a, b, c]

def pick_word(counter):
    "choose a random element"
    return random.choice(list(counter.elements()))

prefix = ("he", "is")
print(" ".join(prefix))
s = " ".join(prefix)
for i in range(19):
    suffix = pick_word(d[prefix])
    s = s + ' ' + suffix
    print(s)
    prefix = (prefix[1], suffix)
