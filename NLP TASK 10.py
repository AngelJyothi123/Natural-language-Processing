from bs4 import BeautifulSoup
import spacy

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

def pos_tag_and_extract_info(text):
    doc = nlp(text)
    nouns, verbs, adjectives, entities = [], [], [], []

    for token in doc:
        if token.pos_ == "NOUN":
            nouns.append(token.text)
        elif token.pos_ == "VERB":
            verbs.append(token.text)
        elif token.pos_ == "ADJ":
            adjectives.append(token.text)

    for entity in doc.ents:
        entities.append((entity.text, entity.label_))

    return nouns, verbs, adjectives, entities

# Example HTML document
web_document = """
<html>
<head>
<title>Example Web Page</title>
</head>
<body>
<p>This is an example web page. It contains some text with various parts of speech.</p>
<p>For example, "The cat jumps over the lazy dog" contains a noun, a verb, and prepositions.</p>
</body>
</html>
"""

# Extract text from HTML
def extract_text_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text()

# Run processing
text_content = extract_text_from_html(web_document)
nouns, verbs, adjectives, entities = pos_tag_and_extract_info(text_content)

print("Nouns:", nouns)
print("Verbs:", verbs)
print("Adjectives:", adjectives)
print("Entities:", entities)