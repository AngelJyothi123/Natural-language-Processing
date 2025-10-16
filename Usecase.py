from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load a pre-trained English->Hindi translation model
model_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Input English text
english_text = "I am doing good?"

# Tokenize the text
inputs = tokenizer(english_text, return_tensors="pt")

# Generate translation
translated_tokens = model.generate(**inputs)

# Decode translation to Hindi text
hindi_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

print("English:", english_text)
print("Hindi:", hindi_text)
