import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Load pre-trained model and tokenizer
@st.cache_resource
def load_model():
    model_name = "Helsinki-NLP/opus-mt-en-hi"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    print("Model and Tokenizer Loaded Successfully!")
    return tokenizer, model

# Load model and tokenizer
tokenizer, model = load_model()

# Function to translate text
def translate_text(text):
    if not text:
        return "Please enter some text to translate."

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    print(f"Tokenized input: {inputs}")  # Print the tokenized input

    # Generate translation
    translated_ids = model.generate(**inputs, max_length=512, num_beams=5, early_stopping=True)
    translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)

    print(f"Translated text: {translated_text}")  # Print the translated text
    return translated_text

# Streamlit App UI
st.title("English to Hindi Translation App")
st.write("Translate English sentences to Hindi using a pre-trained Hugging Face model.")

# Text input from user
input_text = st.text_area("Enter text in English:", placeholder="Type here...")

# Translate button
if st.button("Translate"):
    with st.spinner("Translating..."):
        result = translate_text(input_text)
        st.subheader("Translated Text:")
        st.write(result)
