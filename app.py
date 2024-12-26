import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import PyPDF2  # To handle PDF file uploads and text extraction

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

    # Generate translation
    translated_ids = model.generate(**inputs, max_length=512, num_beams=5, early_stopping=True)
    translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)

    return translated_text

# Function to extract text from uploaded PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Streamlit App UI
st.title("English to Hindi Translation App")
st.write("Translate English sentences or PDF content to Hindi using a pre-trained Hugging Face model.")

# Option 1: Text Input from user
st.header("Translate Text")
input_text = st.text_area("Enter text in English:", placeholder="Type here...")

if st.button("Translate Text"):
    with st.spinner("Translating..."):
        result = translate_text(input_text)
        st.subheader("Translated Text:")
        st.write(result)

# Option 2: PDF Upload for Translation
st.header("Upload PDF for Translation")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        extracted_text = extract_text_from_pdf(uploaded_file)
        st.text_area("Extracted Text:", extracted_text, height=300)

    # Translate the extracted text
    if st.button("Translate PDF Content"):
        with st.spinner("Translating PDF content..."):
            translated_pdf_text = translate_text(extracted_text)
            st.subheader("Translated PDF Content:")
            st.write(translated_pdf_text)
