import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from PyPDF2 import PdfReader
from docx import Document

# Load the T5-small model and tokenizer
@st.cache_resource  # Cache the model to avoid reloading
def load_model():
    model_name = "t5-small"
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer


# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def split_text_into_chunks(text, max_length=200):
    """
    Split the text into smaller chunks based on the max length.
    Ensure no chunk cuts off a sentence in the middle.
    """
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:  # Add the last chunk
        chunks.append(current_chunk.strip())

    return chunks

def translate_large_text(text, model, tokenizer):
    """
    Translate large text by splitting it into manageable chunks.
    """
    chunks = split_text_into_chunks(text)
    translated_chunks = []

    for chunk in chunks:
        translated_chunk = translate_text(chunk, model, tokenizer)
        translated_chunks.append(translated_chunk)

    return " ".join(translated_chunks)


# Load model and tokenizer
model, tokenizer = load_model()

# Streamlit app layout
st.title("English to French Translator")
st.markdown("Enter an English sentence below, and the model will translate it into French.")

# Input box for the user
english_text = st.text_area("Enter English text:", "")

# Translate function
def translate_text(text, model, tokenizer):
    input_text = f"translate English to French: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model.generate(**inputs, max_length=1000)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# Button to translate
if st.button("Translate"):
    if english_text.strip() == "":
        st.warning("Please enter some text to translate.")
    else:
        with st.spinner("Translating..."):
            french_translation = translate_text(english_text, model, tokenizer)
            st.success("Translation Complete!")
            st.text_area("French Translation:", french_translation, height=200)


# Function to extract text from uploaded files
def extract_text_from_file(uploaded_file):
    file_type = uploaded_file.name.split(".")[-1].lower()
    content = ""

    if file_type == "txt":
        content = uploaded_file.read().decode("utf-8")
    elif file_type == "pdf":
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            content += page.extract_text()
    elif file_type == "docx":
        doc = Document(uploaded_file)
        for para in doc.paragraphs:
            content += para.text + "\n"
    else:
        st.error("Unsupported file format. Please upload a TXT, PDF, or DOCX file.")
    
    return content.strip()

# Streamlit app layout
st.markdown("Upload a file in TXT, PDF, or DOCX format, and the app will translate its contents into French.")

uploaded_file = st.file_uploader("Upload your file here", type=["txt", "pdf", "docx"])

if uploaded_file:
    with st.spinner("Processing the file..."):
        content = extract_text_from_file(uploaded_file)
        if content:
            st.text_area("File Content (English):", content, height=200)

            if st.button("Translate to French"):
                with st.spinner("Translating..."):
                    translated_content = translate_large_text(content, model, tokenizer)
                    st.success("Translation Complete!")
                    st.text_area("Translated Content (French):", translated_content, height=200)
        else:
            st.error("The file is empty or could not be processed.")

# Footer
st.markdown("---")
st.markdown("**Powered by T5-small and Streamlit. Made by Akshit Singh.**")
