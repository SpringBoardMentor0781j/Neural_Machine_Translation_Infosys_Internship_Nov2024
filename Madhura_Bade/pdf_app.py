
import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber  # For PDF handling
from docx import Document  # For Word Documents
import os
import time
import io

# Load MarianMT model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-es"  # Path to your fine-tuned MarianMT model
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Translation function
# Translation function
def translate(input_sentence):
    inputs = tokenizer(input_sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        translated_tokens = model.generate(inputs["input_ids"], num_beams=4, early_stopping=True)
    translated_sentence = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_sentence

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()  # Extract text from each page
    return text

# Function to extract text from Word (.docx) files
def extract_text_from_docx(docx_file):
    document = Document(docx_file)
    text = " ".join(paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip())
    return text


# Streamlit App with Tabs
st.markdown("<h1 style='text-align: center; color: blue; font-family:'Times New Roman'; >English to Spanish Translator</h1>", unsafe_allow_html=True)


# Create Tabs
tab1, tab2 = st.tabs(["Text Translation", "Document Translation"])

# Tab 1: Text Translation
with tab1:
    st.markdown("<h2 style='text-align: center; 'font-size: 20px;'>Text Translation </h2>", unsafe_allow_html=True)

    input_sentence = st.text_area("English Sentence:")
    if st.button("Translate"):
        if input_sentence.strip() == "":
            st.warning("Please enter a sentence to translate.")  # Warning for empty input
        else:
            start_time = time.time()
            translated_sentence = translate(input_sentence)
            end_time = time.time()  # End timing
            translation_time = end_time - start_time
            st.write("### Translated Sentence:")
            st.text_area("Spanish Sentence:", translated_sentence, height=150)
             # Calculate time taken
            st.write(f"**Time taken for translation:** {translation_time:.2f} seconds")
            


# Tab 2: Document Translation
with tab2:
    st.markdown("<h2 style='text-align: center; 'font-size: 20px;'>Document Translation </h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a document (PDF, Word, or Plain Text):", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        # Extract content based on file type
        file_type = os.path.splitext(uploaded_file.name)[1].lower()
        if file_type == ".pdf":
            st.write("Extracting text from the PDF file...")
            content = extract_text_from_pdf(uploaded_file)
        elif file_type == ".docx":
            st.write("Extracting text from the Word document...")
            content = extract_text_from_docx(uploaded_file)
        elif file_type == ".txt":
            st.write("Extracting text from the plain text file...")
            content = uploaded_file.read().decode("utf-8")
        else:
            st.error("# Unsupported file type. Please upload a PDF, Word, or plain text file.")
            content = None

        # Display and translate content
        if content:
            st.write("## Original Content:")
            st.text_area(" ", content, height=200)
            start_time = time.time()
            chunks = [content[i:i+100] for i in range(0, len(content), 100)]  # Chunk size: 1000 characters
            translated_text = ""
            for chunk in chunks:
                translated_text += translate(chunk)
            end_time = time.time()
            translation_time = end_time - start_time
            st.write("## Translated Content:")
            st.text_area(" ", translated_text, height=200)
            st.write(f"**Time taken for translation:** {translation_time:.2f} seconds")
            # Add download button for translated content
            st.download_button(
                label="Download Translated Content",
                data=translated_text,
                file_name="translated_document.txt",
                mime="text/plain"
            )
st.markdown("<h4 style='text-align: center;bottom: 0; left: 0; color: grey;width: 100%; font-family:'Times New Roman'; '>Infosys Springboard Internship</h4>",unsafe_allow_html=True
)

