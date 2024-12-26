from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import MarianMTModel, MarianTokenizer
import torch
import streamlit as st
import PyPDF2

# Load MarianMT model and tokenizer
model_path = "/content/drive/MyDrive/models/en-ru-translation_model"  # Replace with your actual model path
model = MarianMTModel.from_pretrained(model_path, local_files_only=True)
tokenizer = MarianTokenizer.from_pretrained(model_path, local_files_only=True)

# Load Sentence-BERT model
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Translation function
def translate(input_sentence):
    inputs = tokenizer(input_sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        translated_tokens = model.generate(inputs["input_ids"], max_length=128, num_beams=4, early_stopping=True)
    translated_sentence = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_sentence

# Function to calculate semantic similarity using Sentence-BERT
def calculate_semantic_similarity(original_text, translated_text):
    embeddings = sbert_model.encode([original_text, translated_text])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
    return similarity[0][0]

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Streamlit UI
st.set_page_config(page_title="English-to-Russian Translator", layout="centered", initial_sidebar_state="expanded")

# Page title and description
st.title("🌍 English-to-Russian Translator")
st.markdown(
    """
    This app translates English sentences to Russian using a fine-tuned MarianMT model and calculates the semantic similarity between the original and translated text using Sentence-BERT.
    """
)

# Input form
st.subheader("Translate Your Sentence")

# Option to upload PDF file
uploaded_file = st.file_uploader("Upload a PDF file for translation", type=["pdf"])

input_sentence = ""
if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        input_sentence = extract_text_from_pdf(uploaded_file)
        st.write("**Extracted Text from PDF:**")
        st.write(input_sentence)

# Text area for manual input
input_sentence = st.text_area("Or enter your English sentence below:", value=input_sentence, placeholder="Type your sentence here...")

if st.button("Translate"):
    if not input_sentence.strip():
        st.warning("Please enter a sentence before translating.")
    else:
        # Translate and calculate similarity
        with st.spinner("Translating and analyzing..."):
            translated_sentence = translate(input_sentence)
            similarity_score = calculate_semantic_similarity(input_sentence, translated_sentence)
        
        # Display results
        st.success("Translation Complete!")
        st.write(f"**Original Sentence (English):** {input_sentence}")
        st.write(f"**Translated Sentence (Russian):** {translated_sentence}")
        st.write(f"**Semantic Similarity Score:** {similarity_score:.4f}")
        
        # Add visual feedback for similarity
        if similarity_score > 0.8:
            st.success("The translation is highly semantically similar to the original sentence!")
        elif similarity_score > 0.5:
            st.warning("The translation has moderate semantic similarity.")
        else:
            st.error("The translation has low semantic similarity.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Asthik s Shetty")

