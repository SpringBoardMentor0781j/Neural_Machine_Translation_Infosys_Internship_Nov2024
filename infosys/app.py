import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sentencepiece as spm
import base64
import time

# Function to load the model and tokenizer for a specific translation model
def load_model_and_tokenizer(model_path):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

# Translation function for individual sentences
def translate(sentence, model, tokenizer, sp_source, sp_target, max_length=1000):
    input_ids = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).input_ids
    outputs = model.generate(
        input_ids, 
        max_length=max_length, 
        num_beams=5, 
        early_stopping=True, 
        return_dict_in_generate=True, 
        output_scores=True
    )
    translated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
    # Calculate confidence score based on token-level log probabilities
    scores = outputs.scores
    avg_log_prob = sum([score.max().item() for score in scores]) / len(scores)
    confidence_score = (2 ** avg_log_prob) * 100  # Convert to percentage
    
    return translated_text, confidence_score

# Function to load SentencePiece tokenizers
def load_spm_models(model_path):
    sp_source = spm.SentencePieceProcessor(model_file=f"{model_path}/source.spm")
    sp_target = spm.SentencePieceProcessor(model_file=f"{model_path}/target.spm")
    return sp_source, sp_target

# Function to translate multi-line text
def translate_multiline(text, model, tokenizer, sp_source, sp_target):
    lines = text.splitlines()
    translated_lines = []
    total_confidence = 0

    for line in lines:
        if line.strip():
            translated_text, confidence_score = translate(line, model, tokenizer, sp_source, sp_target)
            translated_lines.append(translated_text)
            total_confidence += confidence_score

    avg_confidence = total_confidence / len(translated_lines) if translated_lines else 0
    return "\n".join(translated_lines), avg_confidence

# Streamlit UI
st.title("Language Translator")
st.write("Translate text between multiple languages.")

# Define supported languages and their model paths
languages = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "Italian": "it"
}

model_paths = {
    "en-es": "models/en-es-translation_model",
    "en-fr": "models/en-fr-translation_model",
    "en-it": "models/en-it-translation_model",
    "es-en": "models/es-en-translation_model",
    "fr-en": "models/fr-en-translation_model",
    "it-en": "models/it-en-translation_model"
}

# Language selection boxes
col1, col2 = st.columns([3, 3])

with col1:
    input_language = st.selectbox("Input Language:", list(languages.keys()), key="input_language")

with col2:
    if input_language == "English":
        target_languages = [lang for lang in languages.keys() if lang != "English"]
    else:
        target_languages = ["English"]
    target_language = st.selectbox("Target Language:", target_languages, key="target_language")

# Process based on the selected input method
input_method = st.radio("Choose input method:", ["Enter text manually", "Upload a file"])

if input_method == "Enter text manually":
    # Multi-line text input
    input_text = st.text_area("Enter text:", height=200)

    if st.button("Translate"):
        if input_text.strip():
            with st.spinner("Translating..."):
                start_time = time.time()

                # Determine the model path based on selected languages
                model_key = f"{languages[input_language]}-{languages[target_language]}"
                model_path = model_paths[model_key]

                # Load the selected model and tokenizers
                model, tokenizer = load_model_and_tokenizer(model_path)
                sp_source, sp_target = load_spm_models(model_path)

                # Translate multi-line input
                translated_text, avg_confidence = translate_multiline(input_text, model, tokenizer, sp_source, sp_target)
                translation_time = time.time() - start_time

                # Display translated text
                st.success(f"Translation completed in {translation_time:.2f} seconds.")
                st.write(f"Average Confidence Score: {avg_confidence:.2f}%")
                st.text_area("Translated Text:", translated_text, height=300)
        else:
            st.error("Please enter text to translate.")

elif input_method == "Upload a file":
    # File upload
    uploaded_file = st.file_uploader("Upload a text file for translation:", type="txt")

    if st.button("Translate"):
        if uploaded_file is not None:
            with st.spinner("Translating..."):
                file_content = uploaded_file.read().decode("utf-8")
                start_time = time.time()

                # Determine the model path based on selected languages
                model_key = f"{languages[input_language]}-{languages[target_language]}"
                model_path = model_paths[model_key]

                # Load the selected model and tokenizers
                model, tokenizer = load_model_and_tokenizer(model_path)
                sp_source, sp_target = load_spm_models(model_path)

                # Translate file content
                translated_file_content, avg_confidence = translate_multiline(file_content, model, tokenizer, sp_source, sp_target)
                translation_time = time.time() - start_time

                # Display translated file content
                st.success(f"Translation completed in {translation_time:.2f} seconds.")
                st.write(f"Average Confidence Score: {avg_confidence:.2f}%")
                st.text_area("Translated Text:", translated_file_content, height=300)

                # Provide download option for translated text
                b64_content = base64.b64encode(translated_file_content.encode()).decode()
                href = f'<a href="data:file/txt;base64,{b64_content}" download="{input_language}-{target_language}-translated.txt">Download Translated File</a>'
                st.markdown(href, unsafe_allow_html=True)
        else:
            st.error("Please upload a file to translate.")
