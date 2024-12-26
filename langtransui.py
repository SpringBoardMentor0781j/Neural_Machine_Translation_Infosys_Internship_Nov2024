# -*- coding: utf-8 -*-
import os
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pickle
from nltk.translate.bleu_score import sentence_bleu
import gradio as gr
from docx import Document  # For reading .docx files
import pandas as pd  # For handling .csv files

# Load necessary data
with open('/Users/Naresh Chandra/OneDrive/Desktop/language translation eng-fr/language-translator-ml-codes/training_data.pkl', "rb") as f:
    training_data = pickle.load(f)

input_characters = training_data['input_characters']
target_characters = training_data['target_characters']
max_input_length = training_data['max_input_length']
max_target_length = training_data['max_target_length']
num_en_chars = training_data['num_en_chars']
num_dec_chars = training_data['num_dec_chars']

# Load the trained model
model = load_model("s2s.model.keras")

# Helper functions for translation
def preprocess_text(input_text, characters, max_length):
    input_text = input_text.lower()
    cv = CountVectorizer(binary=True, tokenizer=lambda txt: txt.split(), stop_words=None, analyzer='char')
    cv.fit(characters)

    encoded_data = cv.transform(list(input_text)).toarray().tolist()
    pad_data = [0] * len(characters)
    if '\t' in characters:
        pad_data[characters.index('\t')] = 1

    if len(encoded_data) < max_length:
        for _ in range(max_length - len(encoded_data)):
            encoded_data.append(pad_data)

    return np.array([encoded_data], dtype="float32")
def translate(input_text, input_characters, target_characters, max_input_length, max_target_length):
    en_in_data = preprocess_text(input_text, input_characters, max_input_length)

    # Validate input shape
    if en_in_data.shape[-1] != len(input_characters):
        raise ValueError(
            f"Mismatch in input dimensions: expected {len(input_characters)}, got {en_in_data.shape[-1]}"
        )

    # Initialize decoder input
    decoder_input = np.zeros((1, max_target_length, len(target_characters)))
    decoder_input[0, 0, target_characters.index('\t')] = 1

    translated = ""
    for i in range(max_target_length - 1):
        output = model.predict([en_in_data, decoder_input])

        # Get the character index with the highest probability
        sampled_token_index = np.argmax(output[0, i, :])
        sampled_char = target_characters[sampled_token_index]

        # Stop at the end character
        if sampled_char == '\n':
            break

        translated += sampled_char
        decoder_input[0, i + 1, sampled_token_index] = 1

    return translated.strip()


# Reverse translation function
def reverse_translate(input_text):
    return translate(input_text, target_characters, input_characters, max_target_length, max_input_length)

# File reading and processing
def read_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension == ".txt":
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    elif file_extension == ".csv":
        df = pd.read_csv(file_path)
        return "\n".join(df.iloc[:, 0].astype(str))  # Assuming first column has the text
    elif file_extension == ".docx":
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        raise ValueError("Unsupported file type. Please upload .txt, .csv, or .docx files.")

def process_file(file, is_reverse):
    text = read_file(file.name)
    if is_reverse:
        return reverse_translate(text)
    return translate(text, input_characters, target_characters, max_input_length, max_target_length)

# BLEU score evaluation
def evaluate_translation(input_text, target_text):
    translated_text = translate(input_text, input_characters, target_characters, max_input_length, max_target_length)
    target_tokens = target_text.split()
    translated_tokens = translated_text.split()
    return sentence_bleu([target_tokens], translated_tokens)

# Gradio interface
def translate_text_interface(input_text, is_reverse):
    if is_reverse:
        return reverse_translate(input_text)
    return translate(input_text, input_characters, target_characters, max_input_length, max_target_length)

def handle_file_interface(file, is_reverse):
    return process_file(file, is_reverse)

# Gradio app with both functionalities
app = gr.Blocks()

with app:
    gr.Markdown("# Bidirectional Translator (English ↔ French)")
    with gr.Tabs():
        with gr.Tab("Text Translation"):
            input_text = gr.Textbox(label="Enter Text", placeholder="Enter text here...")
            is_reverse = gr.Checkbox(label="Reverse Translation (French to English)")
            translate_button = gr.Button("Translate")
            output_text = gr.Textbox(label="Translated Text")
            translate_button.click(translate_text_interface, inputs=[input_text, is_reverse], outputs=output_text)
        
        with gr.Tab("File Translation"):
            file_input = gr.File(label="Upload File", file_types=[".txt", ".csv", ".docx"])
            is_reverse_file = gr.Checkbox(label="Reverse Translation (French to English)")
            file_translate_button = gr.Button("Translate File")
            file_output = gr.Textbox(label="File Translation Output")
            file_translate_button.click(handle_file_interface, inputs=[file_input, is_reverse_file], outputs=file_output)
        
        with gr.Tab("Evaluate Translation"):
            eval_input_text = gr.Textbox(label="Input Text", placeholder="Enter source text here...")
            eval_target_text = gr.Textbox(label="Target Text", placeholder="Enter target text here...")
            evaluate_button = gr.Button("Evaluate")
            bleu_score = gr.Textbox(label="BLEU Score")
            evaluate_button.click(evaluate_translation, inputs=[eval_input_text, eval_target_text], outputs=bleu_score)

app.launch()