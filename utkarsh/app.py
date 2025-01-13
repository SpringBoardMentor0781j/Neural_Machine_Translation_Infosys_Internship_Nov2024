from google.colab import drive
from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import os

# Mount Google Drive
drive.mount('/content/drive')

# Load models and tokenizers
model_path = '/content/drive/MyDrive/Project/nmt_model.h5'
eng_tokenizer_path = '/content/drive/MyDrive/Project/eng_tokenizer.pkl'
fr_tokenizer_path = '/content/drive/MyDrive/Project/fr_tokenizer.pkl'

# Load the trained model
nmt_model = load_model(model_path)

# Load the tokenizers
with open(eng_tokenizer_path, 'rb') as f:
    eng_tokenizer = pickle.load(f)
with open(fr_tokenizer_path, 'rb') as f:
    fr_tokenizer = pickle.load(f)

# Reverse-lookup dictionary for French tokenizer
reverse_fr_index = {v: k for k, v in fr_tokenizer.word_index.items()}

# Define the translation function
def translate_sentence(input_sentence, max_len=20):
    """
    Translate an input English sentence to French using the trained NMT model.
    """
    # Preprocess the input sentence
    input_sentence = input_sentence.lower().strip()
    input_sequence = eng_tokenizer.texts_to_sequences([input_sentence])
    input_sequence = pad_sequences(input_sequence, maxlen=max_len, padding='post')

    # Encode the input sentence
    encoder_inputs = nmt_model.input[0]
    encoder_outputs, state_h, state_c = nmt_model.layers[4].output
    encoder_states = [state_h, state_c]

    encoder_model = load_model(model_path)
    states_value = encoder_model.predict(input_sequence)

    # Initialize target sequence with <start> token
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = fr_tokenizer.word_index['je']  # Assuming 'je' is the start token

    # Generate translated sentence
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_fr_index.get(sampled_token_index, '')

        if sampled_word == 'salutations' or len(decoded_sentence.split()) >= max_len:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word

        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence.strip()

# Flask web app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    english_text = request.form['english_text']
    translated_text = translate_sentence(english_text)
    return render_template('index.html', english_text=english_text, translated_text=translated_text)