# -*- coding: utf-8 -*-
# DataFlair Project

# Load all the required modules
from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, LSTM, Dense
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pickle
import gradio as gr

# Initialize all variables
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

# Read dataset file
with open('/Users/Naresh Chandra/OneDrive/Desktop/language translation eng-fr/language-translator-ml-codes/eng-french.txt', 'r', encoding='utf-8') as f:
    rows = f.read().split('\n')

# Read first 10,000 rows from the dataset
for row in rows[:10000]:
    # Split input and target by '\t' = tab
    input_text, target_text = row.split('\t')
    # Add '\t' at start and '\n' at end of text
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text.lower())
    target_texts.append(target_text.lower())
    # Split characters from text and add to respective sets
    input_characters.update(list(input_text.lower()))
    target_characters.update(list(target_text.lower()))

# Sort input and target characters 
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))

# Get the total length of input and target characters
num_en_chars = len(input_characters)
num_dec_chars = len(target_characters)

# Get the maximum length of input and target text
max_input_length = max([len(i) for i in input_texts])
max_target_length = max([len(i) for i in target_texts])

def bagofcharacters(input_texts, target_texts):
    # Initialize encoder, decoder input and target data.
    en_in_data = [] 
    dec_in_data = [] 
    dec_tr_data = []
    
    # Padding variable with first character as 1 as the rest are 0
    pad_en = [1] + [0] * (len(input_characters) - 1)
    pad_dec = [0] * (len(target_characters)) 
    pad_dec[2] = 1  # Padding for '\t'
    
    # CountVectorizer for one-hot encoding
    cv = CountVectorizer(binary=True, tokenizer=lambda txt: txt.split(), stop_words=None, analyzer='char')
    
    for i, (input_t, target_t) in enumerate(zip(input_texts, target_texts)):
        # Fit the input characters into the CountVectorizer
        cv_inp = cv.fit(input_characters)
        
        # Transform the input text from the help of CountVectorizer fit
        en_in_data.append(cv_inp.transform(list(input_t)).toarray().tolist())
        
        cv_tar = cv.fit(target_characters)
        dec_in_data.append(cv_tar.transform(list(target_t)).toarray().tolist())
        
        # Decoder target will be one timestep ahead (ignores first character '\t')
        dec_tr_data.append(cv_tar.transform(list(target_t)[1:]).toarray().tolist())
        
        # Add padding if the length of the input or target text is smaller
        # than their respective maximum input or target length.
        if len(input_t) < max_input_length:
            for _ in range(max_input_length - len(input_t)):
                en_in_data[i].append(pad_en)
        if len(target_t) < max_target_length:
            for _ in range(max_target_length - len(target_t)):
                dec_in_data[i].append(pad_dec)
        if (len(target_t) - 1) < max_target_length:
            for _ in range(max_target_length - len(target_t) + 1):
                dec_tr_data[i].append(pad_dec)
    
    # Convert list to numpy array with dtype float32
    en_in_data = np.array(en_in_data, dtype="float32")
    dec_in_data = np.array(dec_in_data, dtype="float32")
    dec_tr_data = np.array(dec_tr_data, dtype="float32")

    return en_in_data, dec_in_data, dec_tr_data

# Create input object of total number of encoder characters
en_inputs = Input(shape=(None, num_en_chars))

# Create LSTM with the hidden dimension of 256
encoder = LSTM(256, return_state=True)

# Discard encoder output and store hidden and cell state
en_outputs, state_h, state_c = encoder(en_inputs)
en_states = [state_h, state_c]

# Create input object of total number of decoder characters
dec_inputs = Input(shape=(None, num_dec_chars))

# Create LSTM with the hidden dimension of 256
dec_lstm = LSTM(256, return_sequences=True, return_state=True)

# Initialize the decoder model with the states from encoder
dec_outputs, _, _ = dec_lstm(dec_inputs, initial_state=en_states)

# Output layer with the shape of total number of decoder characters
dec_dense = Dense(num_dec_chars, activation="softmax")
dec_outputs = dec_dense(dec_outputs)

# Create Model and store all variables 
model = Model([en_inputs, dec_inputs], dec_outputs)

# Save necessary data for later use
pickle.dump({
    'input_characters': input_characters,
    'target_characters': target_characters,
    'max_input_length': max_input_length,
    'max_target_length': max_target_length,
    'num_en_chars': num_en_chars,
    'num_dec_chars': num_dec_chars
}, open("/Users/Naresh Chandra/OneDrive/Desktop/language translation eng-fr/language-translator-ml-codes/training_data.pkl", "wb"))

# Load the data and train the model
en_in_data, dec_in_data, dec_tr_data = bagofcharacters(input_texts, target_texts)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit([en_in_data, dec_in_data], dec_tr_data, batch_size=64, epochs=200, validation_split=0.2)

# Save model
model.save("s2s.model.keras")

# Summary and model plot
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

