# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
import re
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    max_features = 10000  # Vocabulary size used during training
    
    # Clean and tokenize text (remove punctuation, convert to lowercase)
    text = text.lower()
    # Replace punctuation with spaces, then split
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    words = text.split()
    
    encoded_review = []
    for word in words:
        if not word:  # Skip empty strings
            continue
        # Get word index from dictionary (1-indexed, where 1 is most frequent)
        idx = word_index.get(word, 2)  # 2 is OOV token in word_index
        
        # IMDB load_data(num_words=10000) keeps top 10000 words and adds 3
        # Embedding(input_dim=10000) expects indices in [0, 9999]
        # So we cap word_index to ensure (idx + 3) <= 9999
        # Therefore: idx <= 9996
        if idx > max_features - 4:  # Cap at 9996
            idx = 2  # Use OOV token for words outside vocabulary
        
        # Add 3 to match IMDB's encoding (reserves 0,1,2,3 for special tokens)
        encoded_review.append(idx + 3)
    
    # Pad sequences to maxlen=500 (same as training)
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


import streamlit as st
## streamlit app
# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):

    preprocessed_input=preprocess_text(user_input)

    ## MAke prediction
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')

