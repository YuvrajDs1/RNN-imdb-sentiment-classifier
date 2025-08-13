import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# load model
word_index = imdb.get_word_index()
reverse_word_index = {index: word for word, index in word_index.items()}

# load pretrained model with ReLU activation
model = load_model('simple_RNN_imdb.h5')

# helper function
# function to decode reviews
def decode_review(encoded_review):
  return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])

# function to preprocess user input
def preprocess_text(text):
  words = text.lower().split()
  encoded_review = [word_index.get(word,2) + 3 for word in words]
  padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
  return padded_review

# prediction function
def predict_sentiment(review):
  preprocessed_input = preprocess_text(review)
  prediction = model.predict(preprocessed_input)

  sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
  return sentiment, prediction[0][0]

# Streamlit code

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review for classifying it as positive or negative review')

# user input
user_input = st.text_input('Movie Review')

if st.button('Classify'):
  preprocessed_input = preprocess_text(user_input)

  prediction = model.predict(preprocessed_input)
  sentiment = 'Positive' if prediction > 0.5 else 'Negative'

  st.write('Sentiment:', sentiment)
  st.write('Prediction Score:', prediction[0][0])
else:
  st.write('Please enter a movie review')