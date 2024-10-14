import streamlit as st
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle

# Load the trained model
model = keras.models.load_model('chat_model.keras')

# Load the fitted tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the fitted label encoder
with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

# Load the intents
with open('intents.json') as file:
    data = json.load(file)

# Streamlit app
def main():
    st.title("Chatbot")

    user_input = st.text_input("You: ")

    if user_input:
        # Preprocess the input
        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([user_input]),
                                                                          truncating='post', maxlen=20))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                st.text_area("Bot:", value=np.random.choice(i['responses']), height=100)

if __name__ == "__main__":
    main()
