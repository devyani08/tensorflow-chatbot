import streamlit as st
import json
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import pickle
import random

# File upload functions
def save_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        file_contents = uploaded_file.read()
        with open(uploaded_file.name, 'wb') as f:
            f.write(file_contents)
        return True
    return False

# Streamlit file uploaders
st.sidebar.header("Upload Files")
intents_file = st.sidebar.file_uploader("Upload intents.json", type="json")
model_file = st.sidebar.file_uploader("Upload chat_model.keras", type="keras")
tokenizer_file = st.sidebar.file_uploader("Upload tokenizer.pickle", type="pickle")
label_encoder_file = st.sidebar.file_uploader("Upload label_encoder.pickle", type="pickle")

# Check if all files are uploaded
if intents_file and model_file and tokenizer_file and label_encoder_file:
    # Save uploaded files
    save_uploaded_file(intents_file)
    save_uploaded_file(model_file)
    save_uploaded_file(tokenizer_file)
    save_uploaded_file(label_encoder_file)

    # Load the intents data
    with open("intents.json") as file:
        data = json.load(file)

    # Load the trained model
    @st.cache_resource
    def load_model():
        return keras.models.load_model('chat_model.keras')

    model = load_model()

    # Load tokenizer object
    @st.cache_resource
    def load_tokenizer():
        with open('tokenizer.pickle', 'rb') as handle:
            return pickle.load(handle)

    tokenizer = load_tokenizer()

    # Load label encoder object
    @st.cache_resource
    def load_label_encoder():
        with open('label_encoder.pickle', 'rb') as enc:
            return pickle.load(enc)

    lbl_encoder = load_label_encoder()

    # Parameters
    max_len = 20

    st.title("Chatbot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is your question?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get bot response
        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([prompt]),
                                             truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                response = np.random.choice(i['responses'])

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.warning("Please upload all required files.")
