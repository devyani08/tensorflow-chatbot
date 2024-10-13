# import streamlit as st
# import json
# import numpy as np
# from tensorflow import keras
# from sklearn.preprocessing import LabelEncoder
# import pickle
# import random
# import tempfile

# # File upload function (uses temporary directory)
# def save_uploaded_file(uploaded_file):
#     if uploaded_file is not None:
#         temp_dir = tempfile.mkdtemp()  # Create a temporary directory
#         file_path = f"{temp_dir}/{uploaded_file.name}"
#         with open(file_path, 'wb') as f:
#             f.write(uploaded_file.read())
#         return file_path
#     return None

# # Load intents.json directly from the same directory
# try:
#     with open("intents.json") as file:
#         data = json.load(file)
# except Exception as e:
#     st.error(f"Error loading intents.json: {e}")

# # Streamlit file uploaders for other files
# st.sidebar.header("Upload Model Files")
# model_file = st.sidebar.file_uploader("Upload chat_model.keras", type="keras")
# tokenizer_file = st.sidebar.file_uploader("Upload tokenizer.pickle", type="pickle")
# label_encoder_file = st.sidebar.file_uploader("Upload label_encoder.pickle", type="pickle")

# # Check if all files are uploaded
# if model_file and tokenizer_file and label_encoder_file:
#     # Save uploaded files to temporary directories
#     model_path = save_uploaded_file(model_file)
#     tokenizer_path = save_uploaded_file(tokenizer_file)
#     label_encoder_path = save_uploaded_file(label_encoder_file)

#     try:
#         # Load the trained model with a spinner for user experience
#         @st.cache_resource
#         def load_model():
#             with st.spinner("Loading model..."):
#                 return keras.models.load_model(model_path)

#         model = load_model()

#         # Load tokenizer object
#         @st.cache_resource
#         def load_tokenizer():
#             with open(tokenizer_path, 'rb') as handle:
#                 return pickle.load(handle)

#         tokenizer = load_tokenizer()

#         # Load label encoder object
#         @st.cache_resource
#         def load_label_encoder():
#             with open(label_encoder_path, 'rb') as enc:
#                 return pickle.load(enc)

#         lbl_encoder = load_label_encoder()

#         # Parameters
#         max_len = 20
#         confidence_threshold = 0.5  # Set a confidence threshold

#         st.title("Chatbot")

#         # Initialize chat history
#         if "messages" not in st.session_state:
#             st.session_state.messages = []

#         # Display chat messages from history on app rerun
#         for message in st.session_state.messages:
#             with st.chat_message(message["role"]):
#                 st.markdown(message["content"])

#         # React to user input
#         if prompt := st.chat_input("What is your question?"):
#             # Display user message in chat message container
#             st.chat_message("user").markdown(prompt)
#             # Add user message to chat history
#             st.session_state.messages.append({"role": "user", "content": prompt})

#             # Get bot response
#             try:
#                 result = model.predict(keras.preprocessing.sequence.pad_sequences(
#                     tokenizer.texts_to_sequences([prompt]), truncating='post', maxlen=max_len))

#                 # Check if the model's prediction confidence exceeds the threshold
#                 if np.max(result) > confidence_threshold:
#                     tag = lbl_encoder.inverse_transform([np.argmax(result)])
#                     # Find response corresponding to the tag
#                     for intent in data['intents']:
#                         if intent['tag'] == tag:
#                             response = np.random.choice(intent['responses'])
#                             break
#                 else:
#                     response = "I'm not sure I understand. Could you rephrase that?"

#             except Exception as e:
#                 response = f"Error occurred during prediction: {e}"

#             # Display assistant response in chat message container
#             with st.chat_message("assistant"):
#                 st.markdown(response)
#             # Add assistant response to chat history
#             st.session_state.messages.append({"role": "assistant", "content": response})

#     except Exception as e:
#         st.error(f"Error: {e}")
# else:
#     st.warning("Please upload all required model files.")

import streamlit as st
import json
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import pickle
import random

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

