# import json
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.preprocessing import LabelEncoder
# import pickle

# # Load the intents file
# with open('intents.json') as file:
#     data = json.load(file)

# training_sentences = []
# training_labels = []
# labels = []
# responses = {}

# for intent in data['intents']:
#     for pattern in intent['patterns']:
#         training_sentences.append(pattern)
#         training_labels.append(intent['tag'])
#     # Store responses in a dictionary with the tag as the key
#     responses[intent['tag']] = intent['responses']

#     if intent['tag'] not in labels:
#         labels.append(intent['tag'])

# num_classes = len(labels)

# # Encode the labels
# lbl_encoder = LabelEncoder()
# lbl_encoder.fit(training_labels)
# training_labels = lbl_encoder.transform(training_labels)

# # Tokenize the sentences
# vocab_size = 1000
# embedding_dim = 16
# max_len = 20
# oov_token = "<OOV>"

# tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
# tokenizer.fit_on_texts(training_sentences)
# word_index = tokenizer.word_index
# sequences = tokenizer.texts_to_sequences(training_sentences)
# padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

# # Build the model
# model = Sequential()
# model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
# model.add(GlobalAveragePooling1D())
# model.add(Dense(16, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(num_classes, activation='softmax'))

# # Compile the model
# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer='adam', metrics=['accuracy'])

# model.summary()

# # Train the model
# epochs = 500
# history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)

# # Save the model
# model.save("chat_model.keras")

# # Save the tokenizer and label encoder
# with open('tokenizer.pickle', 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('label_encoder.pickle', 'wb') as enc_file:
#     pickle.dump(lbl_encoder, enc_file, protocol=pickle.HIGHEST_PROTOCOL)

# # Save the responses
# with open('responses.pickle', 'wb') as resp_file:
#     pickle.dump(responses, resp_file, protocol=pickle.HIGHEST_PROTOCOL)

import json
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import pickle

# Assuming you have the necessary training code to generate these files

# Sample dummy code to generate and save the model, tokenizer, and label encoder
def save_files():
    # Example for saving a Keras model
    model = keras.Sequential()  # Your model structure
    model.add(keras.layers.Input(shape=(20,)))  # Assuming a placeholder input
    model.add(keras.layers.Dense(10, activation='relu'))
    model.add(keras.layers.Dense(5, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.save("chat_model.keras")

    # Example for saving a tokenizer (use your actual tokenizer logic)
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(["dummy text"])  # Replace with actual data
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Example for saving label encoder
    lbl_encoder = LabelEncoder()
    lbl_encoder.fit(["tag1", "tag2"])  # Replace with actual tags
    with open('label_encoder.pickle', 'wb') as enc_file:
        pickle.dump(lbl_encoder, enc_file)

save_files()

