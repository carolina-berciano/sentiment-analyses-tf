# import libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd

# loading data in csv file
path = tf.keras.utils.get_file('reviews.csv',
                               'https://drive.google.com/uc?id=13ySLC_ue6Umt9RJYSeM2t-V0kCv-4C-P')
print('path', path)

# Read the csv file
dataset = pd.read_csv(path)

# Review the first few entries in the dataset
print('first five reviews', dataset.head())

# get the reviews from the text column and labels from sentiment column
sentences = dataset['text'].tolist()
labels = dataset['sentiment'].tolist()

# Separate out the sentences and labels into training and test sets
training_size = int(len(sentences) * 0.8)

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

# Make labels into numpy arrays for use with the network later
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

vocab_size = 500
embedding_dim = 16
max_length = 50
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"


####################################
######### DATA PREPARATION #########
####################################

# create tokenizer
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)  # num_words: maxNumber of words to tokenize (vocabulary size)
# (OOV) token represents words that are not in the index

# tokenize the words
tokenizer.fit_on_texts(training_sentences)  # tokenize the text, in other words, generate numbers for the words

# examine the word index
word_index = tokenizer.word_index
print(word_index)
print(word_index['favorite'])  # get the number for a given word

# generate sequences for the training sentences
training_sequences = tokenizer.texts_to_sequences(training_sentences)
# Make sequences all the same length
# if maxlen not passed, the length of the longest sequence is taken
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(training_padded)

# generate sequences for the testing sentences
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
# Make sequences all the same length
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

####################################
######### CREATE MODEL #########
####################################

# define model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(48, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# compile model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

# train model
num_epochs = 100
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "nlp-sentiment.h5", save_best_only=True)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=20)
history = model.fit(training_padded, training_labels_final, epochs=num_epochs, callbacks=[early_stopping, model_checkpoint],
                    validation_data=(testing_padded, testing_labels_final))