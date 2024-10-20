# import libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds

# loading data in csv file
path = tf.keras.utils.get_file('reviews.csv',
                               'https://drive.google.com/uc?id=13ySLC_ue6Umt9RJYSeM2t-V0kCv-4C-P')
print('path', path)

# Read the csv file
dataset = pd.read_csv(path)

# get the reviews from the text column and labels from sentiment column
sentences = dataset['text'].tolist()
labels = dataset['sentiment'].tolist()

vocab_size = 500
embedding_dim = 16
max_length = 50
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

# create subwords datasets from full words datasets
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(sentences, vocab_size, max_subword_length=5)
for i, sentence in enumerate(sentences):
  sentences[i] = tokenizer.encode(sentence)


####################################
######### DATA PREPARATION #########
####################################

# Pad all sentences
sentences_padded = pad_sequences(sentences, maxlen=max_length,
                                 padding=padding_type, truncating=trunc_type)

# Separate out the sentences and labels into training and test sets
training_size = int(len(sentences) * 0.8)

training_sentences = sentences_padded[0:training_size]
testing_sentences = sentences_padded[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

# Make labels into numpy arrays for use with the network later
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

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
history = model.fit(training_sentences, training_labels_final, epochs=num_epochs, callbacks=[early_stopping, model_checkpoint],
                    validation_data=(testing_sentences, testing_labels_final))