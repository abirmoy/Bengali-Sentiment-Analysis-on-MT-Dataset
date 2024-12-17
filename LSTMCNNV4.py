import pandas as pd
from pandas import read_excel
import numpy as np
import re
from re import sub
import multiprocessing
from unidecode import unidecode
import os
from matplotlib import pyplot as plt
from time import time 
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation, Embedding, Flatten, Bidirectional, Conv1D
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras import regularizers
import pickle
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, precision_score
from sklearn.model_selection import train_test_split
from preprocessor_function import*




Kaggle_bengali_dataset = './Data/Kaggle_bengali_dataset.xlsx'
IMDB_EN_BN_GGL_translation = './Data/IMDB_EN_BN_GGL_translation.xlsx'
EN_to_HN_IMDB_GGL_translation = './Data/EN_to_HN_IMDB_GGL_translation.xlsx'
Hindi_Amazon_Review = './Data/Hindi_Amazon_Review.xlsx'


location = Hindi_Amazon_Review
tokenizer_model_name = 'Hindi_Amazon_Review'


tokenizer_path = f'./output/test model/lstmcnn_{tokenizer_model_name}_tokenizer.pickle'
model_path = f'./output/test model/lstmcnn_{tokenizer_model_name}_model.keras'
test_ratio = 0.2





df = pd.read_excel(location)
df['sentence'] = df.sentence.apply(lambda x: preprocessing(str(x)))
df.reset_index(drop=True, inplace=True)

train1, test1 = train_test_split(df, random_state=69, test_size=test_ratio)
training_sentences = []
testing_sentences = []

train_sentences = train1['sentence'].values
train_labels = train1['sentiment'].values
for i in range(train_sentences.shape[0]): 
    x = str(train_sentences[i])
    training_sentences.append(x)
    
training_sentences = np.array(training_sentences)

test_sentences = test1['sentence'].values
test_labels = test1['sentiment'].values
for i in range(test_sentences.shape[0]): 
    x = str(test_sentences[i])
    testing_sentences.append(x)
    
testing_sentences = np.array(testing_sentences)

train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)

print("Training Set Length: " + str(len(train1)))
print("Testing Set Length: " + str(len(test1)))
print("training_sentences shape: " + str(training_sentences.shape))
print("testing_sentences shape: " + str(testing_sentences.shape))
print("train_labels shape: " + str(train_labels.shape))
print("test_labels shape: " + str(test_labels.shape))

vocab_size = 45000 # Updated vocab size
embedding_dim = 300
max_length = 100
trunc_type = 'post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

# Save the tokenizer
with open(tokenizer_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(len(word_index))
print("Word index length: " + str(len(tokenizer.word_index)))

sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

test_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(test_sequences, maxlen=max_length)

print("Sentence :--> \n")
print(training_sentences[2] + "\n")
print("Sentence Tokenized and Converted into Sequence :--> \n")
print(str(sequences[2]) + "\n")
print("After Padding the Sequence with padding length 100 :--> \n")
print(padded[2])

print("Padded shape(training): " + str(padded.shape))
print("Padded shape(testing): " + str(testing_padded.shape))

with tf.device('/gpu:0'):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
    model.add(Conv1D(200, kernel_size=3, activation="relu"))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, kernel_regularizer=regularizers.l2(0.01), activation="relu"))
    model.add(Dense(2, activation='softmax'))

    adam = Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    history = model.fit(padded, train_labels, epochs=5, batch_size=256, validation_data=(testing_padded, test_labels), use_multiprocessing=True, workers=8)

    print(history.history.keys())
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val_loss'])
    plt.show()

    accuracy = history.history['accuracy']
    val_accuracy= history.history['val_accuracy']
    plt.plot(accuracy)
    plt.plot(val_accuracy)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'val_accuracy'])
    plt.show()
# Save the model
model.save(model_path)

# Predict on the test set
test_predictions = model.predict(testing_padded)
test_predictions = np.argmax(test_predictions, axis=1)
test_labels = np.argmax(test_labels, axis=1)

# Print metrics
accuracy = accuracy_score(test_labels, test_predictions)
recall = recall_score(test_labels, test_predictions, average='weighted')
f1 = f1_score(test_labels, test_predictions, average='weighted')
precision = precision_score(test_labels, test_predictions, average='weighted')
print("\nAccuracy: {:.4f}".format(accuracy))
print("F1 Score: {:.4f}".format(f1))
print("Recall: {:.4f}".format(recall))
print("Precision: {:.4f}".format(precision))
print("\nClassification Report:\n", classification_report(test_labels, test_predictions))
