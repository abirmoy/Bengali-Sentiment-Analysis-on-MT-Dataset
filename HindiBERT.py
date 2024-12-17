import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TFAutoModelForSequenceClassification, AutoModel,TFAutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
import pickle
import re

# PREPROCESSING FUNCTIONS
def text_to_word_list(text):
    text = text.split()
    return text

def replace_strings(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\u00C0-\u017F"          # latin
                               u"\u2000-\u206F"          # general punctuations
                               "]+", flags=re.UNICODE)
    english_pattern = re.compile('[a-zA-Z0-9]+', flags=re.I)
    
    text = emoji_pattern.sub(r'', text)
    text = english_pattern.sub(r'', text)

    return text

def remove_punctuations(my_str):
    punctuations = '''````£|¢|Ñ+-*/=EROero৳₹০১২৩৪৫৬৭৮৯012–34567•89।!()-[]{};:'"“\’,<>./?@#$%^&*_~‘—॥”‰⚽️✌�￰৷￰'''
    
    no_punct = ""
    for char in my_str:
        if char not in punctuations:
            no_punct += char

    return no_punct

def joining(text):
    out = ' '.join(text)
    return out

def preprocessing(text):
    out = remove_punctuations(replace_strings(text))
    return out

def load_and_preprocess_data(file_path):
    df = pd.read_excel(file_path)
    df['sentence'] = df['sentence'].apply(lambda x: preprocessing(str(x)))
    return df

# 2. Preprocess the Data
def preprocess_data(tokenizer, sentences, max_length):
    return tokenizer(
        text=list(sentences),
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors="tf"
    )

# 3. Build the Model using BanglaBERT
def build_model(bert_model, max_length):
    input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")

    bert_output = bert_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    dropout = tf.keras.layers.Dropout(0.3)(bert_output)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dropout[:, 0, :])

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 4. Train and Evaluate the Model
def train_and_evaluate(data_file, tokenizer_path, model_path, max_length=128, batch_size=32, epochs=5):
    data = load_and_preprocess_data(data_file)

    tokenizer = AutoTokenizer.from_pretrained("monsoon-nlp/hindi-bert")
    sentences = data['sentence'].values
    labels = data['sentiment'].values

    # Preprocess the data
    processed_data = preprocess_data(tokenizer, sentences, max_length)

    # Convert TensorFlow tensors to numpy arrays
    input_ids = processed_data['input_ids'].numpy()
    attention_mask = processed_data['attention_mask'].numpy()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(input_ids, labels, test_size=0.2, random_state=42)
    X_train_mask, X_test_mask = train_test_split(attention_mask, test_size=0.2, random_state=42)

    # Load the TensorFlow version of BanglaBERT model
    bert_model = TFAutoModel.from_pretrained("monsoon-nlp/hindi-bert")

    # Build and train the model
    model = build_model(bert_model, max_length)
    history = model.fit([X_train, X_train_mask], y_train, validation_data=([X_test, X_test_mask], y_test), epochs=epochs, batch_size=batch_size)

    # Save the tokenizer
    with open(tokenizer_path, 'wb') as file:
        pickle.dump(tokenizer, file)

    # Save the model in .keras format
    model.save(model_path, save_format='keras')

    # Evaluate the model
    y_pred = (model.predict([X_test, X_test_mask]) > 0.5).astype("int32")

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Load Model and Perform Sentiment Analysis on New Data
def load_model_and_predict(test_file, tokenizer_path, model_path, output_file, max_length=128):
    data = load_and_preprocess_data(test_file)

    with open(tokenizer_path, 'rb') as file:
        tokenizer = pickle.load(file)

    model = tf.keras.models.load_model(model_path)

    processed_data = preprocess_data(tokenizer, data['sentence'].values, max_length)

    predictions = (model.predict([processed_data['input_ids'], processed_data['attention_mask']]) > 0.5).astype("int32")

    data['predicted_sentiment'] = predictions

    # Save predictions to an Excel file
    data.to_excel(output_file, index=False)

    y_true = data['sentiment'].values
    y_pred = predictions

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

# Convert PyTorch BanglaBERT model to TensorFlow
def convert_pytorch_to_tf(pytorch_model_name, tf_model_name):
    # Load the PyTorch model
    model = AutoModelForSequenceClassification.from_pretrained(pytorch_model_name)
    
    # Convert it to TensorFlow
    tf_model = TFAutoModelForSequenceClassification.from_pretrained(pytorch_model_name, from_pt=True)
    
    # Save the TensorFlow model
    tf_model.save_pretrained(tf_model_name)

# Convert model (run this once to convert and save the model)
convert_pytorch_to_tf("monsoon-nlp/hindi-bert", "monsoon-nlp/hindi-bert-tf")

# File paths
EN_to_HN_IMDB_GGL_translation = './Data/EN_to_HN_IMDB_GGL_translation.xlsx'
Hindi_Amazon_Review = './Data/Hindi_Amazon_Review.xlsx'

location = Hindi_Amazon_Review
tokenizer_model_name = 'Hindi_Amazon_Review'
tokenizer_path = f'./output/Saved Model/HindiBERT_{tokenizer_model_name}_tokenizer.pickle'
model_path = f'./output/Saved Model/HindiBERT_{tokenizer_model_name}_model.keras'




# Example usage
train_and_evaluate(location, tokenizer_path=tokenizer_path, model_path=model_path)
