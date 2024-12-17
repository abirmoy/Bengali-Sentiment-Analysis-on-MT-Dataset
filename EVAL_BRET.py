import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, TFElectraForSequenceClassification, TFAutoModelForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
import pickle
import re
from preprocessor_function import*

# PREPROCESSING FUNCTIONS
def replace_strings(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               u"\u00C0-\u017F"          # Latin characters with diacritics
                               u"\u2000-\u206F"          # General punctuations
                               "]+", flags=re.UNICODE)
    
    english_pattern = re.compile('[a-zA-Z0-9]+', flags=re.I)
    
    text = emoji_pattern.sub(r'', text)
    text = english_pattern.sub(r'', text)

    return text

def remove_punctuations(my_str):
    punctuations = '''```` £|¢| Ñ+-*/=EROero৳০১২৩৪৫৬৭৮৯012–34567•89।!()-[]{};:'"“\’,<>./?@#$%^&*_~‘—॥”‰⚽️✌ ￰৷￰'''
    
    no_punct = ""
    for char in my_str:
        if char not in punctuations:
            no_punct += char

    return no_punct

def preprocessing(text):
    out = remove_punctuations(replace_strings(text))
    return out

def load_and_preprocess_data(file_path):
    df = pd.read_excel(file_path)
    df['sentence'] = df['sentence'].apply(lambda x: preprocessing(str(x)))
    return df

# Preprocess the Data
def preprocess_data(tokenizer, sentences, max_length):
    return tokenizer(
        text=list(sentences),
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors="tf"
    )


# Load Model and Perform Sentiment Analysis on New Data
def load_model_and_predict(test_file, tokenizer_path, model_path, output_file, max_length=128):
    # Load and preprocess data
    data = load_and_preprocess_data(test_file)

    # Load the tokenizer
    with open(tokenizer_path, 'rb') as file:
        tokenizer = pickle.load(file)

    # Load the model with custom_objects
    custom_objects = {"TFElectraForSequenceClassification": TFElectraForSequenceClassification}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    # Preprocess the data
    processed_data = preprocess_data(tokenizer, data['sentence'].values, max_length)

    # Perform predictions
    predictions = (model.predict([processed_data['input_ids'], processed_data['attention_mask']]) > 0.5).astype("int32")

    # Add predictions to the data and save to Excel
    data['predicted_sentiment'] = predictions
    data.to_excel(output_file, index=False)

    # Print evaluation metrics
    y_true = data['sentiment'].values
    y_pred = predictions
    
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))


# Example usage
Kaggle_bengali_dataset = './Data/Kaggle_bengali_dataset.xlsx'
IMDB_EN_BN_GGL_translation = './Data/EN_to_HN_IMDB_GGL_translation.xlsx'
Human_translation = './Data/Human_translation.xlsx'

location = IMDB_EN_BN_GGL_translation
tokenizer_model_name = 'Kaggle_bengali_dataset'


tokenizer_model_name = 'IMDB_EN_BN_GGL_translation'
output_file_name = 'Kaggle_bengali_dataset'
location_of_testdata = Kaggle_bengali_dataset


tokenizer_path = f'./input/output/Saved Model/BnBERT_{tokenizer_model_name}_tokenizer.pkl'
model_path = f'./input/output/Saved Model/BnBERT_{tokenizer_model_name}_model.keras'
output_file = f'./input/output/Saved Result/BnBERT_{tokenizer_model_name}_to_{output_file_name}_predictions_output.xlsx'


load_model_and_predict(location_of_testdata, tokenizer_path, model_path, output_file)
