'''
1. The program thakes the tokenizer and saved model to perform sentiment of another excel file
2. it  print analysis report such as Accuracy	F1	Recall	Precision etc using Sklearn 

3. it  save an excel containing the prediction of the given excel file for testing
'''


import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFElectraModel
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

# 8. Load Model and Perform Sentiment Analysis on New Data
def load_model_and_predict(test_file, tokenizer_path, model_path, output_file, max_length=128):
    # Load and preprocess the data
    data = load_and_preprocess_data(test_file)

    # Load the tokenizer
    with open(tokenizer_path, 'rb') as file:
        tokenizer = pickle.load(file)

    # Load the trained model with custom objects
    custom_objects = {'TFElectraModel': TFElectraModel}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    # Preprocess the data
    processed_data = preprocess_data(tokenizer, data['sentence'].values, max_length)

    # Perform predictions
    predictions = (model.predict([processed_data['input_ids'], processed_data['attention_mask']]) > 0.5).astype("int32")

    # Add predictions to the dataframe
    data['predicted_sentiment'] = predictions

    # Save the predictions to an Excel file
    data.to_excel(output_file, index=False)

    # Calculate evaluation metrics
    if 'sentiment' in data.columns:  # Assuming the test file has ground truth labels
        y_true = data['sentiment'].values
        y_pred = predictions

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)

        # Print the evaluation metrics
        print("Accuracy:", accuracy)
        print("F1 Score:", f1)
        print("Recall:", recall)
        print("Precision:", precision)
        print("\nClassification Report:\n", report)

# Example usage
EN_to_HN_IMDB_GGL_translation = './Data/EN_to_HN_IMDB_GGL_translation.xlsx'
Hindi_Amazon_Review = './Data/Hindi_Amazon_Review.xlsx'


tokenizer_model_name = 'Hindi_Amazon_Review'
location = EN_to_HN_IMDB_GGL_translation
output_file_name ='EN_to_HN_IMDB_GGL_translation'

tokenizer_path = f'./output/Saved Model/HindiBERT_{tokenizer_model_name}_tokenizer.pickle'
model_path = f'./output/Saved Model/HindiBERT_{tokenizer_model_name}_model.keras'
output_file = f'./output/Saved Result/HindiBERT_{tokenizer_model_name}_to_{output_file_name}_predictions_output.xlsx'


load_model_and_predict(location, tokenizer_path, model_path, output_file)
