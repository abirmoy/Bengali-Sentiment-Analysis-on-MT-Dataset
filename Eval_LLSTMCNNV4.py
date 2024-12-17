import pandas as pd
import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, precision_score
import pickle
from preprocessor_function import*  # Ensure this is implemented and imported correctly

# Function to preprocess text 

# Load tokenizer from file
def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

# Load the pre-trained model
def load_trained_model(model_path):
    model = load_model(model_path)
    return model

# Function to evaluate sentiment on the text data and save predictions
def evaluate_sentiment(excel_file, model_path, tokenizer_path, output_file, max_length=100):
    # Load the Excel file
    df = pd.read_excel(excel_file)

    # Preprocess the text data
    df['sentence'] = df['sentence'].apply(preprocessing)

    # Load the tokenizer and model
    tokenizer = load_tokenizer(tokenizer_path)
    model = load_trained_model(model_path)

    # Tokenize and pad the sequences
    sequences = tokenizer.texts_to_sequences(df['sentence'].values)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, truncating='post')

    # Predict sentiment
    predictions = model.predict(padded_sequences)
    predicted_labels = np.argmax(predictions, axis=1)
    df['predicted_sentiment'] = predicted_labels  # Add predictions to the DataFrame
    
    # If the true labels are available in the Excel file, load them
    if 'sentiment' in df.columns:
        true_labels = df['sentiment'].values
        df['true_sentiment'] = true_labels  # Add true labels to the DataFrame
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        
        # Print the evaluation metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Recall: {recall:.4f}")
        print("Precision: {:.4f}".format(precision))
        
        print("\nClassification Report:\n", classification_report(true_labels, predicted_labels))
    else:
        print("True labels not found in the dataset. Only predictions will be saved.")

    # Save the results to an Excel file
    df.to_excel(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    return df


# Define paths for  model, tokenizer, and input Excel file
Kaggle_bengali_dataset = './Data/Kaggle_bengali_dataset.xlsx'
IMDB_EN_BN_GGL_translation = './Data/IMDB_EN_BN_GGL_translation.xlsx'
EN_to_HN_IMDB_GGL_translation = './Data/EN_to_HN_IMDB_GGL_translation.xlsx'
Hindi_Amazon_Review = './Data/Hindi_Amazon_Review.xlsx'

tokenizer_model_name = 'EN_to_HN_IMDB_GGL_translation'
location_of_testdata = Hindi_Amazon_Review
output_file_name = 'Hindi_Amazon_Review'



tokenizer_path = f'./output/Saved Model/lstmcnn_{tokenizer_model_name}_tokenizer.pickle'
model_path = f'./output/Saved Model/lstmcnn_{tokenizer_model_name}_model.keras'
output_file_path = f'./output/Saved Result/lstmcnn_{tokenizer_model_name}_to_{output_file_name}_predictions.xlsx'

# Evaluate sentiment and save predictions
evaluate_sentiment(location_of_testdata, model_path, tokenizer_path, output_file_path)
