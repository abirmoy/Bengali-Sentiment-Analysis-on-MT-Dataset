import pandas as pd
import re
import pickle
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score

# Preprocessing functions
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
                               u"\u2000-\u206F"          # generalPunctuations
                               "]+", flags=re.UNICODE)
    english_pattern = re.compile('[a-zA-Z0-9]+', flags=re.I)
    text = emoji_pattern.sub(r'', text)
    text = english_pattern.sub(r'', text)
    return text

def remove_punctuations(my_str):
    punctuations = '''````£|¢|Ñ+-*/=EROero৳০১২৩৪৫৬৭৮৯012–34567•89।!()-[]{};:'"“\’,<>./?@#$%^&*_~‘—॥”‰⚽️✌�￰৷￰'''
    no_punct = ""
    for char in my_str:
        if char not in punctuations:
            no_punct = no_punct + char
    return no_punct

def joining(text):
    out = ' '.join(text)
    return out

def preprocessing(text):
    out = remove_punctuations(replace_strings(text))
    return out






Kaggle_bengali_dataset = './Data/Kaggle_bengali_dataset.xlsx'
IMDB_EN_BN_GGL_translation = './Data/EN_to_HN_IMDB_GGL_translation.xlsx'
EN_to_HN_IMDB_GGL_translation = './Data/EN_to_HN_IMDB_GGL_translation.xlsx'
Hindi_Amazon_Review = './Data/Hindi_Amazon_Review.xlsx'


tokenizer_model_name = 'Hindi_Amazon_Review'
location_of_testdata = EN_to_HN_IMDB_GGL_translation
output_file_name = 'EN_to_HN_IMDB_GGL_translation'

tokenizer_path = f'./output/Saved Model/MNB_{tokenizer_model_name}_tokenizer.pkl'
model_path = f'./output/Saved Model/MNB_{tokenizer_model_name}_model.pkl'
output_path = f'./output/Saved Result/MNB_{tokenizer_model_name}_to_{output_file_name}_predictions_output.xlsx'

# Load the tokenizer and model from pickle files
with open(tokenizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load the new dataset
df_new = pd.read_excel(location_of_testdata)

# Apply preprocessing to the 'sentence' column
df_new['sentence'] = df_new['sentence'].apply(preprocessing)
df_new['sentence'] = df_new['sentence'].apply(text_to_word_list)
df_new['sentence'] = df_new['sentence'].apply(joining)

# Extract features and labels
X_new = df_new['sentence'].values
y_new = df_new['sentiment'].values

# Transform the text data using the loaded tokenizer
X_new_transformed = vectorizer.transform(X_new)

# Predict the sentiments for the new data
y_new_pred = model.predict(X_new_transformed)

# Save predictions in the DataFrame
df_new['predicted_sentiment'] = y_new_pred

# Save the DataFrame to an Excel file
df_new.to_excel(output_path, index=False)

# Print analysis report
accuracy = accuracy_score(y_new, y_new_pred)
f1 = f1_score(y_new, y_new_pred, average='weighted')  # Adjust to 'weighted' for multi-class
recall = recall_score(y_new, y_new_pred, average='weighted')
precision = precision_score(y_new, y_new_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_new, y_new_pred))
