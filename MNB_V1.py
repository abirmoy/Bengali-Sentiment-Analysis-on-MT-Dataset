import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
import pickle
import re
import os

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
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\u00C0-\u017F"          # latin
                               u"\u2000-\u206F"          # generalPunctuations
                               "]+", flags=re.UNICODE)
    english_pattern = re.compile('[a-zA-Z0-9]+', flags=re.I)
    text = emoji_pattern.sub(r'', text)
    text = english_pattern.sub(r'', text)
    return text

def remove_punctuations(my_str):
    punctuations = '''```` £|¢| Ñ+-*/=EROero₹৳০১২৩৪৫৬৭৮৯०१२३४५६७८९१012–34567•89।!()-[]{};:'"“\’,<>./?@#$%^&*_~‘—॥”‰⚽️✌ ￰৷￰'''
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

file_path = EN_to_HN_IMDB_GGL_translation
tokenizer_model_name = 'EN_to_HN_IMDB_GGL_translation'



tokenizer_path = f'./output/Saved Model/MNB_{tokenizer_model_name}_tokenizer.pkl'
model_path = f'./output/Saved Model/MNB_{tokenizer_model_name}_model.pkl'




# Load the dataset
df = pd.read_excel(file_path)
# Apply preprocessing to the 'sentence' column
df['sentence'] = df['sentence'].apply(preprocessing)
df['sentence'] = df['sentence'].apply(text_to_word_list)
df['sentence'] = df['sentence'].apply(joining)

# Extract features and labels
X = df['sentence'].values
y = df['sentiment'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize CountVectorizer for tokenizing and transforming the text data
vectorizer = CountVectorizer()

# Transform the text data into numerical data
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

# Save the tokenizer (vectorizer) as a pickle file
with open(tokenizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)

# Initialize the Naive Bayes model
model = MultinomialNB()

# Train the model
model.fit(X_train_transformed, y_train)

# Save the trained model as a pickle file
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

# Predict the sentiments for the test data
y_pred = model.predict(X_test_transformed)

# Print analysis report
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
