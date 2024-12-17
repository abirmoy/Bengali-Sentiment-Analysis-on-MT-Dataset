import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score, precision_score
from preprocessor_function import*

# Load the data
Kaggle_bengali_dataset = './Data/predicted_unsupervised_sentiment.xlsx'
IMDB_EN_BN_GGL_translation = './Data/IMDB_BN_Row.xlsx'
EN_to_HN_IMDB_GGL_translation = './Data/EN_to_HN_IMDB_GGL_translation.xlsx'
Hindi_Amazon_Review = './Data/Hindi_Amazon_Review.xlsx'




file_location = EN_to_HN_IMDB_GGL_translation
tokenizer_model_name = 'EN_to_HN_IMDB_GGL_translation'



vectorizer_path = f'./output/Saved Model/RF_{tokenizer_model_name}_tfidf_vectorizer.pkl'
model_path = f'./output/Saved Model/RF_{tokenizer_model_name}_model.pkl'





data = pd.read_excel(file_location)
# Apply preprocessing to the 'sentence' column
data['sentence'] = data['sentence'].apply(preprocessing)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['sentence'], data['sentiment'], test_size=0.35, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Save the vectorizer for future use
# vectorizer_path = r'G:\Desktop\Playground\IMDB\output\test model\RFWord_tfidf_vectorizer.pkl'
joblib.dump(vectorizer, vectorizer_path)

# Initialize the RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model to the training data
clf.fit(X_train, y_train)

# Save the trained model
# model_path = r'G:\Desktop\Playground\IMDB\output\test model\rfword_model.pkl'
joblib.dump(clf, model_path)

# Make predictions on the test set
y_pred = clf.predict(X_test)




# Print metrics
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
print("\nAccuracy: {:.4f}".format(accuracy))
print("F1 Score: {:.4f}".format(f1))
print("Recall: {:.4f}".format(recall))
print("Precision: {:.4f}".format(precision))
# Print the classification report
print(classification_report(y_test, y_pred))

# Print the confusion matrix
print('Confusion Matrix:',confusion_matrix(y_test, y_pred))