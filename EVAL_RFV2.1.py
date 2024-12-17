import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score, precision_score
from preprocessor_function import*


# File paths
Kaggle_bengali_dataset = './Data/Kaggle_bengali_dataset.xlsx'
IMDB_EN_BN_GGL_translation = './Data/EN_to_HN_IMDB_GGL_translation.xlsx'
EN_to_HN_IMDB_GGL_translation = 'G:/Desktop/Playground/IMDB/input/New folder/Data/EN_to_HN_IMDB_GGL_translation.xlsx'
Hindi_Amazon_Review = 'G:/Desktop/Playground/IMDB/input/New folder/Data/Hindi_Amazon_Review.xlsx'



# Configuration
tokenizer_model_name = 'Hindi_Amazon_Review'
location_of_testdata = EN_to_HN_IMDB_GGL_translation
output_file_name = 'EN_to_HN_IMDB_GGL_translation'

vectorizer_path = f'./output/Saved Model/RF_{tokenizer_model_name}_tfidf_vectorizer.pkl'
model_path = f'./output/Saved Model/RF_{tokenizer_model_name}_model.pkl'
output_file = f'./output/Saved Result/RF_{tokenizer_model_name}_to_{output_file_name}_predictions_output.xlsx'

# Load the pre-trained model
clf = joblib.load(model_path)

# Load the vectorizer that was used during model training
vectorizer = joblib.load(vectorizer_path)

# Load the new data
new_data = pd.read_excel(location_of_testdata)

# Apply preprocessing to the 'sentence' column
new_data['sentence'] = new_data['sentence'].apply(preprocessing)

# Vectorize the preprocessed text using the same vectorizer
X_new = vectorizer.transform(new_data['sentence'])

# Make predictions
new_data['predicted_sentiment'] = clf.predict(X_new)

# Save the predictions to a new Excel file
new_data.to_excel(output_file, index=False)

# Initialize a dictionary to store evaluation metrics
metrics = {}

# If the new data contains true sentiment labels, compare them with predictions
if 'sentiment' in new_data.columns:
    y_true = new_data['sentiment']
    y_pred = new_data['predicted_sentiment']

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')

    # Store metrics in the dictionary
    metrics = {
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Recall": recall,
        "Precision": precision,
        "Classification Report": classification_report(y_true, y_pred, output_dict=True),
        "Confusion Matrix": confusion_matrix(y_true, y_pred).tolist()
    }

    # Print metrics
    print("\nAccuracy: {:.4f}".format(accuracy))
    print("F1 Score: {:.4f}".format(f1))
    print("Recall: {:.4f}".format(recall))
    print("Precision: {:.4f}".format(precision))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

    # Save metrics to an Excel file
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'F1 Score', 'Recall', 'Precision'],
        'Value': [accuracy, f1, recall, precision]
    })

    confusion_matrix_df = pd.DataFrame(
        confusion_matrix(y_true, y_pred),
        index=['True Neg', 'True Pos'], 
        columns=['Pred Neg', 'Pred Pos']
    )

    with pd.ExcelWriter(output_file, mode='a', engine='openpyxl') as writer:
        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
        confusion_matrix_df.to_excel(writer, sheet_name='Confusion Matrix', index=True)
else:
    print("No true sentiment labels found in the new data.")
