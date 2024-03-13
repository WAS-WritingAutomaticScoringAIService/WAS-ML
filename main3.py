from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Load data
train_data = pd.read_csv('train_data_num.csv')  # Assuming the training data is in CSV format with columns: student_id, essay, grade
test_data = pd.read_csv('test_data_withoutgrade.csv')    # Assuming the test data is in CSV format with columns: student_id, essay

# Tokenize essays using KoNLPy (Okt)
okt = Okt()

def tokenize(text):
    if isinstance(text, str):
        tokens = okt.nouns(text)  # Tokenize the text if it's a string
    else:
        tokens = []  # Return an empty list if the input is not a string
    return tokens

# Drop rows with missing target values (NaN)
train_data.dropna(subset=['grade'], inplace=True)

# Preprocess and tokenize essays
train_data['tokenized_essay'] = train_data['essay'].apply(tokenize)
test_data['tokenized_essay'] = test_data['essay'].apply(tokenize)

# Convert tokenized essays into strings
train_data['tokenized_essay'] = train_data['tokenized_essay'].apply(lambda x: ' '.join(x))
test_data['tokenized_essay'] = test_data['tokenized_essay'].apply(lambda x: ' '.join(x))

# Separate features (X) and target variable (y)
X = train_data['tokenized_essay']
y = train_data['grade']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# Train a regression model (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred_train = model.predict(X_train_tfidf)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
print("Training RMSE:", train_rmse)

y_pred_val = model.predict(X_val_tfidf)
val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
print("Validation RMSE:", val_rmse)

# Predict grades for test data
X_test_tfidf = vectorizer.transform(test_data['tokenized_essay'])
test_data['predicted_grade'] = model.predict(X_test_tfidf)

# Save predictions to a CSV file
test_data[['StudentID', 'predicted_grade']].to_csv('predicted_grades.csv', index=False)
