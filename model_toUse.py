from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

# Load data
train_data = pd.read_csv('train_data_num.csv')  # Assuming the training data is in CSV format with columns: student_id, essay, grade

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

# Convert tokenized essays into strings
train_data['tokenized_essay'] = train_data['tokenized_essay'].apply(lambda x: ' '.join(x))

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

# Model Selection and Tuning
# Initialize and train a Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train_tfidf, y_train)

# Evaluate the model using cross-validation
cv_scores = cross_val_score(gb_model, X_train_tfidf, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores.mean())

# Example: Predict and categorize grade for a single essay text
essay_text = "대한민국은 아름다운 나라입니다. 우리 모두 함께 더 나은 미래를 위해 노력해야 합니다."

# Tokenize the essay text using KoNLPy
tokenized_essay = tokenize(essay_text)

# Convert tokenized essay into a string
tokenized_essay_str = ' '.join(tokenized_essay)

# Vectorize the tokenized essay using the TF-IDF vectorizer trained on the training data
X_essay_tfidf = vectorizer.transform([tokenized_essay_str])

# Predict grades for the essay text using the Gradient Boosting model
predicted_grade_gb = gb_model.predict(X_essay_tfidf)

# Predict grades for the essay text using the Random Forest model
predicted_grade_rf = model.predict(X_essay_tfidf)

# Ensemble Learning
# Combine predictions from Random Forest and Gradient Boosting models
predicted_grade_ensemble = (predicted_grade_rf + predicted_grade_gb) / 2

# Categorize predicted grades
def categorize_grade(grade):
    if grade == 5:
        return '하'
    elif grade < 10:
        return '중하'
    elif grade == 10:
        return '중'
    elif grade < 15:
        return '중상'
    else:
        return '상'

# Categorize the predicted grade
grade_category = categorize_grade(predicted_grade_ensemble)

# Print or save the grade category
print("Grade Category:", grade_category)