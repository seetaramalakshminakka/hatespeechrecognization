import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from sklearn.feature_extraction.text import CountVectorizer  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.tree import DecisionTreeClassifier  # type: ignore
from preprocess import clean  # Import the cleaning function

# Load data
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['labels'] = df['class'].map({0: "Hate Speech Detected", 1: "Offensive language detected", 2: "No hate and offensive speech"})
    df = df[['tweet', 'labels']]
    df['tweet'] = df['tweet'].apply(clean)
    return df

# Train model
def train_model(df):
    x = np.array(df['tweet'])
    y = np.array(df['labels'])
    cv = CountVectorizer()
    x = cv.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    return clf, cv

# Predict function
def predict_hate_speech(clf, cv, test_data):
    test_data_cleaned = clean(test_data)
    test_data_transformed = cv.transform([test_data_cleaned])
    return clf.predict(test_data_transformed)
