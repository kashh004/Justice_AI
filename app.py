import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, render_template

# Load the dataset
file_path = '/Users/akashngowda/Desktop/5th sem mini pro/ipc_sections 2.csv'  # Adjust this path
df = pd.read_csv(file_path)

df = df[['Offense', 'Description']]

# Features and target variables
X = df['Offense'].astype(str).values  # Offense text (features)
y = df['Description'].astype(str).values  # Description (target)

# Encode the target labels (Description)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Use TF-IDF Vectorizer to convert text into numerical features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

# Train a classifier (e.g., Logistic Regression)
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Initialize Flask app
app = Flask(__name__)

# Function to predict description based on offense text
def predict_description(offense_text):
    try:
        offense_tfidf = vectorizer.transform([offense_text])  # Convert input text to tf-idf format
        predicted_label = model.predict(offense_tfidf)
        return label_encoder.inverse_transform(predicted_label)[0]
    except Exception as e:
        return f"An error occurred during prediction: {str(e)}"

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html', prediction=None)

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    offense_input = request.form['offense']  # Get input from the form
    predicted_description = predict_description(offense_input)
    return render_template('index.html', prediction=predicted_description, offense=offense_input)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
