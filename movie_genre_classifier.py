import zipfile
import pandas as pd
import nltk
import re
import ast
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

# Download stopwords
nltk.download('stopwords')

# Function to clean text
def clean_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = text.lower()
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Function to predict genre from custom plot
def predict_genres(plot, vectorizer, model, mlb):
    cleaned = clean_text(plot)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    return mlb.inverse_transform(prediction)[0]

def main():
    # Load dataset from zip
    zip_path = 'C:/Users/HP/OneDrive/Codsoft/MOVIE GENRE CLASSIFICATION/archive.zip'
    csv_filename = 'Top 100 IMDB Movies.csv'
    extract_path = 'C:/Users/HP/OneDrive/Codsoft/MOVIE GENRE CLASSIFICATION/'

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extract(csv_filename, path=extract_path)

    df = pd.read_csv(extract_path + csv_filename)

    print(df.head())
    print("\nColumns in dataset:")
    print(df.columns)

    # Clean text
    df['description'] = df['description'].apply(clean_text)

    # Convert genre column to list
    df['genre'] = df['genre'].apply(lambda g: ast.literal_eval(g) if isinstance(g, str) and g.startswith("[") else [g])

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(df['description'])

    # Encode genres
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['genre'])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = MultiOutputClassifier(LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=mlb.classes_))

    # Test with a custom plot
    test_plot = "A young wizard joins a magical school and battles dark forces."
    predicted = predict_genres(test_plot, vectorizer, model, mlb)
    print("\nTest Plot:", test_plot)
    print("Predicted Genres:", predicted)

    # Show some sample predictions from dataset
    print("\nSample Predictions:")
    for i in range(5):
        desc = df['description'].iloc[i]
        actual_genres = df['genre'].iloc[i]
        predicted_genres = predict_genres(desc, vectorizer, model, mlb)
        print(f"\nMovie {i+1}")
        print("Actual Genres   :", actual_genres)
        print("Predicted Genres:", predicted_genres)

if __name__ == "__main__":
    main()
