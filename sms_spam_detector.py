import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib


# Load dataset
data = pd.read_csv("C:/Users/HP/Downloads/spam.csv", encoding="latin-1")

# Keep only necessary columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

print("Sample data:")
print(data.head())
# Encode 'spam' as 1 and 'ham' as 0
label_encoder = LabelEncoder()
data['label_num'] = label_encoder.fit_transform(data['label'])
# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['message'])  # Features
y = data['label_num']                          # Labels
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)
# Save model and vectorizer to files
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("Model and vectorizer saved to disk!")


# Predict on test set
y_pred = model.predict(X_test)

# Show accuracy and confusion matrix
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# Try it on a custom SMS
def predict_sms(message):
    msg_vector = vectorizer.transform([message])
    prediction = model.predict(msg_vector)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Example usage
sample_sms = "Congratulations! You've won a $1000 Walmart gift card. Click to claim."
print("\nCustom message prediction:")
print(sample_sms, "->", predict_sms(sample_sms))

def predict_sms(message):
    msg_vector = vectorizer.transform([message])
    prediction = model.predict(msg_vector)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Interactive loop for user to test SMS messages
print("\nType your own messages to test the spam detector.")
print("Type 'exit' to quit.")

while True:
    sms = input("Enter an SMS: ")
    if sms.lower() == 'exit':
        print("Goodbye!")
        break
    print("Prediction:", predict_sms(sms))

