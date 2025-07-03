import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

file_path = "C:\\Users\\HP\\OneDrive\\Codsoft\\CREDIT CARD FRAUD DETECTION\\creditcard.csv"
df = pd.read_csv(file_path)

X = df.drop('Class', axis=1)
y = df['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

with open("fraud_detection_result.txt", "w") as f:
    f.write("Confusion Matrix:\n")
    f.write(str(conf_matrix))
    f.write("\n\nClassification Report:\n")
    f.write(class_report)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Non-Fraud", "Fraud"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression (Balanced)")
plt.savefig("confusion_matrix.png")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title("Fraud (1) vs Non-Fraud (0) Transactions")
plt.savefig("class_distribution.png")
plt.show()
