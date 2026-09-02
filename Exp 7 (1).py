from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
# Load Digits Dataset
digits = load_digits()
# Features and Target
X = digits.data
y = digits.target
# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# Create Logistic Regression Model
model = LogisticRegression(max_iter=5000)
# Train Model
model.fit(X_train, y_train)
# Predict Test Data
y_pred = model.predict(X_test)
# Display Actual Labels
print("Actual Labels:")
print(y_test)
# Display Predicted Labels
print("\nPredicted Labels:")
print(y_pred)
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy = {:.2f}%".format(accuracy * 100))
