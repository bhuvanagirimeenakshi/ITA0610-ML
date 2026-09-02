from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Custom Dataset
# Study Hours vs Marks

X = [
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10]
]

y = [
    35,
    40,
    48,
    55,
    62,
    70,
    78,
    85,
    92,
    98
]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create Linear Regression Model
model = LinearRegression()

# Train Model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Display Results
print("Actual Marks:")
print(y_test)

print("\nPredicted Marks:")
print(y_pred)

# Model Parameters
print("\nSlope =", model.coef_[0])
print("Intercept =", model.intercept_)

# Performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error =", round(mse,2))
print("R2 Score =", round(r2,2))

# Predict for a New Student
hours = [[7.5]]
prediction = model.predict(hours)

print("\nPredicted Marks for 7.5 Study Hours =", round(prediction[0],2))
