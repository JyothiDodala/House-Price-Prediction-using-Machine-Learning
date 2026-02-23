import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("train.csv")

print("Dataset preview:")
print(data.head())  # see first 5 rows
print("\nColumns:")
print(data.columns)

# Select features and target (change columns if needed)
X = data[['GrLivArea', 'BedroomAbvGr']]  # features
y = data['SalePrice']                     # target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
predictions = model.predict(X_test)

# Print accuracy
print("\nAccuracy:", model.score(X_test, y_test))

# Sample predictions
print("\nSample predictions:", predictions[:5])

# Plot graph
plt.figure(figsize=(8,5))
plt.plot(y_test.values[:20], label="Actual Prices", marker='o')
plt.plot(predictions[:20], label="Predicted Prices", marker='x')
plt.title("Actual vs Predicted House Prices")
plt.xlabel("Samples")
plt.ylabel("Price")
plt.legend()
plt.show()

# Predict custom input
sample_house = [[2500, 3]]  # 2500 sq ft, 3 bedrooms
predicted_price = model.predict(sample_house)
print(f"\nPredicted price for 2500 sq ft, 3 bedrooms: ${predicted_price[0]:.2f}")