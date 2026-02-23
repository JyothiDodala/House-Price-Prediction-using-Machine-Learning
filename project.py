import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load dataset
data = pd.read_csv("train.csv")

print(data.head())
print(data.columns)

# Example (IMPORTANT)
# After running once, check column names and change below

X = data[['GrLivArea', 'BedroomAbvGr']]   # example columns
y = data['SalePrice']

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train model
model = LinearRegression()
model.fit(X_train, y_train)

# accuracy
print("Accuracy:", model.score(X_test, y_test))

predictions = model.predict(X_test)
print("Sample Predictions:", predictions[:5])