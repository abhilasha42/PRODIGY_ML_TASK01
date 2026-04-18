import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# loading dataset
data = pd.read_csv("Task1/data/train.csv")
print("Dataset loaded successfully")
print(data.head())

# select imp column
features = data[['OverallQual', 'GarageCars', 'TotalBsmtSF']]
target = data['SalePrice']

# handle missing values
features = features.fillna(features.mean())

# split data for train and test
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2
)

# create nd train model
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training completed")

# predicting
predicted_prices = model.predict(X_test)

# model performance check
score = r2_score(y_test, predicted_prices)
print("Model Accuracy (R2 Score):", score)

# visualize result
plt.scatter(y_test, predicted_prices)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()