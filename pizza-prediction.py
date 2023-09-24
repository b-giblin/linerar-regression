from sklearn.linear_model import LinearRegression
import numpy as np

sizes = np.array([6, 8, 10, 12, 14]).reshape(-1, 1) # size in column format
prices = np.array([7, 9, 11, 13, 15]) # corresponding price

# choose the model
model = LinearRegression()

# train the model
model.fit(sizes, prices)

# make predictions
predicted_price_for_16_inch = model.predict([[16]])
print(predicted_price_for_16_inch)