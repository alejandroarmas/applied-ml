import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from utils.regression import least_squares
from utils.correlation import pearson_corr_coef
from utils.mse import mse
from utils.gradient_descent import gradient_descent
from utils.datasets import LinearRegressionDataset, PolynomialRegressionDataset, MultipleRegressionDataset 

# gen = LinearRegressionDataset()

# dataset = gen.generate()


# X = dataset['X']['train']
# Y = dataset['Y']['train']

# slope, intercept = least_squares(X, Y)
# X = dataset['X']['test']
# Y = dataset['Y']['test']

# predict_y = X * slope + intercept

# coefficient = pearson_corr_coef(Y, predict_y)
# error = mse(Y, predict_y)


# line = f'Regression line using Least Squares: y={intercept:.2f}+{slope:.2f}x, r={coefficient:.2f}, mse={error:.2f}'

# fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(9, 4)) # two axes on figure
# fig1.set_dpi(300)
# ax1.scatter(X, Y)
# ax1.plot(X, predict_y, 'b')
# ax1.set_ylabel('Y')
# ax1.set_title(line)


# predict_slope, predict_intercept, costs = gradient_descent(X, Y)

# X = dataset['X']['test']
# Y = dataset['Y']['test']
# predict_y = X * predict_slope + predict_intercept

# print(f'Initial Cost: {costs[0]}')
# print(f'Final Cost: {costs[-1]}')

# ax2.scatter(X, Y)
# ax2.set_xlabel('X')
# ax2.set_ylabel('Y')
# ax2.plot(X, predict_y)
# coefficient = pearson_corr_coef(Y, predict_y)
# line = f'Regression line using Gradient Descent: y={predict_intercept:.2f}+{predict_slope:.2f}x, r={coefficient:.2f}, mse={costs[-1]:.2f}'
# ax2.set_title(line)
# plt.show()


# Problem Polynomial Regression

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression

# degree = 3

# gen = PolynomialRegressionDataset('data/polydata-1.csv', 'x', 'y')
# dataset = gen.generate()

# X = dataset['X']['train']
# Y = dataset['Y']['train']

# poly = PolynomialFeatures(degree=degree, include_bias=False)
# poly_features = poly.fit_transform(X.reshape(-1, 1))

# poly_reg_model = LinearRegression()
# poly_reg_model.fit(poly_features, Y)

# y_predicted = poly_reg_model.predict(poly_features)


# plt.scatter(X, Y)
# plt.plot(X, y_predicted, color='purple')
# plt.show()


# Problem Multiple Regression

data_gen = MultipleRegressionDataset()
dataset = data_gen.generate()

X_train = dataset['X']['train']
y_train = dataset['Y']['train']




# print(data['DESCR'])
# print(f'Size of the dataset: {len(data["data"])=}')
# print(f'Feature names of dataset: {data["feature_names"]=}')
# print(f'Target name of dataset: {data["target_names"]=}')

import matplotlib.pyplot as plt

plt.scatter(X_train[:, 0], y_train)
plt.xlabel("Feature 1")
plt.ylabel("Target")
plt.show()


plt.hist(X_train[:, 0], bins=30)
plt.xlabel("Feature 1")
plt.ylabel("Frequency")
plt.show()

# import seaborn as sns

# sns.heatmap(data.corr())
# plt.show()
