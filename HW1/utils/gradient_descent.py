import numpy as np
from utils.mse import mse

def gradient_descent(X, Y, learning_rate: float =0.01, num_iterations: int=1000, threshold: float=1e-6):
    n = len(X)
    assert n == len(Y)

    predict_slope = 1
    predict_intercept = 0
    costs = []
    for epoch in range(num_iterations):
        predict_y = X * predict_slope + predict_intercept
        cost = mse(predict_y, Y)
        costs.append(cost)

        slope_grad = 2 * np.sum((predict_y - Y) * X) / n
        intercept_grad = 2 * np.sum(predict_y - Y) / n

        predict_slope -= learning_rate * slope_grad
        predict_intercept -= learning_rate * intercept_grad

        if epoch > 0 and np.abs(cost - costs[-2]) < threshold:
            break

    return predict_slope, predict_intercept, costs