import numpy as np

def least_squares(x, y):
    """
    Calculates the coefficients of the fitted line using the Least Squares method.
    
    Parameters:
    x (array-like): Independent variable.
    y (array-like): Dependent variable.
    
    Returns:
    tuple: Coefficients of the fitted line (slope, intercept).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x*y)
    sum_x_squared = np.sum(x**2)
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept
