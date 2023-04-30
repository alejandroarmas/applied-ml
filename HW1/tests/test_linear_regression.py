import unittest
from utils.regression import least_squares
from utils.correlation import pearson_corr_coef
from utils.shuffle import unison_shuffled_copies


import numpy as np
import scipy.stats

class TestRegression(unittest.TestCase):

    def test_basic_usage(self):

        x = np.arange(10, 20)
        y = np.array([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])

        result = scipy.stats.linregress(x, y)
        slope, intercept = least_squares(x, y)

        self.assertAlmostEqual(slope, result.slope)
        self.assertAlmostEqual(intercept, result.intercept)



class TestPearsonCorrelationCoefficient(unittest.TestCase):


    def test_basic_usage(self):

        x = np.arange(10, 20)
        y = np.array([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])

        result = scipy.stats.linregress(x, y)
        slope, intercept = least_squares(x, y)

        predict_y = x * slope + intercept
        coefficient = pearson_corr_coef(y, predict_y)

        corr_coef: float = np.corrcoef(x, y)[0, 1]
        self.assertAlmostEqual(coefficient, result.rvalue)
        self.assertAlmostEqual(coefficient, corr_coef)

    def test_shuffled_input(self):
        x = np.arange(10, 20)
        y = np.array([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])
        
        x_sorted, y_sorted = unison_shuffled_copies(x, y)
        
        result = scipy.stats.linregress(x_sorted, y_sorted)
        corr_coef: float = np.corrcoef(x_sorted, y_sorted)[0, 1]
        
        slope, intercept = least_squares(x_sorted, y_sorted)

        predict_y = x_sorted * slope + intercept
        coefficient = pearson_corr_coef(y_sorted, predict_y)


        self.assertAlmostEqual(corr_coef, result.rvalue)
        self.assertAlmostEqual(coefficient, result.rvalue)
        self.assertAlmostEqual(coefficient, corr_coef)
