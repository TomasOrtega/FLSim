import numpy as np
import unittest
from sklearn.metrics import log_loss
from scipy.special import expit as sigmoid
import log_reg_utils


def loss(weights, X, y, reg):
    probabilities = sigmoid(np.dot(X, weights))
    res = log_loss(y, probabilities)
    res += reg * np.square(weights).sum() / 2
    return res


def logistic_loss(weights, X, y, reg):
    raw_scores = np.dot(X, weights)
    probabilities = 1 / (1 + np.exp(-raw_scores))
    res = np.mean(-y * np.log(probabilities) -
                  (1 - y) * np.log(1 - probabilities))
    res += reg * np.sum(np.square(weights)) / 2
    return res


class TestLogisticRegressionLoss(unittest.TestCase):

    def test_loss_function(self):
        # Test with a simple example
        weights = np.array([1.0, .5])
        X = 0.1 * np.array([[1.0, 2.0], [2.0, 3.0]])
        y = np.array([1, 0.0])
        reg = 0.1

        # Compute the expected loss
        expected_loss = loss(weights, X, y, reg)

        computed_loss = logistic_loss(weights, X, y, reg)

        # Assert that the computed loss is close to the expected value
        self.assertAlmostEqual(computed_loss, expected_loss, places=6)

    def test_loss_function_2(self):
        # Test with a simple example
        weights = np.array([1.0, .5])
        X = 0.1 * np.array([[1.0, 2.0], [2.0, 3.0]])
        y = np.array([1, 0.0])
        reg = 0.1

        # Compute the expected loss
        expected_loss = logistic_loss(weights, X, y, reg)

        computed_loss = log_reg_utils.loss(weights, X, y, reg)

        # Assert that the computed loss is close to the expected value
        self.assertAlmostEqual(computed_loss, expected_loss, places=12)


if __name__ == '__main__':
    unittest.main()
