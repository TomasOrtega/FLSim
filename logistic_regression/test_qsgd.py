import unittest
import numpy as np
from log_reg_utils import qsgd


class TestQSGDQuantizer(unittest.TestCase):

    def test_output_shape(self):
        x = np.random.rand(10)
        d = 4
        quantized_x = qsgd(x, d)
        self.assertEqual(x.shape, quantized_x.shape)

    def test_output_range(self):
        x = np.random.rand(10)
        d = 4
        quantized_x = qsgd(x, d)
        self.assertTrue(np.all(quantized_x >= -np.linalg.norm(x)))
        self.assertTrue(np.all(quantized_x <= np.linalg.norm(x)))

    def test_zero_input(self):
        x = np.zeros(10)
        d = 4
        quantized_x = qsgd(x, d)
        self.assertTrue(np.all(quantized_x == x))

    def test_large_d(self):
        x = np.random.rand(10)
        d = 1000
        quantized_x = qsgd(x, d)
        self.assertTrue(np.all(quantized_x >= -np.linalg.norm(x)))
        self.assertTrue(np.all(quantized_x <= np.linalg.norm(x)))

    def test_negative_input(self):
        x = np.random.rand(10) - 1.0
        d = 4
        quantized_x = qsgd(x, d)
        self.assertTrue(np.all(quantized_x >= -np.linalg.norm(x)))
        self.assertTrue(np.all(quantized_x <= np.linalg.norm(x)))

    def test_unbiased(self):
        x = np.random.rand(10)
        d = 4
        num_samples = 10000
        quantized_x = np.zeros(x.shape)
        for _ in range(num_samples):
            quantized_x += qsgd(x, d)
        quantized_x /= num_samples
        self.assertAlmostEqual(np.linalg.norm(quantized_x), np.linalg.norm(
            x), delta=1/np.sqrt(num_samples) * np.linalg.norm(x))


if __name__ == '__main__':
    unittest.main()
