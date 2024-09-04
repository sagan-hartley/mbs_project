import unittest
from datetime import datetime
import numpy as np
from utils import ( 
    get_ZCB_vector
)

class TestGetZCBVector(unittest.TestCase):
    """
    Unit tests for the get_ZCB_vector function.
    """

    def setUp(self):
        """
        Set up test data for get_ZCB_vector.
        """
        self.payment_dates = [datetime(2026, 1, 1), datetime(2027, 1, 1)]
        self.rate_vals = [0.05, 0.04]
        self.rate_dates = [datetime(2025, 1, 1), datetime(2026, 1, 1)]

    def test_get_ZCB_vector(self):
        """
        Test the generation of a vector of zero-coupon bond (ZCB) values.
        Verifies that a vector of ZCB values is correctly generated for
        given payment dates and discount rates.
        """
        expected_values = [
            np.exp(-self.rate_vals[0] * 1.0),  # 1 year
            np.exp(-self.rate_vals[0] * 1.0 - self.rate_vals[1] * 1.0)   # 2 years
        ]
        result = get_ZCB_vector(self.payment_dates, self.rate_vals, self.rate_dates)
        np.testing.assert_array_almost_equal(result, expected_values, decimal=6)

if __name__ == '__main__':
    unittest.main()
