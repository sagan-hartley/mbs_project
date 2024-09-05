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

    def test_basic_functionality(self):
        """
        Test the generation of a vector of zero-coupon bond (ZCB) values.
        Verifies that a vector of ZCB values is correctly generated for
        given payment dates and discount rates.
        """
        payment_dates = [datetime(2025, 7, 1), datetime(2026, 1, 1), datetime(2026, 7, 1)]
        rate_vals = [0.05, 0.04, 0.05]
        rate_dates = [datetime(2025, 1, 1), datetime(2026, 1, 1), datetime(2027, 1, 1)]

        expected_values = [
            np.exp(-rate_vals[0] * 0.5),  # 1/2 year
            np.exp(-rate_vals[0] * 1.0) ,  # 1 years
            np.exp(-rate_vals[0] * 1.0 - rate_vals[1] * 0.5) # 1 1/2 years
        ]
        result = get_ZCB_vector(payment_dates, rate_vals, rate_dates)
        np.testing.assert_array_almost_equal(result, expected_values, decimal=3)

if __name__ == '__main__':
    unittest.main()
