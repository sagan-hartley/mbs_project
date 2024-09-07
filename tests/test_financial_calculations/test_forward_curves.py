import unittest
from datetime import datetime
import numpy as np
from financial_calculations.forward_curves import (
    bootstrap_forward_curve
)

class TestBootstrapForwardCurve(unittest.TestCase):

    def setUp(self):
        # Common data for multiple tests
        self.cmt_data = [
            (1, 0.03),  # 1-year bond, 3% coupon
            (2, 0.04),  # 2-year bond, 4% coupon
            (3, 0.10)   # 3-year bond, 10% coupon
        ]
        self.market_close_date = datetime(2024, 8, 10)
        self.par_value = 1000
        self.initial_guess = 0.03

    def test_basic_bootstrap(self):
        # Test that the function returns a correct structure with valid inputs
        spot_rate_dates, spot_rates = bootstrap_forward_curve(self.cmt_data, self.market_close_date, self.par_value)

        # Check that output lengths match the input
        self.assertEqual(len(spot_rate_dates), len(self.cmt_data))
        self.assertEqual(len(spot_rates), len(self.cmt_data))

        # Verify that spot rates are non-negative
        print(spot_rates)
        self.assertTrue(np.all(np.array(spot_rates) >= 1))

if __name__ == '__main__':
    unittest.main()