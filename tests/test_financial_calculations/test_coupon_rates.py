import unittest
from datetime import datetime
import numpy as np
from financial_calculations.coupon_rates import calculate_coupon_rate

class TestCalculateCouponRate(unittest.TestCase):
    """
    Unit test class for testing the `calculate_coupon_rate` function.

    Tests include:
    - Functionality when the start date matches the market close date.
    - Functionality when the start date is after the market close date.
    - Error handling when the start date is before the market close date.
    - Validation of coupon rate bounds.
    """

    def setUp(self):
        """
        Set up common data for multiple tests.
        Initializes the forward curve, par value, and maturity years for use in test cases.
        """
        self.forward_curve = (
            np.array([datetime(2024, 8, 10), datetime(2025, 2, 10), datetime(2025, 8, 10)]), 
            np.array([0.02, 0.025, 0.03])  # Sample spot rates
        )
        self.par_value = 1000
        self.maturity_years = 1

    def test_start_date_is_market_close_date(self):
        """
        Test when the start date is exactly the market close date.
        """
        start_date = datetime(2024, 8, 10)
        coupon_rate = calculate_coupon_rate(start_date, self.maturity_years, self.par_value, self.forward_curve)
        expected_rate = self.forward_curve[2][0]  # Expect rate for market close date
        self.assertAlmostEqual(coupon_rate, expected_rate, places=4)

    def test_start_date_is_after_market_close_date(self):
        """
        Test when the start date is after the market close date.
        Validates that the coupon rate is within expected bounds.
        """
        start_date = datetime(2024, 9, 10)
        coupon_rate = calculate_coupon_rate(start_date, self.maturity_years, self.par_value, self.forward_curve)
        # Spot rates should be bounded between 0 and 1
        self.assertTrue(coupon_rate >= 0, "Coupon rate should be >= 0")
        self.assertTrue(coupon_rate <= 1, "Coupon rate should be <= 1")

    def test_invalid_start_date(self):
        """
        Test when the start date is before the market close date.
        Ensures that a ValueError is raised in this case.
        """
        start_date = datetime(2024, 7, 10)
        with self.assertRaises(ValueError):
            calculate_coupon_rate(start_date, self.maturity_years, self.par_value, self.forward_curve)

    def test_coupon_rate_bounds(self):
        """
        Test that the coupon rate is within the valid range [0, 1].
        Ensures that the calculated coupon rate adheres to this range.
        """
        start_date = datetime(2024, 8, 10)
        coupon_rate = calculate_coupon_rate(start_date, self.maturity_years, self.par_value, self.forward_curve)
        self.assertGreaterEqual(coupon_rate, 0, "Coupon rate should be >= 0")
        self.assertLessEqual(coupon_rate, 1, "Coupon rate should be <= 1")

if __name__ == '__main__':
    unittest.main()
