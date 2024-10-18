import unittest
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
from financial_calculations.coupon_rates import (
    calculate_coupon_rate
)
from financial_calculations.forward_curves import (
    bootstrap_forward_curve,
    calibrate_finer_forward_curve
)

class TestCalculateCouponRate(unittest.TestCase):
    """
    Unit test class for testing the `calculate_coupon_rate` function.

    Tests include:
    - Functionality when the start date matches the market close date.
    - Functionality when the start date is after the market close date.
    - Error handling when the start date is before the market close date.
    - Validation that the calculated coupon rate is within valid bounds.
    """

    def setUp(self):
        """
        Set up common data for multiple tests.
        Initializes the forward curve, par value, and maturity years for use in test cases.
        """
        # Test data for bond maturities and coupon rates
        self.cmt_data = [
            (1, 0.03),  # 1-year bond, 3% coupon rate
            (2, 0.04),  # 2-year bond, 4% coupon rate
            (3, 0.05)   # 3-year bond, 5% coupon rate
        ]
        # Market close date in datetime format
        self.market_close_date = datetime(2024, 8, 10)
        # Par value for the bonds
        self.par_value = 100

        # Generate forward curve using bootstrap_forward_curve function
        self.forward_curve = bootstrap_forward_curve(self.cmt_data, self.market_close_date, self.par_value)

        # Generate a finer forward curve using calibrate_finer_forward_curve function
        self.finer_forward_curve = calibrate_finer_forward_curve(self.cmt_data, self.market_close_date, self.par_value)
        
        # Set the maturity years for bond calculations
        self.maturity_years = 2

    def test_start_date_is_market_close_date(self):
        """
        Test when the start date is exactly the market close date.
        Verifies that the coupon rate returned matches the spot rate for the market close date.
        """
        start_date = self.market_close_date
        coarse_coupon_rate = calculate_coupon_rate(start_date, self.maturity_years, self.forward_curve)
        fine_coupon_rate = calculate_coupon_rate(start_date, self.maturity_years, self.finer_forward_curve)
        # The expected rate should match the spot rate at the market close date
        expected_rate = self.cmt_data[1][1]  # The coupon rate corresponding to the market close date
        self.assertAlmostEqual(coarse_coupon_rate, expected_rate, places=4)
        self.assertAlmostEqual(fine_coupon_rate, expected_rate, places=4)

    def test_start_date_is_after_market_close_date(self):
        """
        Test when the start date is after the market close date.
        Validates that the coupon rate is within the expected bounds [0, 1].
        """
        start_date = self.market_close_date + relativedelta(months = 1)
        coupon_rate = calculate_coupon_rate(start_date, self.maturity_years, self.forward_curve)
        # Coupon rate should be within the range [0, 1]
        self.assertTrue(coupon_rate >= 0, "Coupon rate should be >= 0")
        self.assertTrue(coupon_rate <= 1, "Coupon rate should be <= 1")

    def test_invalid_start_date(self):
        """
        Test when the start date is before the market close date.
        Ensures that a ValueError is raised in this case to handle invalid input.
        """
        start_date = self.market_close_date - relativedelta(months = 1)
        with self.assertRaises(ValueError):
            calculate_coupon_rate(start_date, self.maturity_years, self.forward_curve)

    def test_coupon_rate_bounds(self):
        """
        Test that the calculated coupon rate is within the valid range [0, 1].
        Ensures that the returned coupon rate adheres to this range.
        """
        start_date = datetime(2024, 8, 10)
        coupon_rate = calculate_coupon_rate(start_date, self.maturity_years, self.forward_curve)
        self.assertGreaterEqual(coupon_rate, 0, "Coupon rate should be >= 0")
        self.assertLessEqual(coupon_rate, 1, "Coupon rate should be <= 1")

if __name__ == '__main__':
    unittest.main()
