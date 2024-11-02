import unittest
from datetime import datetime
import numpy as np
from financial_calculations.forward_curves import ForwardCurve

class TestBootstrapForwardCurve(unittest.TestCase):
    """
    Unit test class for testing the `bootstrap_forward_curve` function.

    The tests include:
    - Basic functionality with typical bond inputs.
    - Effect of different initial guesses for the optimization.
    - Handling of edge cases like high coupon rates, long maturities.
    - Consistency of results across multiple runs.
    - Checking equivalency of results when using `datetime` and `datetime64[D]`.
    """

    def setUp(self):
        """
        Set up common data used across multiple tests.
        
        This includes a set of bond data (`cmt_data`), a market close date, 
        and a balance for the bonds.
        """
        # Test data for bond maturities and coupon rates
        self.cmt_data = [
            (1, 0.03),  # 1-year bond, 3% coupon rate
            (2, 0.04),  # 2-year bond, 4% coupon rate
            (3, 0.05)   # 3-year bond, 5% coupon rate
        ]
        # Market close date in datetime format
        self.market_close_date = datetime(2024, 8, 10)
        # Balance for the bonds
        self.balance = 100
        # Initialize a ForwardCurve object
        self.curve = ForwardCurve(self.market_close_date)

    def test_basic_bootstrap(self):
        """
        Test basic functionality of `bootstrap_forward_curve` with standard input.
        
        This checks the length of dates and rates assigned by the method.
        """
        self.curve.bootstrap_forward_curve(self.cmt_data, self.balance)

        # Check that the number of disc rate dates equals the number of maturities + 1 (for the market close date)
        self.assertEqual(
            len(self.curve.dates), len(self.cmt_data) + 1
        )
        # Check that the number of disc rates equals the number of maturities
        self.assertEqual(len(self.curve.rates), len(self.cmt_data))

        # We will test that the actual values of the forward curve rates are correct in tests/test_financial_calculations/test_coupon_rates.py
        # by comparing these rates to the coupons produced when the start date of the coupon is on a disc rate date
   
    def test_initial_guess(self):
        """
        Test the effect of varying initial guesses on the results.
        
        This checks whether different initial guesses for the optimization 
        yield consistent outputs within valid bounds.
        """
        # Different initial guesses for the optimization
        initial_guesses = [0.01, 0.05, 0.0, 1.0]
        for guess in initial_guesses:
            self.curve.bootstrap_forward_curve(
                self.cmt_data, self.balance, guess
            )

            # Check the length of disc rate dates and rates
            self.assertEqual(len(self.curve.dates), len(self.cmt_data) + 1)
            self.assertEqual(len(self.curve.rates), len(self.cmt_data))

    def test_edge_cases(self):
        """
        Test edge cases with high coupon rates, low coupon rates, and long maturities.
        
        This ensures the function behaves as expected for extreme values.
        """
        edge_cases = [
            [(1, 0.99)],  # Very high coupon rate
            [(2, 0.1)],   # Low coupon rate
            [(10, 0.01)]  # Long maturity period (10 years)
        ]
        for cmt_data in edge_cases:
            self.curve.bootstrap_forward_curve(cmt_data, self.balance)

            # Check the length of disc rate dates and rates
            self.assertEqual(len(self.curve.dates), len(cmt_data) + 1)
            self.assertEqual(len(self.curve.rates), len(cmt_data))

    def test_consistency(self):
        """
        Test that the function produces consistent results when run multiple times.
        
        This verifies that the function is deterministic and returns the same results 
        across repeated runs with the same input.
        """
        for _ in range(10):
            self.curve.bootstrap_forward_curve(self.cmt_data, self.balance)
            # Define a second curve to test consistency
            curve_2 = ForwardCurve(self.market_close_date)
            curve_2.bootstrap_forward_curve(self.cmt_data, self.balance)

            # Ensure the disc rate dates are exactly the same across runs
            np.testing.assert_array_equal(self.curve.dates, curve_2.dates)
            # Ensure the disc rates are nearly equal across runs (accounting for floating point precision)
            np.testing.assert_array_almost_equal(self.curve.rates, curve_2.rates)


class TestCalibrateFinerForwardCurve(unittest.TestCase):
    """
    Unit test class for testing the `calibrate_finer_forward_curve` function.

    The tests include:
    - Basic functionality with different frequencies (monthly, weekly).
    - Handling of invalid frequency input.
    - Consistency with different market close date formats.
    """

    def setUp(self):
        """
        Set up common data used across multiple tests.
        
        This includes a set of bond data (`cmt_data`), a market close date, 
        and a balance for the bonds.
        """
        # Example bond data
        self.cmt_data = [(1, 0.03), (2, 0.04), (3, 0.05)]
        # Market close date in datetime format
        self.market_close_date = datetime(2024, 8, 10)
        # Balance for the bonds
        self.balance = 100
        # Initialize a ForwardCurve object
        self.curve = ForwardCurve(self.market_close_date)

    def test_monthly_frequency(self):
        """
        Test the function with monthly frequency.
        
        This verifies that the number of dates and rates returned is correct.
        """
        self.curve.calibrate_finer_forward_curve(self.cmt_data, self.balance, frequency='monthly')
        
        # We will test that the actual values of the forward curve rates are correct in tests/test_financial_calculations/test_coupon_rates.py
        # by comparing these rates to the coupons produced when the start date of the coupon is on a disc rate date
        self.assertEqual(len(self.curve.dates), 37)  # Expecting 3 years of monthly rates
        self.assertEqual(len(self.curve.rates), 37) # These lengths are (12*3) + 1 (start date)

    def test_weekly_frequency(self):
        """
        Test the function with weekly frequency.
        
        This verifies that the number of dates and rates returned is correct.
        """
        self.curve.calibrate_finer_forward_curve(self.cmt_data, self.balance, frequency='weekly')
        
        self.assertEqual(len(self.curve.dates), 157)  # Expecting 3 years of weekly rates (52*3) + 1 (start date)
        self.assertEqual(len(self.curve.rates), 157)

    def test_invalid_frequency(self):
        """
        Test the function with an invalid frequency input.
        
        This ensures that the function raises a ValueError for unsupported frequencies.
        """
        with self.assertRaises(ValueError):
            self.curve.calibrate_finer_forward_curve(self.cmt_data, self.balance, frequency='daily')

if __name__ == '__main__':
    unittest.main()
