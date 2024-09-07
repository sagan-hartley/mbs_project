import unittest
import numpy as np
from datetime import datetime
from financial_calculations.bond_cash_flows_and_pricing import (
    create_semi_bond_cash_flows
)

class TestCreateSpotSemiBondCashFlows(unittest.TestCase):
    """
    Test suite for the create_spot_semi_bond_cash_flows function.
    This suite tests the function's behavior across various scenarios, 
    including basic functionality, handling of edge cases, and correct 
    interpretation of inputs.
    """
    
    def setUp(self):
        """Set up common test variables."""
        self.market_close_date = np.datetime64('2024-08-10', 'D')
        self.balance = 1000000
        self.coupon = 0.05  # 5% coupon
        self.maturity_years = 5
    
    def test_basic_functionality(self):
        """
        Test the basic functionality of the create_spot_semi_bond_cash_flows function.
        
        This test verifies that the function returns the correct payment dates and 
        cash flows for a bond with a standard 5-year maturity, 5% coupon, and a 
        market close date of "2024-08-10".
        """
        expected_dates = np.array([
            np.datetime64('2025-02-10', 'D'),
            np.datetime64('2025-08-10', 'D'),
            np.datetime64('2026-02-10', 'D'),
            np.datetime64('2026-08-10', 'D'),
            np.datetime64('2027-02-10', 'D'),
            np.datetime64('2027-08-10', 'D'),
            np.datetime64('2028-02-10', 'D'),
            np.datetime64('2028-08-10', 'D'),
            np.datetime64('2029-02-10', 'D'),
            np.datetime64('2029-08-10', 'D')
        ], dtype='datetime64[D]')
        
        expected_cash_flows = np.array([25000.0] * 9 + [1025000.0])
        
        payment_dates, cash_flows = create_semi_bond_cash_flows(
            self.market_close_date, 
            self.balance, 
            self.coupon, 
            self.maturity_years
        )
        
        # Check if the payment dates match
        np.testing.assert_array_equal(payment_dates, expected_dates)
        
        # Check if the cash flows match
        np.testing.assert_array_equal(cash_flows, expected_cash_flows)
    
    def test_day_of_month_greater_than_28(self):
        """
        Test that the function raises a ValueError for invalid market close dates.
        
        This test verifies that the function correctly raises a ValueError 
        when the market close date is beyond the 28th of the month, 
        to avoid potential end-of-month issues.
        """
        with self.assertRaises(ValueError):
            create_semi_bond_cash_flows("2024-08-29", self.balance, self.coupon, self.maturity_years)
    
    def test_coupon_as_percentage(self):
        """
        Test that the function raises a ValueError when the coupon rate 
        is provided as a percentage (i.e., greater than 1).
    
        This test ensures that the function correctly rejects coupon rates 
        greater than 1, which should be input as a decimal.
        """

        coupon_as_percentage = 5  # 5% coupon, invalid as it should be 0.05
        expected_error_message = "Coupon should not be greater than 1 as it should be a decimal and not a percentage."

        with self.assertRaises(ValueError) as context:
            create_semi_bond_cash_flows(
                self.market_close_date, 
                self.balance, 
                coupon_as_percentage, 
                self.maturity_years
            )
    
        # Check if the error message is correct
        self.assertEqual(str(context.exception), expected_error_message)

    def test_market_close_date_type(self):
        """
        Test that the function returns the same values when the market close date 
        is input as a string, datetime object, or datetime64[D] object
    
        This test ensures that the function correctly deals with different types
        of input for the market close date
        """

        np_market_close_date = self.market_close_date
        dt_market_close_date = datetime(2024, 8, 10)
        str_market_close_date = '2024-08-10'

        np_payment_dates, np_cash_flows = create_semi_bond_cash_flows(
            np_market_close_date, 
            self.balance, 
            self.coupon, 
            self.maturity_years
        )

        dt_payment_dates, dt_cash_flows = create_semi_bond_cash_flows(
            dt_market_close_date, 
            self.balance, 
            self.coupon, 
            self.maturity_years
        )

        str_payment_dates, str_cash_flows = create_semi_bond_cash_flows(
            str_market_close_date, 
            self.balance, 
            self.coupon, 
            self.maturity_years
        )

        # Check all payment dates match
        np.array_equal(np_payment_dates, dt_payment_dates)
        np.array_equal(dt_payment_dates, str_payment_dates)

        # Check all cash flows match
        np.array_equal(np_cash_flows, dt_cash_flows)
        np.array_equal(dt_cash_flows, str_cash_flows)
    
    def test_maturity_less_than_one_year(self):
        """
        Test the function's behavior for bonds with a maturity of less than one year.
        
        This test verifies that the function correctly handles cases where 
        the bond has a very short maturity, specifically ensuring that the 
        final payment date and cash flow are accurate.
        """
        short_maturity_years = 0.5
        expected_dates = [np.datetime64('2025-02-10', 'D')]
        expected_cash_flows = np.array([1025000.0])
        
        payment_dates, cash_flows = create_semi_bond_cash_flows(
            self.market_close_date, 
            self.balance, 
            self.coupon, 
            short_maturity_years
        )
        
        # Check if the payment dates match
        np.testing.assert_array_equal(payment_dates, expected_dates)
        
        # Check if the cash flows match
        np.testing.assert_array_equal(cash_flows, expected_cash_flows)
        
if __name__ == "__main__":
    unittest.main()