import unittest
import numpy as np
import pandas as pd
from financial_calculations.bonds import (
    SemiBondContract,
     CashFlowData,
    create_semi_bond_cash_flows,
    calculate_coupon_rate
)
from financial_calculations.forward_curves import (
    bootstrap_forward_curve,
    calibrate_fine_curve
)

class TestCreateSemiBondCashFlows(unittest.TestCase):
    """Unit tests for the create_semi_bond_cash_flows function."""

    def setUp(self):
        """Set up test variables for the test cases."""
        self.origination_date = '2024-01-01'
        self.term_in_months = 60  # 5 years
        self.coupon = 0.05  # 5%
        self.balance = 1000.0

    def test_valid_cash_flows(self):
        """Test the creation of cash flows with valid parameters."""
        bond_contract = SemiBondContract(
            self.origination_date,
            self.term_in_months,
            self.coupon,
            self.balance
        )
        cash_flows = create_semi_bond_cash_flows(bond_contract)

        self.assertIsInstance(cash_flows, CashFlowData, "Should return an instance of CashFlowData")
        self.assertEqual(cash_flows.balances.shape[0], 11, "There should be 11 balance entries for a 5-year bond")
        self.assertEqual(cash_flows.principal_payments[0], 0.0, "The first payment should be zero")
        self.assertEqual(cash_flows.principal_payments[-1], 1000.0, "The last payment should include the principal repayment")
        self.assertEqual(cash_flows.interest_payments[0], 0, "The first interest payment should be 0")
        self.assertEqual(cash_flows.interest_payments[1], 25.0, "The second interest payment should be 25.0")

    def test_coupon_greater_than_one(self):
        """Test that ValueError is raised if coupon is greater than 1."""
        bond_contract = SemiBondContract(
            self.origination_date,
            self.term_in_months,
            1.05,  # Invalid coupon greater than 1
            self.balance
        )
        with self.assertRaises(ValueError) as context:
            create_semi_bond_cash_flows(bond_contract)
        self.assertEqual(
            str(context.exception),
            "Coupon should not be greater than 1 as it should be a decimal and not a percentage."
        )

    def test_edge_case_balance(self):
        """Test the cash flow creation with a zero balance."""
        bond_contract = SemiBondContract(
            self.origination_date,
            self.term_in_months,
            self.coupon,
            0.0  # Zero balance
        )
        cash_flows = create_semi_bond_cash_flows(bond_contract)

        self.assertTrue(np.all(cash_flows.balances == 0.0), "All balances should be zero for a zero balance bond")
        self.assertEqual(cash_flows.principal_payments[0], 0.0, "The first payment should still be zero")
        self.assertEqual(cash_flows.principal_payments[-1], 0.0, "The last payment should be zero as well")

    def test_payment_dates_generation(self):
        """Test that payment dates are generated correctly."""
        bond_contract = SemiBondContract(
            self.origination_date,
            self.term_in_months,
            self.coupon,
            self.balance
        )
        cash_flows = create_semi_bond_cash_flows(bond_contract)

        expected_dates = pd.date_range(start=self.origination_date, periods=11, freq=pd.DateOffset(months=6))
        np.testing.assert_array_equal(cash_flows.payment_dates, expected_dates, "Payment dates do not match expected dates")

class TestCalculateCouponRate(unittest.TestCase):
    """
    Unit test class for testing the `calculate_coupon_rate` function.
    """

    def setUp(self):
        """
        Set up common data for multiple tests.
        Initializes the forward curves, par value, and maturity years for use in test cases.
        """
        # Test data for bond maturities and coupon rates
        # All bonds effective date is 01/01/2024
        self.cmt_data = [
            (pd.Timestamp("2024-01-01"), 1, 0.03),  # 1-year bond, 3% coupon rate
            (pd.Timestamp("2024-01-01"), 2, 0.04),  # 2-year bond, 4% coupon rate
            (pd.Timestamp("2024-01-01"), 3, 0.05)   # 3-year bond, 5% coupon rate
        ]
        # Market close date in datetime format
        self.market_close_date = pd.Timestamp("2024-01-01")
        # Par value for the bonds
        self.par_value = 100

        # Generate forward curve using bootstrap_forward_curve function
        self.forward_curve= bootstrap_forward_curve(self.market_close_date, self.cmt_data, self.par_value)

        # Generate a fine forward curve using calibrate_fine_curve function
        self.finer_forward_curve = calibrate_fine_curve(self.market_close_date, self.cmt_data, self.par_value)
        
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
        expected_rate = self.cmt_data[1][2]  # The coupon rate corresponding to the market close date
        self.assertAlmostEqual(coarse_coupon_rate, expected_rate, places=4)
        self.assertAlmostEqual(fine_coupon_rate, expected_rate, places=4)

    def test_start_date_is_after_market_close_date(self):
        """
        Test when the start date is after the market close date.
        Validates that the coupon rate is within the expected bounds [0, 1].
        """
        start_date = self.market_close_date + pd.DateOffset(months = 1)
        coupon_rate = calculate_coupon_rate(start_date, self.maturity_years, self.forward_curve)
        # Coupon rate should be within the range [0, 1]
        self.assertTrue(coupon_rate >= 0, "Coupon rate should be >= 0")
        self.assertTrue(coupon_rate <= 1, "Coupon rate should be <= 1")

    def test_invalid_start_date(self):
        """
        Test when the start date is before the market close date.
        Ensures that a ValueError is raised in this case to handle invalid input.
        """
        start_date = self.market_close_date - pd.DateOffset(months = 1)
        with self.assertRaises(ValueError):
            calculate_coupon_rate(start_date, self.maturity_years, self.forward_curve)

    def test_coupon_rate_bounds(self):
        """
        Test that the calculated coupon rate is within the expected range [0, 1].
        Ensures that the returned coupon rate adheres to this range.
        """
        start_date = pd.Timestamp("2024-06-01")
        coupon_rate = calculate_coupon_rate(start_date, self.maturity_years, self.forward_curve)
        self.assertGreaterEqual(coupon_rate, 0, "Coupon rate should be >= 0")
        self.assertLessEqual(coupon_rate, 1, "Coupon rate should be <= 1")

if __name__ == '__main__':
    unittest.main()
