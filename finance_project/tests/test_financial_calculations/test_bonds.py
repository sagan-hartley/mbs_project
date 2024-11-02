import unittest
import numpy as np
import pandas as pd
from financial_calculations.bonds import SemiBondContract, create_semi_bond_cash_flows, CashFlowData

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
        self.assertEqual(cash_flows.principal_payments[-1], 1025.0, "The last payment should include the principal repayment")

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

if __name__ == '__main__':
    unittest.main()
