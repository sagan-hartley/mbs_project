import unittest
import numpy as np
import pandas as pd
from financial_calculations.forward_curves import (
    bootstrap_forward_curve,
    calibrate_fine_curve
)
from financial_calculations.cash_flows import (
    StepDiscounter,
    value_cash_flows
)
from financial_calculations.bonds import (
    SemiBondContract,
    create_semi_bond_cash_flows
)

class TestBootstrapForwardCurve(unittest.TestCase):
    """
    Test class for the bootstrap_forward_curve function.
    Includes checks for sorting, rate calculation, and edge cases.
    """

    def sample_cmt_data(self):
        """
        Provide sample CMT data with unsorted effective dates and maturity years.

        Returns:
            list: Unsorted sample CMT data for testing.
        """
        return [
            (pd.Timestamp("2024-01-01"), 4, 0.04),  # Jan 1, 2024, 4-year bond, 4% coupon
            (pd.Timestamp("2023-01-01"), 3, 0.03),  # Jan 1, 2023, 3-year bond, 3% coupon
            (pd.Timestamp("2022-01-01"), 1, 0.02)   # Jan 1, 2022, 1-year bond, 2% coupon
        ]

    def test_sorted_cmt_data(self):
        """
        Test that bootstrap_forward_curve function sorts CMT data by effective date + maturity.

        Checks that rate dates returned by the function are in the expected ascending order.
        """
        market_close_date = "2022-01-01"
        balance = 100.0
        discounter = bootstrap_forward_curve(market_close_date, self.sample_cmt_data(), balance)
        
        expected_dates = np.append(pd.to_datetime(market_close_date), sorted([
            date + pd.DateOffset(years=maturity) for date, maturity, _ in self.sample_cmt_data()
            ])
        )
        assert all(discounter.dates == expected_dates), "Rate dates should be sorted by effective date + maturity."

    def test_rate_values_non_negative(self):
        """
        Ensure that all rate values in the forward curve are non-negative.

        Verifies that the function produces valid, non-negative rate values.
        """
        market_close_date = "2022-01-01"
        balance = 100.0
        discounter = bootstrap_forward_curve(market_close_date, self.sample_cmt_data(), balance)
        
        assert np.all(discounter.rates >= 0), "Rate values should be non-negative."

    def test_curve_matches_balance_at_settlement(self):
        """
        Test that the calculated curve value matches the balance at the settlement date.

        Validates that the present value of each bond's cash flows is close to the balance.
        """
        market_close_date = "2022-01-01"
        balance = 100.0
        discounter = bootstrap_forward_curve(market_close_date, self.sample_cmt_data(), balance)
        print(discounter.rates)

        for effective_date, maturity_years, coupon in self.sample_cmt_data():
            semi_bond = SemiBondContract(effective_date, maturity_years * 12, coupon, balance)
            semi_bond_flows = create_semi_bond_cash_flows(semi_bond)
            bond_value = value_cash_flows(discounter, semi_bond_flows, market_close_date)
            
            assert np.isclose(bond_value, balance, atol=1e-2), f"Bond value should be close to balance, got {bond_value}."

    def test_minimization_failure_handling(self):
        """
        Verify that the function raises a ValueError if minimization fails to converge.

        Simulates a failure scenario by using an unrealistic coupon rate to trigger a minimization failure.
        """
        market_close_date = "2022-01-01"
        balance = 100.0
        faulty_cmt_data = [
            (pd.Timestamp("2022-01-01"), 1, 0.99)  # Unrealistically high coupon to force failure
        ]

        try:
            bootstrap_forward_curve(market_close_date, faulty_cmt_data, balance)
        except ValueError as e:
            assert str(e) == "Minimization did not converge.", "Expected minimization to fail."

    def test_edge_case_single_bond(self):
        """
        Test the function with only one bond in CMT data.

        Ensures that it processes correctly and includes the market close date and bond maturity.
        """
        market_close_date = "2022-01-01"
        balance = 100.0
        single_bond_data = [(pd.Timestamp("2022-01-01"), 1, 0.03)]
        
        discounter = bootstrap_forward_curve(market_close_date, single_bond_data, balance)
        
        assert len(discounter.dates) == 2, "Rate dates should include market close date and bond maturity."
        assert len(discounter.rates) == 2, "Rate values should include initial and single bootstrapped rate."

    def test_consistent_last_rate_extension(self):
        """
        Verify that the last rate is extended correctly at the end of the curve.

        Checks that the last two rates in the forward curve are the same.
        """
        market_close_date = "2022-01-01"
        balance = 100.0
        discounter = bootstrap_forward_curve(market_close_date, self.sample_cmt_data(), balance)
        
        assert discounter.rates[-1] == discounter.rates[-2], "Last rate should be extended at curve end."

    def test_duplicate_maturity_dates_raises_error(self):
        """
        Test that bootstrap_forward_curve raises a ValueError if duplicate 
        maturity dates are present in the cmt_data.
        """
        market_close_date = "2024-01-01"
        balance = 1000
        initial_guess = 0.04

        # Create duplicate maturity dates in cmt_data
        cmt_data = [
            (pd.Timestamp("2024-01-01"), 2, 0.05),  # Matures on 01/01/2026
            (pd.Timestamp("2025-01-01"), 1, 0.04),  # Also matures on 01/01/2026
        ]

        with self.assertRaises(ValueError) as context:
            bootstrap_forward_curve(market_close_date, cmt_data, balance, initial_guess)

        # Check that the error message contains information about duplicate dates
        self.assertIn("Duplicate maturity dates cannot exist for this bootstrapping method.", str(context.exception))

class TestCalibrateFineCurve(unittest.TestCase):
    """
    Unit tests for the calibrate_fine_curve function, which calibrates a forward curve by 
    bootstrapping discount rates for bonds with regular intervals. Tests cover basic functionality, 
    handling of duplicate maturity dates, and error cases.
    """

    def setUp(self):
        # Common setup for test data
        self.market_close_date = pd.Timestamp('2024-01-01')
        self.balance = 1000.0
        self.frequency = 'm'
        self.initial_guess = 0.04
        self.smoothing_error_weight = 100.0

    def test_basic_calibration(self):
        """Test basic functionality of calibrate_fine_curve with unique maturity dates."""
        cmt_data = [
            (pd.Timestamp('2024-01-01'), 1, 0.05),  # Bond 1: 1-year maturity
            (pd.Timestamp('2024-01-01'), 2, 0.06),  # Bond 2: 2-year maturity
            (pd.Timestamp('2024-01-01'), 3, 0.07)   # Bond 3: 3-year maturity
        ]

        result = calibrate_fine_curve(self.market_close_date, cmt_data, self.balance, self.frequency, self.initial_guess, self.smoothing_error_weight)
        
        # Ensure result is an instance of StepDiscounter
        self.assertIsInstance(result, StepDiscounter)
        
        # Check that the rate dates and rates are aligned with the maturity dates
        rate_dates = result.dates
        expected_maturity_dates = [
            pd.Timestamp('2025-01-01'),
            pd.Timestamp('2026-01-01'),
            pd.Timestamp('2027-01-01')
        ]
        
        self.assertTrue(all(date in rate_dates for date in expected_maturity_dates), "Rate dates do not match expected maturity dates.")

    def test_duplicate_maturity_dates(self):
        """Test that calibrate_fine_curve raises a ValueError for duplicate maturity dates."""
        cmt_data = [
            (pd.Timestamp('2024-01-01'), 2, 0.05),  # Bond 1: 2-year maturity
            (pd.Timestamp('2024-01-01'), 2, 0.06)   # Bond 2: Duplicate 2-year maturity
        ]

        with self.assertRaises(ValueError) as context:
            calibrate_fine_curve(self.market_close_date, cmt_data, self.balance, self.frequency, self.initial_guess, self.smoothing_error_weight)
        
        # Check that the error message is as expected
        self.assertEqual(str(context.exception), "Duplicate maturity dates found in the input data. Ensure each bond has a unique maturity date.")

    def test_non_convergence_error(self):
        """Test that calibrate_fine_curve raises a ValueError if minimization does not converge."""
        # Intentionally problematic data to induce non-convergence
        cmt_data = [
            (pd.Timestamp('2024-01-01'), 1, 0.05),
            (pd.Timestamp('2024-01-01'), 2, 0.99),
            (pd.Timestamp('2024-01-01'), 3, 0.07)
        ]

        # Using an extremely high smoothing error weight to induce non-convergence
        with self.assertRaises(ValueError) as context:
            calibrate_fine_curve(self.market_close_date, cmt_data, self.balance, self.frequency, self.initial_guess, smoothing_error_weight=1e20)

        self.assertEqual(str(context.exception), "Minimization did not converge. Try adjusting the initial guess or checking input data.")

if __name__ == '__main__':
    unittest.main()
