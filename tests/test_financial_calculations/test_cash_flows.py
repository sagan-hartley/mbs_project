import unittest
from datetime import datetime
import numpy as np
import pandas as pd
from financial_calculations.cash_flows import (
    CASH_DAYS_IN_YEAR,
    CashFlowData,
    StepDiscounter,
    filter_cash_flows,
    value_cash_flows,
    price_cash_flows,
    get_balance_at_settle,
    calculate_weighted_average_life,
    get_last_coupon_date
)
from utils import (
    days360
)

class TestCashFlowData(unittest.TestCase):
    """
    Test class for the CashFlowData class to validate its functionality.
    """

    def setUp(self):
        """
        Set up sample data to use in test cases.
        """
        self.balances = np.array([1000, 900, 800])
        self.accrual_dates = pd.to_datetime([datetime(2024, 1, 1), datetime(2024, 2, 1), datetime(2024, 3, 1)])
        self.payment_dates = pd.to_datetime([datetime(2024, 1, 15), datetime(2024, 2, 15), datetime(2024, 3, 15)])
        self.principal_payments = np.array([100, 100, 100])
        self.interest_payments = np.array([10, 9, 8])

    def test_initialization_success(self):
        """
        Test successful initialization with matching array lengths.
        """
        cash_flow = CashFlowData(self.balances, self.accrual_dates, self.payment_dates, 
                                 self.principal_payments, self.interest_payments)
        self.assertEqual(cash_flow.get_size(), 3)
    
    def test_initialization_mismatched_lengths(self):
        """
        Test that initialization raises a ValueError when input arrays have different lengths.
        """
        with self.assertRaises(ValueError):
            CashFlowData(self.balances, self.accrual_dates, self.payment_dates, 
                         self.principal_payments, np.array([10, 9]))  # Shorter interest_payments array

    def test_get_size(self):
        """
        Test the get_size method to ensure it returns the correct number of elements.
        """
        cash_flow = CashFlowData(self.balances, self.accrual_dates, self.payment_dates, 
                                 self.principal_payments, self.interest_payments)
        self.assertEqual(cash_flow.get_size(), 3)

    def test_get_total_payments(self):
        """
        Test the get_total_payments method to verify the total of principal and interest payments.
        """
        cash_flow = CashFlowData(self.balances, self.accrual_dates, self.payment_dates, 
                                 self.principal_payments, self.interest_payments)
        expected_total_payments = np.array([110, 109, 108])
        np.testing.assert_array_equal(cash_flow.get_total_payments(), expected_total_payments)

class TestStepDiscounter(unittest.TestCase):
    """
    Unit tests for the StepDiscounter class.
    
    Tests include:
    --------------
    - Initialization with matching dates and rates length.
    - Initialization failure with mismatched dates and rates length.
    - Calculation of zero-coupon bond (ZCB) discount factors for specified dates.
    """

    def setUp(self):
        """Set up test data for StepDiscounter."""
        # Sample dates and rates for initialization
        self.dates = pd.to_datetime(["2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01"])
        self.rates = np.array([0.02, 0.025, 0.03, 0.035])

        # Expected ZCB discount factors (placeholders, replace with actual expected values)
        self.expected_zcb_factors = np.array([1.0, 0.99, 0.98, 0.96])

    def test_initialization_success(self):
        """Test successful initialization with matching dates and rates length."""
        discounter = StepDiscounter(self.dates, self.rates)
        self.assertEqual(len(discounter.dates), len(discounter.rates))
        self.assertEqual(discounter.market_close_date, self.dates[0])
    
    def test_initialization_failure(self):
        """Test initialization fails if dates and rates lengths do not match."""
        with self.assertRaises(ValueError):
            StepDiscounter(self.dates, self.rates[:-1])  # Mismatched length

    def test_zcbs_from_dates(self):
        """Test ZCB discount factors calculation for specified dates."""
        discounter = StepDiscounter(self.dates, self.rates)
        
        # Dates for ZCB discount factor calculation
        zcb_dates = pd.to_datetime(["2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01"])
        
        # Calculate ZCB discount factors
        zcb_factors = discounter.zcbs_from_dates(zcb_dates)

        # Compare with expected ZCB factors (placeholders for actual expected values)
        np.testing.assert_array_almost_equal(zcb_factors, self.expected_zcb_factors, decimal=2)
    
    def test_market_close_date(self):
        """Verify that the market_close date is set correctly."""
        discounter = StepDiscounter(self.dates, self.rates)
        self.assertEqual(discounter.market_close_date, self.dates[0])

    def test_integral_values(self):
        """Test if the calculated integral values match expectations (placeholder for actual expected values)."""
        discounter = StepDiscounter(self.dates, self.rates)
        
        # Placeholder expected integral values (replace with actual values for more precise testing)
        # Note the extremely high last integral value comes from the MAX_INTERPOLATE_YEARS part of the integral_knots logic
        expected_integral_vals = np.array([0.0, 0.01, 0.02, 0.04, 3.54])
        
        np.testing.assert_array_almost_equal(discounter.integral_vals, expected_integral_vals, decimal=2)

class TestFilterCashFlows(unittest.TestCase):
    """Test cases for the filter_cash_flows function."""

    def setUp(self):
        """Set up initial cash flow data for testing."""
        self.balances = np.array([1000, 900, 800, 700])
        self.accrual_dates = pd.to_datetime(['2024-01-01', '2024-07-01', '2025-01-01', '2025-07-01'])
        self.payment_dates = pd.to_datetime(['2024-01-10', '2024-07-10', '2025-01-10', '2025-07-10'])
        self.principal_payments = np.array([100, 100, 100, 100])
        self.interest_payments = np.array([50, 45, 40, 35])

        # Create an instance of CashFlowData for testing
        self.cash_flows = CashFlowData(
            balances=self.balances,
            accrual_dates=self.accrual_dates,
            payment_dates=self.payment_dates,
            principal_payments=self.principal_payments,
            interest_payments=self.interest_payments
        )

    def test_filter_cash_flows_after_settle_date(self):
        """Test filtering cash flows after a given settlement date."""
        settle_date = '2024-06-01'
        filtered_cash_flows = filter_cash_flows(self.cash_flows, settle_date)

        # Expected indices after the settle date
        expected_balances = np.array([900, 800, 700])
        expected_accrual_dates = pd.to_datetime(['2024-07-01', '2025-01-01', '2025-07-01'])
        expected_payment_dates = pd.to_datetime(['2024-07-10', '2025-01-10', '2025-07-10'])
        expected_principal_payments = np.array([100, 100, 100])
        expected_interest_payments = np.array([45, 40, 35])

        # Assert values of the filtered cash flows
        np.testing.assert_array_equal(filtered_cash_flows.balances, expected_balances)
        pd.testing.assert_index_equal(filtered_cash_flows.accrual_dates, expected_accrual_dates)
        pd.testing.assert_index_equal(filtered_cash_flows.payment_dates, expected_payment_dates)
        np.testing.assert_array_equal(filtered_cash_flows.principal_payments, expected_principal_payments)
        np.testing.assert_array_equal(filtered_cash_flows.interest_payments, expected_interest_payments)

    def test_filter_no_cash_flows_after_settle_date(self):
        """Test case where no cash flows occur after the settlement date."""
        settle_date = '2025-08-01'  # after the last accrual date
        filtered_cash_flows = filter_cash_flows(self.cash_flows, settle_date)

        # Expect empty arrays as output
        self.assertEqual(len(filtered_cash_flows.balances), 0)
        self.assertEqual(len(filtered_cash_flows.accrual_dates), 0)
        self.assertEqual(len(filtered_cash_flows.payment_dates), 0)
        self.assertEqual(len(filtered_cash_flows.principal_payments), 0)
        self.assertEqual(len(filtered_cash_flows.interest_payments), 0)

    def test_filter_cash_flows_on_settle_date(self):
        """Test case where settlement date is exactly the first accrual date."""
        settle_date = '2024-01-01'
        filtered_cash_flows = filter_cash_flows(self.cash_flows, settle_date)

        # Expected values should exclude the first accrual date
        expected_balances = np.array([900, 800, 700])
        expected_accrual_dates = pd.to_datetime(['2024-07-01', '2025-01-01', '2025-07-01'])
        expected_payment_dates = pd.to_datetime(['2024-07-10', '2025-01-10', '2025-07-10'])
        expected_principal_payments = np.array([100, 100, 100])
        expected_interest_payments = np.array([45, 40, 35])

        # Assert values of the filtered cash flows
        np.testing.assert_array_equal(filtered_cash_flows.balances, expected_balances)
        pd.testing.assert_index_equal(filtered_cash_flows.accrual_dates, expected_accrual_dates)
        pd.testing.assert_index_equal(filtered_cash_flows.payment_dates, expected_payment_dates)
        np.testing.assert_array_equal(filtered_cash_flows.principal_payments, expected_principal_payments)
        np.testing.assert_array_equal(filtered_cash_flows.interest_payments, expected_interest_payments)

class TestValueCashFlows(unittest.TestCase):
    """Test cases for the value_cash_flows function."""

    def setUp(self):
        """Set up initial conditions for testing."""
        self.balances = np.array([100, 49, 0])
        self.accrual_dates = pd.to_datetime(['2024-01-01', '2024-07-01', '2025-01-01'])
        self.payment_dates = pd.to_datetime(['2024-01-10', '2024-07-10', '2025-01-10'])
        self.principal_payments = np.array([0, 49, 48])
        self.interest_payments = np.array([0, 2, 1])
        
        # Create an instance of CashFlowData
        self.cash_flows = CashFlowData(
            balances=self.balances,
            accrual_dates=self.accrual_dates,
            payment_dates=self.payment_dates,
            principal_payments=self.principal_payments,
            interest_payments=self.interest_payments
        )

        # Dummy rates and dates for StepDiscounter
        dates = pd.to_datetime(['2023-09-01', '2024-01-01', '2025-01-01'])
        rates = np.array([0.01, 0.02, 0.03])
        self.discounter = StepDiscounter(dates, rates)

    def test_value_cash_flows(self):
        """Test the value_cash_flows function."""
        settle_date = '2024-06-01'
        expected_value = 99.28409955613735
        actual_value = value_cash_flows(self.discounter, self.cash_flows, settle_date)
        self.assertAlmostEqual(actual_value, expected_value, places=2)

    def test_value_no_cash_flows(self):
        """Test case where no cash flows are available after the settle date."""
        settle_date = '2025-08-01'
        actual_value = value_cash_flows(self.discounter, self.cash_flows, settle_date)
        self.assertEqual(actual_value, 0)

class TestPriceCashFlows(unittest.TestCase):
    """
    Unit tests for the price_cash_flows function, which calculates the clean price
    of a bond based on present value, balance at settlement, and accrued interest.
    """
    
    def setUp(self):
        """
        Set up common parameters for use in multiple tests.
        """
        self.present_value = 950
        self.balance_at_settle = 1000
        self.settle_date = datetime(2024, 11, 1)
        self.last_coupon_date = datetime(2024, 5, 1)
        self.annual_interest_rate = 0.05  # 5%
        self.par_balance = 100
        self.expected_dirty_price = self.present_value * self.par_balance / self.balance_at_settle
    
    def test_basic_clean_price_calculation(self):
        """
        Test that the clean price is calculated correctly given standard inputs.
        Verifies that the calculated clean price matches the expected result
        based on accrued interest and dirty price.
        """
        clean_price = price_cash_flows(
            self.present_value, self.balance_at_settle, self.settle_date,
            self.last_coupon_date, self.annual_interest_rate, self.par_balance
        )
        
        days_between = days360(self.last_coupon_date, self.settle_date)
        expected_accrued_interest = (self.annual_interest_rate / CASH_DAYS_IN_YEAR) * days_between * self.par_balance
        expected_clean_price = self.expected_dirty_price - expected_accrued_interest
        
        self.assertAlmostEqual(clean_price, expected_clean_price, places=2)

    def test_zero_balance_at_settle(self):
        """
        Test that the clean price is zero when the balance at settlement is zero.
        """
        clean_price = price_cash_flows(
            self.present_value, 0, self.settle_date,
            self.last_coupon_date, self.annual_interest_rate, self.par_balance
        )
        
        self.assertEqual(clean_price, 0)

    def test_no_accrued_interest_with_same_dates(self):
        """
        Test that no accrued interest is added when settle_date equals last_coupon_date.
        Ensures the clean price matches the dirty price in this case.
        """
        clean_price = price_cash_flows(
            self.present_value, self.balance_at_settle, self.settle_date,
            self.settle_date, self.annual_interest_rate, self.par_balance
        )
        
        self.assertAlmostEqual(clean_price, self.expected_dirty_price, places=2)

    def test_different_par_balance(self):
        """
        Test the clean price calculation with a different par balance.
        Verifies that the function adjusts calculations based on the provided par balance.
        """
        clean_price = price_cash_flows(
            self.present_value, self.balance_at_settle, self.settle_date,
            self.last_coupon_date, self.annual_interest_rate, par_balance=200
        )
        
        expected_dirty_price = self.present_value * 200 / self.balance_at_settle
        days_between = days360(self.last_coupon_date, self.settle_date)
        expected_accrued_interest = (self.annual_interest_rate / CASH_DAYS_IN_YEAR) * days_between * 200
        expected_clean_price = expected_dirty_price - expected_accrued_interest
        
        self.assertAlmostEqual(clean_price, expected_clean_price, places=2)

class TestGetBalanceAtSettle(unittest.TestCase):
    """
    Unit tests for the get_balance_at_settle function.
    """    
    def setUp(self):
        """Setup mock data for tests"""
        self.cash_flows = CashFlowData(
            balances=np.array([1000.0, 950.0, 900.0, 850.0]),
            accrual_dates=np.array(['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01']),
            payment_dates=np.array(['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01']),
            principal_payments=np.array([50.0, 50.0, 50.0, 50.0]),
            interest_payments=np.array([0.0, 0.0, 0.0, 0.0])
        )
        
        self.filtered_cfs_no_filter = CashFlowData(
            balances=np.array([1000.0, 950.0, 900.0, 850.0]),
            accrual_dates=np.array(['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01']),
            payment_dates=np.array(['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01']),
            principal_payments=np.array([50.0, 50.0, 50.0, 50.0]),
            interest_payments=np.array([0.0, 0.0, 0.0, 0.0])
        )
        
        self.filtered_cfs_filtered = CashFlowData(
            balances=np.array([950.0, 900.0]),
            accrual_dates=np.array(['2024-02-01', '2024-03-01']),
            payment_dates=np.array(['2024-02-01', '2024-03-01']),
            principal_payments=np.array([50.0, 50]),
            interest_payments=np.array([0.0, 0.0])
        )

    def test_no_filtering(self):
        """Test balance at settle when no filtering has occurred."""
        result = get_balance_at_settle(self.cash_flows, self.filtered_cfs_no_filter)
        self.assertEqual(result, 1000.0)

    def test_with_filtering(self):
        """Test balance at settle when filtering has occurred."""
        result = get_balance_at_settle(self.cash_flows, self.filtered_cfs_filtered)
        self.assertEqual(result, 1000.0)

    def test_first_payment_date_not_found(self):
        """Test ValueError is raised when first payment date in filtered_cfs is not found."""
        filtered_cfs_invalid = CashFlowData(
            balances=np.array([950.0, 900.0]),
            accrual_dates=np.array(['2024-02-15', '2024-03-01']),
            payment_dates=np.array(['2024-02-15', '2024-03-01']),
            principal_payments=np.array([50.0, 50]),
            interest_payments=np.array([0.0, 0.0])
        )
        with self.assertRaises(ValueError):
            get_balance_at_settle(self.cash_flows, filtered_cfs_invalid)

class TestCalculateWeightedAverageLife(unittest.TestCase):
    """Unit tests for the calculate_weighted_average_life function."""

    def setUp(self):
        """Set up a sample CashFlowData instance for testing."""
        self.cash_flows = CashFlowData(
            balances=np.array([1000, 800, 600, 400, 200, 0]),
            accrual_dates=pd.to_datetime([
                '2024-01-01', 
                '2024-07-01', 
                '2025-01-01', 
                '2025-07-01', 
                '2026-01-01',
                '2026-07-01'
            ]),
            payment_dates=pd.to_datetime([
                '2024-01-15', 
                '2024-07-15', 
                '2025-01-15', 
                '2025-07-15', 
                '2026-01-15',
                '2026-07-15'
            ]),
            principal_payments=np.array([0, 200, 200, 200, 200, 200]),
            interest_payments=np.array([0, 20, 16, 12, 8, 4])
        )
    
    def test_weighted_average_life(self):
        """Test WAL calculation for a settlement date before some payments."""
        settle_date = '2024-06-30'
        expected_wal = 1.0427397260273972  # Based on a manual calculation of the setup data
        calculated_wal = calculate_weighted_average_life(self.cash_flows, settle_date)
        self.assertAlmostEqual(calculated_wal, expected_wal, places=2)

    def test_settle_date_equal_to_first_cash_flow(self):
        """Test WAL calculation when the settle date is equal to the first cash flow date."""
        settle_date = '2024-01-01'
        expected_wal = 1.5386301369863014  # Based on a manual calculation of the setup data
        calculated_wal = calculate_weighted_average_life(self.cash_flows, settle_date)
        self.assertAlmostEqual(calculated_wal, expected_wal, places=2)

    def test_settle_date_no_payments(self):
        """Test WAL calculation when the settle date is after all payments."""
        settle_date = '2026-12-31'  # After all payments
        calculated_wal = calculate_weighted_average_life(self.cash_flows, settle_date)
        self.assertEqual(calculated_wal, 0)  # Expect WAL to be 0

class TestGetLastCouponDate(unittest.TestCase):
    """
    Test case for the get_last_coupon_date function using CashFlowData instances.
    
    This test case verifies the correctness of the get_last_coupon_date 
    function in various scenarios including normal cases and edge cases.
    """

    def setUp(self):
        """Set up the test environment with a sample CashFlowData instance."""
        balances = np.array([1000, 800, 600, 400])
        accrual_dates = pd.to_datetime([
            "2023-01-01",
            "2023-04-01",
            "2023-07-01",
            "2023-10-01"
        ])
        payment_dates = pd.to_datetime([
            "2023-01-01",
            "2023-04-01",
            "2023-07-01",
            "2023-10-01"
        ])
        principal_payments = np.array([200, 200, 200, 200])
        interest_payments = np.array([20, 16, 12, 8])
        
        self.cash_flow_data = CashFlowData(
            balances, accrual_dates, payment_dates, principal_payments, interest_payments
        )

    def test_get_last_coupon_date_valid(self):
        """Test case where settle_date falls after some cash flow dates."""
        settle_date = pd.to_datetime("2023-05-01")
        result = get_last_coupon_date(self.cash_flow_data, settle_date)
        self.assertEqual(result, pd.to_datetime("2023-04-01"))

    def test_get_last_coupon_date_before_first_cash_flow(self):
        """Test case where settle_date is before the first cash flow date."""
        settle_date = pd.to_datetime("2022-12-31")
        result = get_last_coupon_date(self.cash_flow_data, settle_date)
        self.assertEqual(result, settle_date)

    def test_get_last_coupon_date_exactly_on_cash_flow_date(self):
        """Test case where settle_date is exactly on a cash flow date."""
        settle_date = pd.to_datetime("2023-04-01")
        result = get_last_coupon_date(self.cash_flow_data, settle_date)
        self.assertEqual(result, pd.to_datetime("2023-01-01"))

    def test_get_last_coupon_date_after_last_cash_flow(self):
        """Test case where settle_date is after the last cash flow date."""
        settle_date = pd.to_datetime("2023-12-31")
        result = get_last_coupon_date(self.cash_flow_data, settle_date)
        self.assertEqual(result, pd.to_datetime("2023-10-01"))

    def test_get_last_coupon_date_no_cash_flows(self):
        """Test case with no cash flows (empty CashFlowData instance)."""
        empty_cash_flow_data = CashFlowData(
            np.array([]), pd.to_datetime([]), pd.to_datetime([]), np.array([]), np.array([])
        )
        settle_date = pd.to_datetime("2023-05-01")
        result = get_last_coupon_date(empty_cash_flow_data, settle_date)
        self.assertEqual(result, settle_date)

if __name__ == '__main__':
    unittest.main()
