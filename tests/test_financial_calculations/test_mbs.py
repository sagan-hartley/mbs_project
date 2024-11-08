import unittest
import numpy as np
import pandas as pd
from financial_calculations.mbs import (
    calculate_monthly_payment,
    calculate_scheduled_balances,
    calculate_actual_balances,
    MbsContract,
    pathwise_evaluate_mbs
)
from financial_calculations.cash_flows import (
    StepDiscounter,
    calculate_weighted_average_life,
    value_cash_flows,
    price_cash_flows
)
from utils import (
    create_regular_dates_grid
)

class BaseTestCase(unittest.TestCase):
    """
    Set up a general test case to be used across each function
    """

    def setUp(self):
        # Initial loan balance
        self.balance = 100
        
        # Number of months for the loan term (15 years)
        self.num_months = 180
        
        # Annual gross interest rate (before service fee)
        self.gross_annual_interest_rate = 0.07
        
        # Net annual interest rate (after deducting the service fee)
        self.net_annual_interest_rate = 0.0675
        
        # Service fee rate, calculated as the difference between gross and net annual interest rates
        self.service_fee_rate = self.gross_annual_interest_rate - self.net_annual_interest_rate

        # Prepayment rates (SMMs) over the first 60 months and beyond
        t_less_60 = np.arange(0, 60)
        smm_less_60 = (t_less_60 / 60) * 0.01
        t_greater_equal_60 = np.arange(60, 180)
        smm_greater_equal_60 = 0.015 - (t_greater_equal_60 / 120) * 0.01
        self.smms = np.concatenate([smm_less_60, smm_greater_equal_60])

        # Origination date
        self.origination_date = pd.Timestamp("2008-01-01")

        # Settle dates for testing payment-related functions
        self.settle_dates = [pd.Timestamp("2008-01-01"), pd.Timestamp("2008-01-15"), pd.Timestamp("2008-02-01"), pd.Timestamp("2008-02-25")]
        self.last_coupon_dates = [pd.Timestamp("2008-01-01"), pd.Timestamp("2008-01-01"), pd.Timestamp("2008-02-01"), pd.Timestamp("2008-02-01")]
        
        # Payment delay in days
        self.payment_delay_days = 24
        
        # Balance at the time of each settlement date
        self.balances_at_settle = [100, 100, 99.68450506248092, 99.68450506248092]

        # Expected present values of cash flows for different settle dates
        self.expected_present_values = [111.01502829546669, 111.04476929266467, 110.20580839957505, 110.57219293568582]
        
        # Expected clean prices for different settle dates
        self.expected_clean_prices = [111.01502829546669, 110.78226929266465, 110.55460257389001, 110.47214669309]

        # calculate the monthly payment and scheduled balance data for the MBS
        self.monthly_payment = calculate_monthly_payment(self.balance, self.num_months, self.gross_annual_interest_rate)
        self.scheduled_mbs = calculate_scheduled_balances(self.balance, self.origination_date, self.num_months, self.gross_annual_interest_rate, payment_delay=self.payment_delay_days)

        # Calculate MBS cash flows with prepayment and payment dates
        self.mbs = calculate_actual_balances(self.scheduled_mbs, self.smms, self.net_annual_interest_rate)
        
        # Discount rates for present value calculations (one rate per month)
        self.discount_rates = np.array([
            0.006983617, 0.050476979, 0.051396376, 0.081552298, 0.045289981,
            0.029759723, 0.073969810, 0.003862879, 0.044871200,
            0.080978350, 0.097701138, 0.039590508, 0.049888840, 0.044515418,
            0.012596944, 0.008476654, 0.055574490, 0.035067305, 0.027167920,
            0.050371274, 0.032307761, 0.063749816, 0.008043366, 0.000253558,
            0.009324560, 0.093725868, 0.018888765, 0.024112974, 0.079826671,
            0.043751914, 0.021189997, 0.056026806, 0.085557557, 0.093818736,
            0.082996689, 0.054250755, 0.001240193, 0.025839389, 0.016727361,
            0.080124400, 0.009524058, 0.094808205, 0.038406391, 0.026020178,
            0.068923774, 0.032581043, 0.042404405, 0.038393590, 0.096957855,
            0.009599077, 0.082073126, 0.038449382, 0.008570729, 0.031334810,
            0.020902689, 0.007693663, 0.075357745, 0.005020825, 0.091085792,
            0.066223832, 0.004008308, 0.082499100, 0.026124495, 0.013431940,
            0.061752758, 0.015007716, 0.067035171, 0.045503376, 0.071814626,
            0.035016636, 0.047617013, 0.036258346, 0.052667676, 0.049712667,
            0.087182244, 0.046867727, 0.056173239, 0.088372271, 0.079152652,
            0.085316035, 0.050163748, 0.092035497, 0.084334787, 0.012739123,
            0.086784384, 0.082238121, 0.012813235, 0.045083599, 0.076907051,
            0.016017826, 0.062559817, 0.071020318, 0.038820162, 0.015048803,
            0.072752192, 0.026428880, 0.019477818, 0.052925817, 0.013761646,
            0.025814134, 0.003362728, 0.097627457, 0.022566484, 0.014211485,
            0.009787030, 0.092952964, 0.062951546, 0.056557990, 0.028243254,
            0.047776915, 0.094574003, 0.075722719, 0.090114523, 0.037370282,
            0.060771702, 0.099045273, 0.047339119, 0.030011234, 0.026550502,
            0.059545442, 0.029886582, 0.017509765, 0.067687091, 0.019248311,
            0.048724795, 0.087316041, 0.082405213, 0.000383088, 0.052046979,
            0.034628922, 0.051488041, 0.039743271, 0.054243464, 0.057612575,
            0.006979987, 0.023464708, 0.087048217, 0.018840603, 0.029695179,
            0.073279064, 0.057930599, 0.084524461, 0.012712518, 0.012014110,
            0.082756616, 0.037463306, 0.097436147, 0.080246738, 0.040601990,
            0.058930602, 0.086115746, 0.088906747, 0.074375454, 0.080537366,
            0.055050880, 0.051720078, 0.070774953, 0.074015762, 0.096252685,
            0.052755209, 0.013849016, 0.090894101, 0.001734406, 0.061806135,
            0.090170217, 0.054950115, 0.079689761, 0.088656840, 0.016996897,
            0.041160525, 0.061011024, 0.096765968, 0.053248733, 0.084173193,
            0.008111603, 0.048784956, 0.086477867, 0.046061337, 0.023838794,
            0.009723155, 0.050000000
        ])

        # Market close date (start date for calculations)
        self.market_close_date = pd.Timestamp("2008-01-01")

        # Corresponding discount rate dates, incremented monthly starting from market close
        self.discount_rate_dates = create_regular_dates_grid(self.market_close_date, self.market_close_date + pd.DateOffset(months=self.num_months))

        self.discounter = StepDiscounter(self.discount_rate_dates, self.discount_rates)

class TestCalculateMonthlyPayment(unittest.TestCase):
    """
    Test cases for the calculate_monthly_payment function.
    """

    def test_valid_cases(self):
        """Test monthly payments with different valid inputs."""
        self.assertAlmostEqual(calculate_monthly_payment(1000, 12, 0.05), 85.61, places=2)
        self.assertAlmostEqual(calculate_monthly_payment(1000, 24, 0.05), 43.87, places=2)

    def test_zero_interest_rate(self):
        """Test monthly payments with zero interest rate."""
        self.assertAlmostEqual(calculate_monthly_payment(1000, 12, 0), 83.33, places=2)

    def test_zero_months(self):
        """Test that zero months raises a ValueError."""
        with self.assertRaises(ValueError):
            calculate_monthly_payment(1000, 0, 0.05)

    def test_zero_principal(self):
        """Test that zero principal raises a ValueError."""
        with self.assertRaises(ValueError):
            calculate_monthly_payment(0, 12, 0.05)


class TestCalculateScheduledBalances(BaseTestCase):
    """
    Test cases for the calculate_scheduled_balances function.
    """

    def test_basic(self):
        """Test calculation of scheduled balances over a set period."""
        scheduled_balances = self.scheduled_mbs.balances
        some_expected_balances = [100.  ,  99.68,  99.37,  99.05,  98.73]
        np.testing.assert_array_almost_equal(scheduled_balances[0:5], some_expected_balances, decimal=2)
        np.testing.assert_almost_equal(0, scheduled_balances[-1], decimal=2)

    def test_invalid_origination_date(self):
        """Test calculation of scheduled balances with an invalid origination date"""
        with self.assertRaises(ValueError):
            calculate_scheduled_balances(self.balance, pd.Timestamp("2024-10-02"), self.num_months, self.gross_annual_interest_rate)

class TestCalculateActualBalances(BaseTestCase):
    """
    Test cases for the calculate_actual_balances function.
    """

    def test_basic(self):
        """Test calculation of balances with prepayment rates and payment dates."""
        dates = self.mbs.accrual_dates
        payment_dates = self.mbs.payment_dates
        actual_balances = self.mbs.balances
        np.testing.assert_almost_equal(actual_balances[0:3], [100, 99.68, 99.35], decimal=2)
        np.testing.assert_almost_equal(actual_balances[59:62], [58.48, 57.58, 56.67], decimal=2)
        np.testing.assert_almost_equal(actual_balances[178:181], [0.72, 0.36, 0], decimal=2)
        expected_dates = create_regular_dates_grid(pd.Timestamp("2008-01-01"), pd.Timestamp("2008-01-01") + pd.DateOffset(months=self.num_months))
        expected_payment_dates = create_regular_dates_grid(pd.Timestamp("2008-01-25"), pd.Timestamp("2008-01-25") + pd.DateOffset(months=self.num_months))
        np.testing.assert_array_equal(dates, expected_dates)
        np.testing.assert_array_equal(payment_dates, expected_payment_dates)

class TestCalculateWeightedAverageLife(BaseTestCase):
    """
    Test cases for the calculate_weighted_average_life function.
    """

    def test_basic(self):
        """Test calculation of the weighted average life of cash flows."""
        wals = []
        for i in range(4):
            wal = calculate_weighted_average_life(self.mbs, self.settle_dates[i])
            wals.append(wal)
        np.testing.assert_array_almost_equal(wals, [6.580976207916489, 6.542620043532927, 6.51639615231036, 6.450642727652829])


class TestPresentValue(BaseTestCase):
    """
    Test cases that ensure the value of the mbs of cash flows aligns with the value_cash_flows function.
    """

    def test_basic(self):
        """Test present value calculation of cash flow schedules."""
        present_values = []
        for i in range(4):
            present_value = value_cash_flows(self.discounter, self.mbs, self.settle_dates[i])
            present_values.append(present_value)
        np.testing.assert_array_almost_equal(present_values, self.expected_present_values)

class TestCalculateCleanPrice(BaseTestCase):
    """
    Test cases that ensure the price of the mbs of cash flows aligns with the price_cash_flows function.
    """

    def test_basic(self):
        """Test clean price calculation."""
        clean_prices = []
        for i in range(4):
            clean_price = price_cash_flows(self.expected_present_values[i], self.balances_at_settle[i], self.settle_dates[i], self.last_coupon_dates[i], self.net_annual_interest_rate)
            clean_prices.append(clean_price)
        np.testing.assert_array_almost_equal(clean_prices, self.expected_clean_prices)

class TestMbsContractDataClass(unittest.TestCase):
    """
    Unit tests for the MbsContract dataclass.

    This class tests various aspects of the MbsContract dataclass, including initialization,
    validation of inputs, and expected behavior of methods and attributes.
    """

    def setUp(self):
        """
        Set up initial data for MBS tests.
        """
        self.mbs_data = {
            'mbs_id': 'MBS001',
            'balance': 1000000.0,
            'origination_date': pd.to_datetime('2024-01-01'),
            'num_months': 360,
            'gross_annual_coupon': 0.03,
            'net_annual_coupon': 0.025,
            'payment_delay': 30,
            'settle_date': pd.to_datetime('2024-02-01')
        }

        # Create the Mbs Contract object
        self.mbs = MbsContract(**self.mbs_data)

    def test_initialization(self):
        """
        Test initialization of the MbsContract dataclass.

        This test ensures that the class initializes correctly with the provided dates,
        balance, and other attributes.
        """
        # Verify if the MBS object has the expected attributes
        self.assertEqual(self.mbs.mbs_id, self.mbs_data['mbs_id'])
        self.assertEqual(self.mbs.balance, self.mbs_data['balance'])
        self.assertEqual(self.mbs.num_months, self.mbs_data['num_months'])
        self.assertEqual(self.mbs.gross_annual_coupon, self.mbs_data['gross_annual_coupon'])
        self.assertEqual(self.mbs.net_annual_coupon, self.mbs_data['net_annual_coupon'])
        self.assertEqual(self.mbs.payment_delay, self.mbs_data['payment_delay'])

        # Ensure that the origination_date and settle_date are pandas Timestamps
        self.assertIsInstance(self.mbs.origination_date, pd.Timestamp)
        self.assertIsInstance(self.mbs.settle_date, pd.Timestamp)

    def test_invalid_origination_date(self):
        """
        Test that the MbsContract dataclass raises an error when an invalid origination date is provided.
        """
        with self.assertRaises(ValueError):
            invalid_date = "not_a_valid_date"
            MbsContract(mbs_id=self.mbs_data['mbs_id'], balance=self.mbs_data['balance'], 
                origination_date=invalid_date, num_months=self.mbs_data['num_months'], 
                gross_annual_coupon=self.mbs_data['gross_annual_coupon'], 
                net_annual_coupon=self.mbs_data['net_annual_coupon'], 
                payment_delay=self.mbs_data['payment_delay'], settle_date=self.mbs_data['settle_date'])

    def test_payment_delay(self):
        """
        Test that the MbsContract dataclass correctly handles the payment delay.
        """
        # Check that the payment_delay is as expected
        self.assertEqual(self.mbs.payment_delay, self.mbs_data['payment_delay'])

    def test_settle_date(self):
        """
        Test that the settle_date is correctly initialized as a pandas Timestamp.
        """
        self.assertIsInstance(self.mbs.settle_date, pd.Timestamp)
        self.assertEqual(self.mbs.settle_date, self.mbs_data['settle_date'])

    def test_balance(self):
        """
        Test the balance attribute to ensure it matches the expected value.
        """
        self.assertEqual(self.mbs.balance, self.mbs_data['balance'])

    def test_invalid_settle_date(self):
        """
        Test that the MbsContract dataclass raises an error when an invalid settle date is provided.
        """
        with self.assertRaises(ValueError):
            invalid_date = "not_a_valid_date"
            MbsContract(mbs_id=self.mbs_data['mbs_id'], balance=self.mbs_data['balance'], 
                origination_date=self.mbs_data['origination_date'], num_months=self.mbs_data['num_months'], 
                gross_annual_coupon=self.mbs_data['gross_annual_coupon'], 
                net_annual_coupon=self.mbs_data['net_annual_coupon'], 
                payment_delay=self.mbs_data['payment_delay'], settle_date=invalid_date)

class TestPathwiseEvaluateMBS(unittest.TestCase):
    """
    Unit tests for the pathwise_evaluate_mbs function.

    This suite tests various input conditions and the correctness of results returned by
    the pathwise_evaluate_mbs function, including edge cases and validation for output
    when different flags (store_vals, store_expecteds, store_stdevs) are set to True or False.

    Note that these tests are not testing the explicit values of the results from the 
    pathwise_evaluate_mbs function as the functions providing them (value_cash_flows, calculate_wal, etc...)
    are already thouroughly tested elsewhere.
    """

    def setUp(self):
        """
        Set up test data for MBS and short rates.
        """
        # Example MbsContract object
        self.mbs = MbsContract(
            mbs_id='MBS001',
            balance=1000000.0,
            origination_date=pd.to_datetime('2024-01-01'),
            num_months=180,
            gross_annual_coupon=0.05,
            net_annual_coupon=0.0475,
            payment_delay=30,
            settle_date=pd.to_datetime('2024-02-01')
        )

        # Simulated short rate paths (2 paths, 200 periods each)
        self.short_rates = np.array([np.ones(200)*0.05, 
                                     np.ones(200)*0.06])

        # Short rate dates (corresponding dates for the short rates)
        self.short_rate_dates = create_regular_dates_grid('2024-01-01', '2040-08-01')

        # List of MBS objects (in this case just one)
        self.mbs_list = [self.mbs]

    def test_basic_functionality(self):
        """
        Test the basic functionality of pathwise_evaluate_mbs with default flags.
        """
        results = pathwise_evaluate_mbs(self.mbs_list, self.short_rates, self.short_rate_dates)

        # Check the result structure for the first MBS
        self.assertEqual(len(results), 1)  # One MBS in the list
        mbs_result = results[0]
        
        # Ensure MBS ID is correct
        self.assertEqual(mbs_result['mbs_id'], self.mbs.mbs_id)
        
        # Check that the results contain expected keys
        self.assertIn('wals', mbs_result)
        self.assertIn('vals', mbs_result)
        self.assertIn('prices', mbs_result)
        self.assertIn('expected_wal', mbs_result)
        self.assertIn('expected_value', mbs_result)
        self.assertIn('expected_price', mbs_result)
        self.assertIn('wal_stdev', mbs_result)
        self.assertIn('value_stdev', mbs_result)
        self.assertIn('price_stdev', mbs_result)

    def test_store_vals_false(self):
        """
        Test that setting store_vals=False does not store individual path values.
        """
        results = pathwise_evaluate_mbs(self.mbs_list, self.short_rates, self.short_rate_dates, store_vals=False)

        mbs_result = results[0]
        self.assertNotIn('wals', mbs_result)
        self.assertNotIn('vals', mbs_result)
        self.assertNotIn('prices', mbs_result)

    def test_store_expecteds_false(self):
        """
        Test that setting store_expecteds=False does not store the expected (mean) values.
        """
        results = pathwise_evaluate_mbs(self.mbs_list, self.short_rates, self.short_rate_dates, store_expecteds=False)

        mbs_result = results[0]
        self.assertNotIn('expected_wal', mbs_result)
        self.assertNotIn('expected_value', mbs_result)
        self.assertNotIn('expected_price', mbs_result)

    def test_store_stdevs_false(self):
        """
        Test that setting store_stdevs=False does not store the standard deviations.
        """
        results = pathwise_evaluate_mbs(self.mbs_list, self.short_rates, self.short_rate_dates, store_stdevs=False)

        mbs_result = results[0]
        self.assertNotIn('wal_stdev', mbs_result)
        self.assertNotIn('value_stdev', mbs_result)
        self.assertNotIn('price_stdev', mbs_result)

    def test_antithetic_stdev(self):
        """
        Test that setting antithetic=True calculates standard deviation of antithetic pairs correctly
        """
        results = pathwise_evaluate_mbs(self.mbs_list, self.short_rates, self.short_rate_dates, antithetic=True)
        
        # Check the result structure for the first MBS
        self.assertEqual(len(results), 1)  # One MBS in the list
        mbs_result = results[0]
        
        # Ensure MBS ID is correct
        self.assertEqual(mbs_result['mbs_id'], self.mbs.mbs_id)
        
        # Check that the results contain expected keys
        self.assertIn('wals', mbs_result)
        self.assertIn('vals', mbs_result)
        self.assertIn('prices', mbs_result)
        self.assertIn('expected_wal', mbs_result)
        self.assertIn('expected_value', mbs_result)
        self.assertIn('expected_price', mbs_result)
        self.assertIn('wal_stdev', mbs_result)
        self.assertIn('value_stdev', mbs_result)
        self.assertIn('price_stdev', mbs_result)

    def test_empty_mbs_list(self):
        """
        Test the case when an empty MBS list is provided.
        """
        results = pathwise_evaluate_mbs([], self.short_rates, self.short_rate_dates)
        self.assertEqual(results, [])

    def test_single_path(self):
        """
        Test the case where only a single short rate path is provided.
        """
        single_path_rates = np.ones(200)*0.05
        results = pathwise_evaluate_mbs(self.mbs_list, single_path_rates, self.short_rate_dates)

        mbs_result = results[0]
        self.assertEqual(len(mbs_result['wals']), 1)  # Only one path should be present
        self.assertEqual(len(mbs_result['vals']), 1)
        self.assertEqual(len(mbs_result['prices']), 1)

    def test_inconsistent_length_of_short_rates_and_dates(self):
        """
        Test that the function raises an error when the lengths of short_rates and short_rate_dates don't match.
        """
        inconsistent_dates = create_regular_dates_grid('2024-01-01', '2040-07-01')  # 199 dates instead of 200
        with self.assertRaises(ValueError):
            pathwise_evaluate_mbs(self.mbs_list, self.short_rates, inconsistent_dates)

if __name__ == '__main__':
    unittest.main()
