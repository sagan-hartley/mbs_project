import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from financial_calculations.mbs_cash_flows import (
    calculate_monthly_payment,
    calculate_scheduled_balances,
    calculate_scheduled_balances_with_service_fee,
    calculate_balances_with_prepayment,
    calculate_balances_with_prepayment_and_dates,
    calculate_weighted_average_life,
    calculate_present_value,
    calculate_dirty_price,
    calculate_clean_price
)

class BaseTestCase(unittest.TestCase):
    """
    Set up a general test case to be used across each function
    """

    def setUp(self):
        self.balance = 100
        self.num_months = 180
        self.gross_annual_interest_rate = 0.07
        self.net_annual_interest_rate = 0.0675
        self.service_fee_rate = self.gross_annual_interest_rate - self.net_annual_interest_rate
        t_less_60 = np.arange(0, 60)
        smm_less_60 = (t_less_60 / 60) * 0.01
        t_greater_equal_60 = np.arange(60, 180)
        smm_greater_equal_60 = 0.015 - (t_greater_equal_60 / 120) * 0.01
        self.smms = np.concatenate([smm_less_60, smm_greater_equal_60])
        self.market_close_date = datetime(2008, 1, 1)
        self.payment_delay_days = 24

        self.gross_rate_monthly_payment = calculate_monthly_payment(self.balance, self.num_months, self.gross_annual_interest_rate)

        self.net_rate_monthly_payment = calculate_monthly_payment(self.balance, self.num_months, self.net_annual_interest_rate)

        self.mbs = calculate_balances_with_prepayment_and_dates(self.balance, self.num_months, self.gross_annual_interest_rate, self.net_annual_interest_rate, self.smms, self.market_close_date, self.payment_delay_days)

        self.mbs_df = pd.DataFrame(list(zip(*self.mbs)), columns=['Month', 'Date', 'Payment Date', 'Scheduled Balance', 'Actual Balance', 'Principal Paydown', 'Interest Paid', 'Net Interest Paid'])

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

        self.discount_rate_dates = [datetime(2008, 1, 1) + relativedelta(months=i) for i in range(181)]


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
        months, scheduled_balances, principal_paydowns, interest_paid = calculate_scheduled_balances(
            self.balance, self.num_months, self.gross_annual_interest_rate, self.gross_rate_monthly_payment
        )
        some_expected_balances = [100.  ,  99.68,  99.37,  99.05,  98.73]
        np.testing.assert_array_almost_equal(scheduled_balances[0:5], some_expected_balances, decimal=2)
        np.testing.assert_almost_equal(0, scheduled_balances[-1], decimal=2)


class TestCalculateScheduledBalancesWithServiceFee(BaseTestCase):
    """
    Test cases for the calculate_scheduled_balances_with_service_fee function.
    """

    def test_basic(self):
        """Test calculation of balances including service fees."""
        months, balances, principal_paydowns, interest_paid, net_interest_paid = calculate_scheduled_balances_with_service_fee(
            self.balance, self.num_months, self.gross_annual_interest_rate, self.gross_rate_monthly_payment, self.service_fee_rate
        )
        some_expected_net_interest_paid = [0.56, 0.56, 0.56, 0.55, 0.55]
        np.testing.assert_almost_equal(net_interest_paid[0:5], some_expected_net_interest_paid, decimal=2)


class TestCalculateBalancesWithPrepayment(BaseTestCase):
    """
    Test cases for the calculate_balances_with_prepayment function.
    """

    def test_basic(self):
        """Test calculation of balances with prepayment rates."""
        
        months, scheduled_balances, actual_balances, principal_paydowns, interest_paid, net_interest_paid = calculate_balances_with_prepayment(
            self.balance, self.num_months, self.gross_annual_interest_rate, self.gross_rate_monthly_payment, self.smms
        )
        np.testing.assert_almost_equal(actual_balances[0:3], [100, 99.68, 99.35], decimal=2)
        np.testing.assert_almost_equal(actual_balances[59:62], [58.48, 57.58, 56.67], decimal=2)
        np.testing.assert_almost_equal(actual_balances[178:181], [0.72, 0.36, 0], decimal=2)


class TestCalculateBalancesWithPrepaymentAndDates(BaseTestCase):
    """
    Test cases for the calculate_balances_with_prepayment_and_dates function.
    """

    def test_basic(self):
        """Test calculation of balances with prepayment rates and payment dates."""
        months, dates, payment_dates, scheduled_balances, actual_balances, principal_paydowns, interest_paid, net_interest_paid = calculate_balances_with_prepayment_and_dates(
            self.balance, self.num_months, self.gross_annual_interest_rate, self.gross_rate_monthly_payment, self.smms, self.market_close_date, self.payment_delay_days
        )
        np.testing.assert_almost_equal(actual_balances[0:3], [100, 99.68, 99.35], decimal=2)
        np.testing.assert_almost_equal(actual_balances[59:62], [58.48, 57.58, 56.67], decimal=2)
        np.testing.assert_almost_equal(actual_balances[178:181], [0.72, 0.36, 0], decimal=2)
        expected_dates = [datetime(2008, 1, 1) + relativedelta(months=i) for i in range(181)]
        expected_payment_dates = [datetime(2008, 1, 25) + relativedelta(months=i) for i in range(181)]
        np.testing.assert_array_equal(dates, expected_dates)
        np.testing.assert_array_equal(payment_dates, expected_payment_dates)


class TestCalculateWeightedAverageLife(BaseTestCase):
    """
    Test cases for the calculate_weighted_average_life function.
    """

    def test_basic(self):
        """Test calculation of the weighted average life of cash flows."""
        wal = calculate_weighted_average_life(self.mbs_df, self.market_close_date, balance_name='Actual Balance')
        self.assertAlmostEqual(wal, 6.580976207916489, places=1)


class TestCalculatePresentValue(BaseTestCase):
    """
    Test cases for the calculate_present_value function.
    """

    def test_basic(self):
        """Test present value calculation of a cash flow schedule."""
        present_value = calculate_present_value(self.mbs_df, self.market_close_date, self.discount_rates, self.discount_rate_dates)
        self.assertAlmostEqual(present_value, 111.01502829546669)


class TestCalculateDirtyPrice(unittest.TestCase):
    """
    Test cases for the calculate_dirty_price function.
    """

    def test_basic(self):
        """Test dirty price calculation from par value and bond price."""
        dirty_price = calculate_dirty_price(1000, 950)
        self.assertAlmostEqual(dirty_price, 105.26, places=2)


class TestCalculateCleanPrice(unittest.TestCase):
    """
    Test cases for the calculate_clean_price function.
    """

    def test_basic(self):
        """Test clean price calculation from dirty price and accrued interest."""
        dirty_price = 1020
        settle_date = datetime(2024, 9, 15)
        last_coupon_date = datetime(2024, 8, 15)
        annual_interest_rate = 0.05
        balance_at_settle = 1000
        clean_price = calculate_clean_price(dirty_price, settle_date, last_coupon_date, annual_interest_rate, balance_at_settle)
        self.assertAlmostEqual(clean_price, 1015.756, places=2)


if __name__ == '__main__':
    unittest.main()
