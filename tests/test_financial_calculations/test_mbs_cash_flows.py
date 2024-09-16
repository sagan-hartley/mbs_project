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

class TestCalculateMonthlyPayment(unittest.TestCase):
    def test_valid_cases(self):
        self.assertAlmostEqual(calculate_monthly_payment(1000, 12, 0.05), 85.61, places=2)
        self.assertAlmostEqual(calculate_monthly_payment(1000, 24, 0.05), 43.87, places=2)

    def test_zero_interest_rate(self):
        self.assertAlmostEqual(calculate_monthly_payment(1000, 12, 0), 83.33, places=2)

    def test_zero_months(self):
        with self.assertRaises(ValueError):
            calculate_monthly_payment(1000, 0, 0.05)

    def test_zero_principal(self):
        with self.assertRaises(ValueError):
            calculate_monthly_payment(0, 12, 0.05)

class TestCalculateScheduledBalances(unittest.TestCase):
    def test_basic(self):
        months, scheduled_balances, principal_paydowns, interest_paid = calculate_scheduled_balances(5000, 24, 0.0525, 219.917)
        some_expected_balances = [5000.0, 4801.96, 4603.05, 4403.27, 4202.62]
        np.testing.assert_array_almost_equal(scheduled_balances[0:5], some_expected_balances, decimal=2)
        np.testing.assert_almost_equal(0, scheduled_balances[-1], decimal=2)

class TestCalculateScheduledBalancesWithServiceFee(unittest.TestCase):
    def test_basic(self):
        months, balances, principal_paydowns, interest_paid, net_interest_paid = calculate_scheduled_balances_with_service_fee(5000, 24, 0.0525, 219.917, 0.01)
        some_expected_net_interest_paid = [0, 17.71, 16.99, 16.27, 15.55]
        np.testing.assert_almost_equal(net_interest_paid[0:5], some_expected_net_interest_paid, decimal=2)

class TestCalculateBalancesWithPrepayment(unittest.TestCase):
    def test_basic(self):
        t_less_60 = np.arange(0, 60)
        smm_less_60 = (t_less_60 / 60) * 0.01
        t_greater_equal_60 = np.arange(60, 181)
        smm_greater_equal_60 = 0.015 - (t_greater_equal_60 / 120) * 0.01
        smms = np.concatenate([smm_less_60, smm_greater_equal_60])
        months, scheduled_balances, actual_balances, principal_paydowns, interest_paid, net_interest_paid = calculate_balances_with_prepayment(100, 180, 0.07, 0.0675, smms)
        np.testing.assert_almost_equal(actual_balances[0:3], [100, 99.68, 99.35], decimal=2)
        np.testing.assert_almost_equal(actual_balances[59:62], [58.48, 57.58, 56.67], decimal=2)
        np.testing.assert_almost_equal(actual_balances[178:181], [0.72, 0.36, 0], decimal=2)

class TestCalculateBalancesWithPrepaymentAndDates(unittest.TestCase):
    def test_basic(self):
        t_less_60 = np.arange(0, 60)
        smm_less_60 = (t_less_60 / 60) * 0.01
        t_greater_equal_60 = np.arange(60, 181)
        smm_greater_equal_60 = 0.015 - (t_greater_equal_60 / 120) * 0.01
        smms = np.concatenate([smm_less_60, smm_greater_equal_60])
        months, dates, payment_dates, scheduled_balances, actual_balances, principal_paydowns, interest_paid, net_interest_paid = calculate_balances_with_prepayment_and_dates(100, 180, 0.07, 0.0675, smms, datetime(2024, 1, 1), payment_delay_days=25)
        expected_dates = [datetime(2024, 1, 1) + relativedelta(months=i) for i in range(181)]
        expected_payment_dates = [datetime(2024, 1, 26) + relativedelta(months=i) for i in range(181)]
        np.testing.assert_array_equal(dates, expected_dates)
        np.testing.assert_array_equal(payment_dates, expected_payment_dates)

class TestCalculateWeightedAverageLife(unittest.TestCase):
    def test_basic(self):
        data = {
            'Payment Date': [datetime(2024, 1, 1) + relativedelta(months=i) for i in range(12)],
            'Scheduled Balance': np.linspace(100, 0, 12)
        }
        df = pd.DataFrame(data)
        wal = calculate_weighted_average_life(df, datetime(2024, 1, 1))
        self.assertAlmostEqual(wal, 0.5, places=1)

class TestCalculatePresentValue(unittest.TestCase):
    def test_basic(self):
        schedule = pd.DataFrame({
            'Payment Date': [datetime(2024, 1, 1) + relativedelta(months=i) for i in range(12)],
            'Principal Paydown': np.linspace(100, 0, 12)
        })
        rate_vals = np.full(12, 0.02)
        rate_dates = [datetime(2024, 1, 15) + relativedelta(months=i) for i in range(12)]
        settle_date = datetime(2024, 2, 1)
        present_value = calculate_present_value(schedule, settle_date, rate_vals, rate_dates)
        self.assertAlmostEqual(present_value, 406.24, places=2)

class TestCalculateDirtyPrice(unittest.TestCase):
    def test_basic(self):
        dirty_price = calculate_dirty_price(1000, 950)
        self.assertAlmostEqual(dirty_price, 105.26, places=2)

class TestCalculateCleanPrice(unittest.TestCase):
    def test_basic(self):
        dirty_price = 1020
        settle_date = datetime(2024, 9, 15)
        last_coupon_date = datetime(2024, 8, 15)
        annual_interest_rate = 0.05
        balance_at_settle = 1000
        clean_price = calculate_clean_price(dirty_price, settle_date, last_coupon_date, annual_interest_rate, balance_at_settle)
        self.assertAlmostEqual(clean_price, 1015.756, places=2)
