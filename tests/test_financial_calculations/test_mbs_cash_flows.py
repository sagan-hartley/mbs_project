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
        months, balances, principal_paydowns, interest_paid = calculate_scheduled_balances(1000, 12, 0.05, 85.61)
        expected_balances = [1000.0, 917.37, 833.53, 748.44, 662.11, 574.51, 485.63, 395.45, 303.94, 210.09, 113.88, 14.29]
        np.testing.assert_almost_equal(balances, expected_balances, decimal=2)

class TestCalculateScheduledBalancesWithServiceFee(unittest.TestCase):
    def test_basic(self):
        months, balances, principal_paydowns, interest_paid, net_interest_paid = calculate_scheduled_balances_with_service_fee(1000, 12, 0.05, 85.61, 0.01)
        expected_net_interest_paid = [4.17, 3.34, 2.50, 1.67, 0.83, 0.00, -0.83, -1.67, -2.50, -3.34, -4.17, -5.00]
        np.testing.assert_almost_equal(net_interest_paid[1:], expected_net_interest_paid[1:], decimal=2)

class TestCalculateBalancesWithPrepayment(unittest.TestCase):
    def test_basic(self):
        smms = np.zeros(12)
        months, scheduled_balances, actual_balances, principal_paydowns, interest_paid, net_interest_paid = calculate_balances_with_prepayment(1000, 12, 0.05, 0.04, smms)
        np.testing.assert_almost_equal(scheduled_balances[1:], [926.36, 852.84, 778.49, 703.31, 627.27, 550.34, 472.50, 393.73, 314.01, 233.31, 150.60, 65.86], decimal=2)

class TestCalculateBalancesWithPrepaymentAndDates(unittest.TestCase):
    def test_basic(self):
        smms = np.zeros(12)
        origination_date = datetime(2024, 1, 1)
        months, dates, payment_dates, scheduled_balances, actual_balances, principal_paydowns, interest_paid, net_interest_paid = calculate_balances_with_prepayment_and_dates(1000, 12, 0.05, 0.04, smms, origination_date)
        self.assertEqual(dates[0], origination_date)
        self.assertEqual(payment_dates[0], origination_date + timedelta(days=24))
        np.testing.assert_almost_equal(scheduled_balances[1:], [926.36, 852.84, 778.49, 703.31, 627.27, 550.34, 472.50, 393.73, 314.01, 233.31, 150.60, 65.86], decimal=2)

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
        dirty_price = 105.26
        settle_date = datetime(2024, 1, 1)
        last_coupon_date = datetime(2023, 12, 1)
        annual_interest_rate = 0.05
        balance_at_settle = 950
        clean_price = calculate_clean_price(dirty_price, settle_date, last_coupon_date, annual_interest_rate, balance_at_settle)
        self.assertAlmostEqual(clean_price, 101.17, places=2)
