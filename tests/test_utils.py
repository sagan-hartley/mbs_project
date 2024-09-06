import unittest
from datetime import datetime
import numpy as np
import pytest
from utils import ( 
    get_ZCB_vector
)

class TestGetZCBVector(unittest.TestCase):
    """
    Unit tests for the get_ZCB_vector function.
    """

    def test_basic_functionality(self):
        """
        Test the generation of a vector of zero-coupon bond (ZCB) values.
        Verifies that a vector of ZCB values is correctly generated for
        given payment dates and discount rates.
        """
        payment_dates = [datetime(2025, 7, 1), datetime(2026, 1, 1), datetime(2026, 7, 1)]
        rate_vals = [0.05, 0.04, 0.03]
        rate_dates = [datetime(2025, 1, 1), datetime(2026, 1, 1), datetime(2027, 1, 1)]

        expected_values = [
            np.exp(-rate_vals[0] * 0.5),  # 1/2 year
            np.exp(-rate_vals[0] * 1.0) ,  # 1 year
            np.exp(-rate_vals[0] * 1.0 - rate_vals[1] * 0.5) # 1 1/2 years
        ]
        result = get_ZCB_vector(payment_dates, rate_vals, rate_dates)
        np.testing.assert_array_almost_equal(result, expected_values, decimal=3)

    def test_last_payment_date(self):
        """
        Test the generation of a vector of zero-coupon bond (ZCB) values
        when a payment date is beyond the last rate date.
        Verifies that a vector of ZCB values is correctly generated for
        given payment dates and discount rates.
        """
        payment_dates = [datetime(2026, 1, 1), datetime(2027, 1, 1), datetime(2028, 1, 1)]
        rate_vals = [0.05, 0.04, 0.03]
        rate_dates = [datetime(2025, 1, 1), datetime(2026, 1, 1), datetime(2027, 1, 1)]

        expected_values = [
            np.exp(-rate_vals[0] * 1.0),  # 1 year
            np.exp(-rate_vals[0] * 1.0 - rate_vals[1] * 1.0) ,  # 2 years
            np.exp(-rate_vals[0] * 1.0 - rate_vals[1] * 1.0 - rate_vals[2] * 1.0) # 3 years
        ]
        result = get_ZCB_vector(payment_dates, rate_vals, rate_dates)
        np.testing.assert_array_almost_equal(result, expected_values, decimal=3)

    def test_early_payment_dates(self):
        """
        Test the generation of a vector of zero-coupon bond (ZCB) values
        when a payment date is before and on the market close date.
        Verifies that a vector of ZCB values is correctly generated for
        given payment dates and discount rates.
        """
        payment_dates = [datetime(2025, 1, 1), datetime(2026, 1, 1), datetime(2027, 1, 1), datetime(2028, 1, 1)]
        rate_vals = [0.05, 0.04, 0.03, 0.06]
        rate_dates = [datetime(2026, 1, 1), datetime(2027, 1, 1), datetime(2028, 1, 1), datetime(2029, 1, 1)]

        expected_values = [
            0.0, # -1 years
            1.0, # 0 years
            np.exp(-rate_vals[0] * 1.0),  # 1 year
            np.exp(-rate_vals[0] * 1.0 - rate_vals[1] * 1.0) ,  # 2 years
        ]
        result = get_ZCB_vector(payment_dates, rate_vals, rate_dates)
        np.testing.assert_array_almost_equal(result, expected_values, decimal=3)

    def test_mismatched_length(self):
        """
        Test to make sure an error is thrown when the lengths of
        rate_vals and rate_dates are mismatched
        """
        payment_dates = [datetime(2025, 7, 1), datetime(2026, 1, 1), datetime(2026, 7, 1)]
        rate_vals = [0.05, 0.04, 0.03, 0.07, 0.08]
        rate_dates = [datetime(2025, 1, 1), datetime(2026, 1, 1), datetime(2027, 1, 1)]

        with pytest.raises(ValueError, match="Rate_vals should be the same length or one index less than rate_dates"):
            get_ZCB_vector(payment_dates, rate_vals, rate_dates)

    def test_rate_values_too_short(self):
        """
        Test the generation of a vector of zero-coupon bond (ZCB) values
        when the rate_vals list in one index too short.
        Verifies that a vector of ZCB values is correctly generated for
        given payment dates and discount rates.
        """
        payment_dates = [datetime(2026, 1, 1), datetime(2027, 1, 1), datetime(2028, 1, 1)]
        rate_vals = [0.05, 0.04]
        rate_dates = [datetime(2025, 1, 1), datetime(2026, 1, 1), datetime(2027, 1, 1)]

        expected_values = [
            np.exp(-rate_vals[0] * 1.0),  # 1 year
            np.exp(-rate_vals[0] * 1.0 - rate_vals[1] * 1.0) ,  # 2 years
            np.exp(-rate_vals[0] * 1.0 - rate_vals[1] * 2.0) # 3 years
        ]
        result = get_ZCB_vector(payment_dates, rate_vals, rate_dates)
        np.testing.assert_array_almost_equal(result, expected_values, decimal=3)

    def test_datetime64_vs_datetime(self):
        """
        Test that the generation of a vector of zero-coupon bond (ZCB) values
        is the same when arguments are type datetime or datetime64[D]
        """
        np_payment_dates = [datetime(2026, 1, 1), datetime(2027, 1, 1), datetime(2028, 1, 1)]
        np_rate_vals = [0.05, 0.04]
        np_rate_dates = [datetime(2025, 1, 1), datetime(2026, 1, 1), datetime(2027, 1, 1)]

        dt_payment_dates = [datetime(2026, 1, 1), datetime(2027, 1, 1), datetime(2028, 1, 1)]
        dt_rate_vals = [0.05, 0.04]
        dt_rate_dates = [datetime(2025, 1, 1), datetime(2026, 1, 1), datetime(2027, 1, 1)]

        np_ZCB_vector = get_ZCB_vector(np_payment_dates, np_rate_vals, np_rate_dates)
        dt_ZCB_vector = get_ZCB_vector(dt_payment_dates, dt_rate_vals, dt_rate_dates)

        np.testing.assert_almost_equal(np_ZCB_vector, dt_ZCB_vector, decimal=3)

if __name__ == '__main__':
    unittest.main()
