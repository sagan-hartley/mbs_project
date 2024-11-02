import unittest
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pytest
from utils import ( 
    convert_to_datetime,
    convert_to_datetime64_array,
    get_ZCB_vector,
    discount_cash_flows,
    create_fine_dates_grid,
    step_interpolate,
    days360,
    calculate_dv,
    calculate_convexity
)

class TestConvertToDatetime(unittest.TestCase):

    def test_string_to_datetime(self):
        """Test conversion from string to datetime"""
        input_date = "12/25/2024"
        expected_output = datetime(2024, 12, 25)
        self.assertEqual(convert_to_datetime(input_date), expected_output)

    def test_np_datetime64_to_datetime(self):
        """Test conversion from numpy.datetime64 to datetime"""
        input_date = np.datetime64('2024-12-25')
        expected_output = datetime(2024, 12, 25)
        self.assertEqual(convert_to_datetime(input_date), expected_output)

    def test_datetime_to_datetime(self):
        """Test passing a datetime object (should return the same object)"""
        input_date = datetime(2024, 12, 25)
        self.assertEqual(convert_to_datetime(input_date), input_date)

    def test_invalid_input(self):
        """Test raising an error for an unsupported input type"""
        with self.assertRaises(ValueError):
            convert_to_datetime(123)  # Invalid type, should raise ValueError

    def test_string_format_invalid(self):
        """Test error handling for incorrect date string format"""
        with self.assertRaises(ValueError):
            convert_to_datetime("2024/25/12")  # Wrong format

class TestConvertToDatetime64Array(unittest.TestCase):

    def test_array_already_datetime64(self):
        """Test when input array is already 'datetime64[D]'"""
        dates = np.array(['2024-01-01', '2024-02-01'], dtype='datetime64[D]')
        result = convert_to_datetime64_array(dates)
        # Assert that the result is still 'datetime64[D]'
        self.assertTrue(np.issubdtype(result.dtype, np.datetime64))
        self.assertEqual(result.dtype, 'datetime64[D]')
        np.testing.assert_array_equal(result, dates)

    def test_array_string_dates(self):
        """Test conversion from string array to 'datetime64[D]'"""
        dates = ['2024-01-01', '2024-02-01']
        result = convert_to_datetime64_array(dates)
        expected = np.array(['2024-01-01', '2024-02-01'], dtype='datetime64[D]')
        # Assert the conversion and dtype
        self.assertTrue(np.issubdtype(result.dtype, np.datetime64))
        self.assertEqual(result.dtype, 'datetime64[D]')
        np.testing.assert_array_equal(result, expected)

    def test_empty_array(self):
        """Test the case where the array is empty"""
        dates = np.array([], dtype='datetime64[D]')
        result = convert_to_datetime64_array(dates)
        # Assert that the result is still an empty array of dtype 'datetime64[D]'
        self.assertEqual(result.size, 0)
        self.assertEqual(result.dtype, 'datetime64[D]')

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

class TestDiscountCashFlows(unittest.TestCase):

    def setUp(self):
        # Set up common test data
        self.payment_dates = [
            np.datetime64('2025-02-13', 'D'),
            np.datetime64('2025-08-13', 'D'),
            np.datetime64('2026-02-13', 'D'),
            np.datetime64('2026-08-13', 'D'),
            np.datetime64('2027-02-13', 'D')
        ]
        self.cash_flows = np.array([10000, 10000, 10000, 10000, 110000])  # 10,000 every 6 months, final payment 110,000
        self.discount_rate_vals = np.array([0.02, 0.025, 0.03, 0.035, 0.04])  # Discount rates corresponding to each payment date
        self.discount_rate_dates = [
            np.datetime64('2025-02-13', 'D'),
            np.datetime64('2025-08-13', 'D'),
            np.datetime64('2026-02-13', 'D'),
            np.datetime64('2026-08-13', 'D'),
            np.datetime64('2027-02-13', 'D')
        ]

    def test_basic_functionality(self):
        """
        Test that the discount_cash_flows function calculates the present value correctly.
        """
        expected_present_value = 143420.645  # Replace with expected present value calculated manually or with a trusted tool
        present_value = discount_cash_flows(
            self.payment_dates,
            self.cash_flows,
            self.discount_rate_vals,
            self.discount_rate_dates
        )
        self.assertAlmostEqual(present_value, expected_present_value, places=2)

    def test_mismatched_array_lengths(self):
        """
        Test that the function raises a ValueError when the length of cash flows
        does not match the number of payment dates.
        """
        mismatched_cash_flows = np.array([10000, 10000, 10000, 10000])  # One less cash flow
        with self.assertRaises(ValueError):
            discount_cash_flows(
                self.payment_dates,
                mismatched_cash_flows,
                self.discount_rate_vals,
                self.discount_rate_dates
            )

class TestCreateFineDatesGrid(unittest.TestCase):
    """
    Unit tests for the create_fine_dates_grid function.
    """

    def test_monthly_intervals(self):
        """
        Test the creation of a date grid with monthly intervals.

        This test verifies that the function generates the correct number of dates
        when creating a monthly grid from the market close date to one year later.
        """
        market_close_date = datetime(2024, 1, 1)
        maturity_years = 1
        result = create_fine_dates_grid(market_close_date, maturity_years, 'monthly')

        # Expected 13 dates: from Jan 1, 2024 through Jan 1, 2025 (inclusive)
        expected_dates = [market_close_date + relativedelta(months=i) for i in range(13)]
        expected = np.array(expected_dates)

        np.testing.assert_array_equal(result, expected)

    def test_weekly_intervals(self):
        """
        Test the creation of a date grid with weekly intervals.

        This test verifies that the function generates the correct number of dates
        when creating a weekly grid for approximately 3 months.
        """
        market_close_date = datetime(2024, 1, 8)
        maturity_years = 1
        result = create_fine_dates_grid(market_close_date, maturity_years, 'weekly')

        # Expected dates starting from Jan 1, 2024, with weekly intervals
        expected_dates = [market_close_date + relativedelta(weeks=i) for i in range(53)]  # Approx 53 weeks
        expected = np.array(expected_dates)

        np.testing.assert_array_equal(result, expected)

    def test_invalid_interval_type(self):
        """
        Test the function with an invalid interval type.

        This test verifies that the function raises a ValueError when provided with
        an invalid interval type.
        """
        market_close_date = datetime(2024, 1, 1)
        maturity_years = 1
        with self.assertRaises(ValueError):
            create_fine_dates_grid(market_close_date, maturity_years, 'daily')  # Invalid interval type

class TestStepInterpolate(unittest.TestCase):
    """
    Unit tests for the step_interpolate function.
    """

    def test_basic_interpolation(self):
        """Basic case where the query dates fall within the range of dates_step"""
        dates_step = np.array(['2024-01-01', '2024-06-01', '2024-12-01'], dtype='datetime64[D]')
        rates = np.array([0.02, 0.03, 0.04])
        query_dates = np.array(['2024-02-01', '2024-07-01'], dtype='datetime64[D]')
        
        expected_rates = np.array([0.02, 0.03])  # Expected rates for the query dates
        result = step_interpolate(dates_step, rates, query_dates)
        np.testing.assert_array_equal(result, expected_rates)

    def test_unsorted_dates_step(self):
        """Test for unsorted dates_step (should raise ValueError)"""
        dates_step = np.array(['2024-06-01', '2024-01-01', '2024-12-01'], dtype='datetime64[D]')
        rates = np.array([0.03, 0.02, 0.04])
        query_dates = np.array(['2024-02-01', '2024-07-01'], dtype='datetime64[D]')

        with self.assertRaises(ValueError):
            step_interpolate(dates_step, rates, query_dates)

    def test_duplicate_dates_step(self):
        """Test for duplicate values contained in dates_step (should raise ValueError)"""
        dates_step = np.array(['2024-06-01', '2025-01-01', '2025-01-01'], dtype='datetime64[D]')
        rates = np.array([0.03, 0.02, 0.04])
        query_dates = np.array(['2024-02-01', '2024-07-01'], dtype='datetime64[D]')

        with self.assertRaises(ValueError):
            step_interpolate(dates_step, rates, query_dates)

    def test_query_date_before_first_date(self):
        """Test for query dates that are before the first element in dates_step"""
        dates_step = np.array(['2024-01-01', '2024-06-01', '2024-12-01'], dtype='datetime64[D]')
        rates = np.array([0.02, 0.03, 0.04])
        query_dates = np.array(['2023-12-01'], dtype='datetime64[D]')

        with self.assertRaises(ValueError):
            step_interpolate(dates_step, rates, query_dates)

    def test_query_date_equal_to_step_dates(self):
        """Test for query dates that exactly match dates in dates_step"""
        dates_step = np.array(['2024-01-01', '2024-06-01', '2024-12-01'], dtype='datetime64[D]')
        rates = np.array([0.02, 0.03, 0.04])
        query_dates = np.array(['2024-01-01', '2024-12-01'], dtype='datetime64[D]')

        expected_rates = np.array([0.02, 0.04])  # Expected rates
        result = step_interpolate(dates_step, rates, query_dates)
        np.testing.assert_array_equal(result, expected_rates)

    def test_query_dates_after_last_step_date(self):
        """Test for query dates that are after the last element in dates_step"""
        dates_step = np.array(['2024-01-01', '2024-06-01', '2024-12-01'], dtype='datetime64[D]')
        rates = np.array([0.02, 0.03, 0.04])
        query_dates = np.array(['2025-01-01'], dtype='datetime64[D]')

        expected_rates = np.array([0.04])  # Should take the last rate
        result = step_interpolate(dates_step, rates, query_dates)
        np.testing.assert_array_equal(result, expected_rates)

class TestDays360(unittest.TestCase):
    """
    Unit tests for the days360 function.
    """

    def test_same_month(self):
        """Test case where both dates are in the same month."""
        d1 = datetime(2024, 10, 1)
        d2 = datetime(2024, 10, 15)
        self.assertEqual(days360(d1, d2), 14)
    
    def test_full_month(self):
        """Test case where the second date is the last day of the month."""
        d1 = datetime(2024, 10, 1)
        d2 = datetime(2024, 10, 30)
        self.assertEqual(days360(d1, d2), 29)

    def test_first_day_is_31(self):
        """Test case where the first date is on the 31st."""
        d1 = datetime(2024, 10, 31)
        d2 = datetime(2024, 11, 20)
        self.assertEqual(days360(d1, d2), 20)

    def test_second_day_is_31(self):
        """Test case where the first date is on the 31st."""
        d1 = datetime(2024, 10, 20)
        d2 = datetime(2024, 9, 30)
        d3 = datetime(2024, 8, 31)
        d4 = datetime(2024, 10, 31)
        self.assertEqual(days360(d1, d4), 10)
        self.assertEqual(days360(d2, d4), 31)
        self.assertEqual(days360(d3, d4), 61)

    def test_different_months(self):
        """Test case where the dates are in different months."""
        d1 = datetime(2024, 10, 20)
        d2 = datetime(2024, 11, 15)
        d3 = datetime(2024, 12, 11)
        self.assertEqual(days360(d1, d2), 25)
        self.assertEqual(days360(d1, d3), 51)

    def test_different_years(self):
        """Test case where the dates are in different years."""
        d1 = datetime(2024, 10, 20)
        d2 = datetime(2025, 11, 15)
        self.assertEqual(days360(d1, d2), 385)

    def test_invalid_date_order(self):
        """Test case where the first date is later than the second date."""
        d1 = datetime(2024, 10, 1)
        d2 = datetime(2024, 9, 30)
        with self.assertRaises(AssertionError):
            days360(d1, d2)

class TestCalculateDV(unittest.TestCase):
    def test_positive_dv(self):
        """Test DV calculation with a positive price change."""
        price_up = 100.05
        price_down = 99.95
        delta = 0.0001
        expected_dv = (price_up - price_down) / delta
        self.assertAlmostEqual(calculate_dv(price_up, price_down, delta), expected_dv, places=6)

    def test_negative_dv(self):
        """Test DV calculation with a negative price change."""
        price_up = 99.95
        price_down = 100.05
        delta = 0.0001
        expected_dv = (price_up - price_down) / delta
        self.assertAlmostEqual(calculate_dv(price_up, price_down, delta), expected_dv, places=6)

    def test_zero_delta(self):
        """Test DV calculation when delta is zero, expecting a ZeroDivisionError."""
        price_up = 100.05
        price_down = 99.95
        delta = 0
        with self.assertRaises(ZeroDivisionError):
            calculate_dv(price_up, price_down, delta)

    def test_zero_price_change(self):
        """Test DV calculation when there is no price change, expecting DV of 0."""
        price_up = 100.0
        price_down = 100.0
        delta = 0.0001
        expected_dv = 0.0
        self.assertAlmostEqual(calculate_dv(price_up, price_down, delta), expected_dv, places=6)

    def test_large_delta(self):
        """Test DV calculation with a larger delta."""
        price_up = 105.0
        price_down = 95.0
        delta = 0.01
        expected_dv = (price_up - price_down) / delta
        self.assertAlmostEqual(calculate_dv(price_up, price_down, delta), expected_dv, places=6)

class TestCalculateConvexity(unittest.TestCase):
    def test_basic_convexity(self):
        """Test the convexity calculation with typical price changes."""
        price = 100
        price_up = 100.05
        price_down = 99.95
        delta_y = 0.0001
        result = calculate_convexity(price, price_up, price_down, delta_y)
        expected_result = (100.05 + 99.95 - 2 * 100) / (100 * (0.0001 ** 2))
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_zero_convexity(self):
        """Test convexity when price changes are symmetric, indicating no curvature."""
        price = 100
        price_up = 100
        price_down = 100
        delta_y = 0.0001
        result = calculate_convexity(price, price_up, price_down, delta_y)
        self.assertAlmostEqual(result, 0, places=6)

    def test_small_delta_y(self):
        """Test the convexity with a very small delta_y to observe handling of potential large values."""
        price = 100
        price_up = 100.001
        price_down = 99.999
        delta_y = 1e-8
        result = calculate_convexity(price, price_up, price_down, delta_y)
        expected_result = (100.001 + 99.999 - 2 * 100) / (100 * (1e-8 ** 2))
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_large_delta_y(self):
        """Test with a large delta_y to confirm correct scaling in convexity calculation."""
        price = 100
        price_up = 101
        price_down = 99
        delta_y = 0.01
        result = calculate_convexity(price, price_up, price_down, delta_y)
        expected_result = (101 + 99 - 2 * 100) / (100 * (0.01 ** 2))
        self.assertAlmostEqual(result, expected_result, places=6)

if __name__ == "__main__":
    unittest.main()
