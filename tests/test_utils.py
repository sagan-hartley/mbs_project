import unittest
from datetime import datetime
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from utils import (
    DISC_DAYS_IN_YEAR,
    MAX_EXTRAPOLATE_YRS,
    days360,
    create_regular_dates_grid,
    years_from_reference,
    step_interpolate,
    integral_knots,
    zcbs_from_deltas,
    zcbs_from_dates
)

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

class TestCreateRegularDatesGrid(unittest.TestCase):
    """
    Unit tests for the create_regular_dates_grid function.
    """

    def test_daily_intervals(self):
        """
        Test the creation of a date grid with daily intervals.
        """
        start_date = datetime(2024, 1, 8)
        end_date = datetime(2024, 1, 18)
        result = create_regular_dates_grid(start_date, end_date, 'd')

        # Expected 11 dates: daily from Jan 8, 2024 through Jan 18, 2024 (inclusive)
        expected_dates = [start_date + relativedelta(days=i) for i in range(11)]
        expected = pd.DatetimeIndex(expected_dates)

        pd.testing.assert_index_equal(result, expected)
    
    def test_weekly_intervals(self):
        """
        Test the creation of a date grid with weekly intervals.
        """
        start_date = datetime(2024, 1, 8)
        end_date = datetime(2024, 3, 18)
        result = create_regular_dates_grid(start_date, end_date, 'w')

        # Expected dates: weekly from Jan 8, 2024 through Mar 18, 2024 (inclusive)
        expected_dates = [start_date + relativedelta(weeks=i) for i in range(11)]
        expected = pd.DatetimeIndex(expected_dates)

        pd.testing.assert_index_equal(result, expected)

    def test_monthly_intervals(self):
        """
        Test the creation of a date grid with monthly intervals.
        """
        start_date = datetime(2024, 1, 8)
        end_date = datetime(2025, 1, 8)
        result = create_regular_dates_grid(start_date, end_date, 'm')

        # Expected 13 dates: from Jan 8, 2024 through Jan 8, 2025 (inclusive)
        expected_dates = [start_date + relativedelta(months=i) for i in range(13)]
        expected = pd.DatetimeIndex(expected_dates)

        pd.testing.assert_index_equal(result, expected)

    def test_quarterly_intervals(self):
        """
        Test the creation of a date grid with quarterly intervals.
        """
        start_date = datetime(2024, 1, 8)
        end_date = datetime(2025, 1, 8)
        result = create_regular_dates_grid(start_date, end_date, 'q')

        # Expected 5 dates: quarterly from Jan 8, 2024 through Jan 8, 2025 (inclusive)
        expected_dates = [start_date + relativedelta(months=3 * i) for i in range(5)]
        expected = pd.DatetimeIndex(expected_dates)

        pd.testing.assert_index_equal(result, expected)

    def test_semi_annual_intervals(self):
        """
        Test the creation of a date grid with semi-annual intervals.
        """
        start_date = datetime(2024, 1, 8)
        end_date = datetime(2025, 1, 8)
        result = create_regular_dates_grid(start_date, end_date, 's')

        # Expected 3 dates: Jan 8, 2024; Jul 8, 2024; Jan 8, 2025
        expected_dates = [start_date + relativedelta(months=6 * i) for i in range(3)]
        expected = pd.DatetimeIndex(expected_dates)

        pd.testing.assert_index_equal(result, expected)

    def test_annual_intervals(self):
        """
        Test the creation of a date grid with annual intervals.
        """
        start_date = datetime(2024, 1, 8)
        end_date = datetime(2026, 1, 8)
        result = create_regular_dates_grid(start_date, end_date, 'a')

        # Expected 3 dates: Jan 8, 2024; Jan 8, 2025; Jan 8, 2026
        expected_dates = [start_date + relativedelta(years=i) for i in range(3)]
        expected = pd.DatetimeIndex(expected_dates)

        pd.testing.assert_index_equal(result, expected)

    def test_invalid_interval_type(self):
        """
        Test the function with an invalid interval type.
        """
        start_date = datetime(2024, 1, 8)
        end_date = datetime(2025, 1, 8)
        with self.assertRaises(ValueError):
            create_regular_dates_grid(start_date, end_date, 'daily')  # Invalid interval type

class TestYearsFromReference(unittest.TestCase):
    """
    Unit tests for the years_from_reference function.
    """

    def test_years_difference_same_day(self):
        """
        Test when the date in the grid is the same as the reference date.
        """
        ref_date = pd.Timestamp(datetime(2020, 1, 1))
        date_grid = pd.date_range(start=ref_date, periods=1)
        result = years_from_reference(ref_date, date_grid)
        expected = np.array([0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_years_difference_one_year(self):
        """
        Test when the date in the grid is one year after the reference date.
        """
        ref_date = pd.Timestamp(datetime(2020, 1, 1))
        date_grid = pd.date_range(start=ref_date, periods=1, freq='YE')  # Yearly frequency
        result = years_from_reference(ref_date, date_grid)
        expected = np.array([1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_years_difference_multiple_dates(self):
        """
        Test the function with multiple dates.
        """
        ref_date = pd.Timestamp(datetime(2020, 1, 1))
        date_grid = pd.to_datetime([
            datetime(2020, 1, 1),
            datetime(2021, 1, 1),
            datetime(2022, 1, 1),
            datetime(2020, 6, 30)
        ])
        result = years_from_reference(ref_date, date_grid)
        # We expect the results to be slightly off due to using the 365 day convention on a leap year
        expected = np.array([0.0, 1.00274, 2.00274, 0.49589])
        np.testing.assert_array_almost_equal(result, expected)

    def test_years_difference_no_dates(self):
        """
        Test when the date grid is empty.
        """
        ref_date = pd.Timestamp(datetime(2020, 1, 1))
        date_grid = pd.DatetimeIndex([])  # Create an empty DateTimeIndex
        result = years_from_reference(ref_date, date_grid)
        expected = np.array([])
        np.testing.assert_array_almost_equal(result, expected)

    def test_pandas_datetime_conversion(self):
        """
        Test when the input date_grid is not a Pandas DateTimeIndex
        """
        ref_date = pd.Timestamp(datetime(2020, 1, 1))
        date_grid = ([
            datetime(2020, 1, 1),
            datetime(2021, 1, 1),
            datetime(2022, 1, 1),
            datetime(2020, 6, 30)
        ])
        result = years_from_reference(ref_date, date_grid)
        # We expect the results to be slightly off due to using the 365 day convention on a leap year
        expected = np.array([0.0, 1.00274, 2.00274, 0.49589])
        np.testing.assert_array_almost_equal(result, expected)

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

class TestIntegralKnots(unittest.TestCase):
    """
    Unit tests for the integral_knots function.
    """

    def test_basic_integration(self):
        """
        Test the basic functionality of integral_knots with simple inputs.
        """
        date_grid = pd.date_range(start='2020-01-01', periods=4, freq='YE')
        rate_grid = np.array([0.01, 0.02, 0.03, 0.04])  # Example rates
        
        expected_years = np.array([0.0, 1.0, 2.0, 3.0, 3.0 + MAX_EXTRAPOLATE_YRS])  # Including extrapolated year
        expected_integral_vals = np.array([0.0, 0.01, 0.03, 0.06, 0.06 + 0.04 * MAX_EXTRAPOLATE_YRS])  # Results from manual integral calculation
        
        yrs, integral_vals = integral_knots(date_grid, rate_grid)

        np.testing.assert_array_almost_equal(yrs, expected_years)
        np.testing.assert_array_almost_equal(integral_vals, expected_integral_vals)

    def test_empty_inputs(self):
        """
        Test the function with empty date and rate grids.
        """
        date_grid = pd.DatetimeIndex([])  # Empty DateTimeIndex
        rate_grid = np.array([])  # Empty rate grid

        with self.assertRaises(IndexError):
            integral_knots(date_grid, rate_grid)  # Expecting an IndexError due to empty inputs

    def test_single_date(self):
        """
        Test the function with a single date and corresponding rate.
        """
        date_grid = pd.date_range(start='2020-01-01', periods=1)
        rate_grid = np.array([0.01])  # Only one rate

        expected_years = np.array([0.0, 0.0 + MAX_EXTRAPOLATE_YRS])  # Including extrapolated year
        expected_integral_vals = np.array([0.0, MAX_EXTRAPOLATE_YRS * 0.01])
        
        yrs, integral_vals = integral_knots(date_grid, rate_grid)

        np.testing.assert_array_almost_equal(yrs, expected_years)
        np.testing.assert_array_almost_equal(integral_vals, expected_integral_vals)

    def test_rate_extrapolation(self):
        """
        Test the function with rates and ensure that the last rate is used for extrapolation.
        """
        date_grid = pd.date_range(start='2020-01-01', periods=4, freq='YE')
        rate_grid = np.array([0.02, 0.03, 0.04, 0.05])  # Example rates

        yrs, integral_vals = integral_knots(date_grid, rate_grid)

        # Ensure the last year is as expected
        self.assertEqual(yrs[-1], 3.0 + MAX_EXTRAPOLATE_YRS)
        # Ensure the last integral value is calculated correctly based on the last rate
        self.assertAlmostEqual(integral_vals[-1], integral_vals[-2] + (MAX_EXTRAPOLATE_YRS * rate_grid[-1]))

    def test_integral_knots_unsorted_dates(self):
        """
        Test the integral_knots function with an unsorted date_grid.
        It should raise a ValueError.
        """
        # Create an unsorted date grid and a corresponding rate grid
        date_grid = pd.DatetimeIndex(['2023-12-31', '2023-01-01', '2023-06-30'])
        rate_grid = np.array([0.01, 0.02, 0.015])  # Different rates

        # Check that ValueError is raised
        with self.assertRaises(ValueError) as context:
            integral_knots(date_grid, rate_grid)

        self.assertEqual(str(context.exception), "date_grid is unsorted or duplicate dates exist.")

    def test_integral_knots_duplicate_dates(self):
        """
        Test the integral_knots function with duplicate values in date_grid.
        It should raise a ValueError.
        """
        # Create a date grid with duplicate values and a corresponding rate grid
        date_grid = pd.DatetimeIndex(['2023-12-31', '2023-12-31', '2023-06-30'])
        rate_grid = np.array([0.01, 0.02, 0.015])  # Different rates

        # Check that ValueError is raised
        with self.assertRaises(ValueError) as context:
            integral_knots(date_grid, rate_grid)

        self.assertEqual(str(context.exception), "date_grid is unsorted or duplicate dates exist.")

    def test_non_matching_lengths(self):
        """
        Test the function with date_grid longer than rate_grid.
        """
        date_grid = pd.date_range(start='2020-01-01', periods=5, freq='YE')
        rate_grid = np.array([0.01, 0.02])  # Shorter rate grid

        with self.assertRaises(ValueError):
            integral_knots(date_grid, rate_grid)  # Expecting a ValueError due to mismatched lengths

class TestZCBsFromDeltas(unittest.TestCase):
    """
    Unit tests for the zcbs_from_deltas function.
    """

    def test_zcbs_from_deltas_basic(self):
        """
        Test basic functionality with known inputs.
        """
        time_deltas = np.array([0.7, 1.0, 1.5])
        integral_vals = np.array([0.00, 0.04, 0.03])
        integral_time_deltas = np.array([0.0, 1.0, 2.0])
        
        expected = np.array([
            np.exp(-integral_vals[1] * 0.7),
            np.exp(-integral_vals[1]) ,  # Expected results calculated using a manual linear interpolation
            np.exp(-integral_vals[1] * 0.5 - integral_vals[2] * 0.5) 
        ])
        result = zcbs_from_deltas(time_deltas, integral_vals, integral_time_deltas)
        
        np.testing.assert_almost_equal(result, expected, decimal=6)

    def test_zcbs_from_deltas_single_value(self):
        """
        Test with a single value for each parameter.
        """
        time_deltas = np.array([1.0])
        integral_vals = np.array([0.03])
        integral_time_deltas = np.array([1.0])

        expected = np.array([0.97044553])
        result = zcbs_from_deltas(time_deltas, integral_vals, integral_time_deltas)

        np.testing.assert_almost_equal(result, expected, decimal=6)

    def test_zcbs_from_deltas_extrapolation(self):
        """
        Test the extrapolation behavior for out-of-range values.
        """
        time_deltas = np.array([0.5, 3.0])  # Including a value outside the integral_time_deltas range
        integral_vals = np.array([0.00, 0.03, 0.04])
        integral_time_deltas = np.array([0.0, 1.0, 2.0])

        expected = np.array([np.exp(-integral_vals[1] * 0.5),
                             np.exp(-integral_vals[2])])
        result = zcbs_from_deltas(time_deltas, integral_vals, integral_time_deltas)

        np.testing.assert_almost_equal(result, expected, decimal=6)

    def test_zcbs_from_deltas_invalid_input(self):
        """
        Test invalid inputs that should raise errors.
        """
        time_deltas = np.array([0.5])
        integral_vals = np.array([0.02])
        integral_time_deltas = np.array([])  # Empty array for integral_time_deltas

        with self.assertRaises(ValueError):
            zcbs_from_deltas(time_deltas, integral_vals, integral_time_deltas)

class TestZCBsFromDates(unittest.TestCase):
    """
    Unit tests for the zcbs_from_dates function.
    """

    def setUp(self):
        """
        Initialize common test data for rate dates and corresponding rate values,
        used across multiple test cases to ensure consistent input data.
        """
        # Define sample rate dates and corresponding rate values
        self.rate_dates = [datetime(2025, 1, 1), datetime(2026, 1, 1), datetime(2027, 1, 1)]
        self.rate_vals = np.array([0.05, 0.04, 0.03])  # Example interest rates

    def test_valid_dates(self):
        """
        Test with valid dates after the reference date to ensure the function
        calculates expected zero-coupon bond prices.
        """
        dates = [datetime(2025, 7, 1), datetime(2026, 1, 1), datetime(2026, 7, 1)]
        result = zcbs_from_dates(dates, self.rate_vals, self.rate_dates)

        # Expected output values based on manual zcb calculations
        # Note that exactly 181 days pass between 01/01 and 07/01 on a leap year
        # This is why the fraction 181/365 (DISC_DAYS_IN_YEAR) is used
        expected_values = np.array([
            np.exp(-self.rate_vals[0] * 181/DISC_DAYS_IN_YEAR),  # approx 1/2 year
            np.exp(-self.rate_vals[0] * 1.0) ,  # 1 year
            np.exp(-self.rate_vals[0] * 1.0 - self.rate_vals[1] * 181/DISC_DAYS_IN_YEAR) # approx 1 1/2 years
        ])

        np.testing.assert_almost_equal(result, expected_values, decimal=5)

    def test_dates_with_preceding_date(self):
        """
        Test with a date that precedes the reference date. The function should
        raise a ValueError in this scenario.
        """
        dates = pd.to_datetime(datetime(2024, 1, 1))  # Precedes the reference date
        with self.assertRaises(ValueError) as context:
            zcbs_from_dates(dates, self.rate_vals, self.rate_dates)
        self.assertEqual(str(context.exception),
                         'No element in dates may precede the reference date (the earliest rate date)')

    def test_edge_case_same_as_reference(self):
        """
        Test with a date that is exactly the reference date to ensure the function
        returns the expected zero-coupon bond price.
        """
        dates = pd.to_datetime(['2025-01-01'])  # Same as reference date
        result = zcbs_from_dates(dates, self.rate_vals, self.rate_dates)

        # Expected output value for ZCB price on reference date is just 1.0
        expected_values = np.array([1.0])

        np.testing.assert_almost_equal(result, expected_values, decimal=5)

if __name__ == '__main__':
    unittest.main()
