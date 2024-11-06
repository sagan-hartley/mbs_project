from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd

DISC_DAYS_IN_YEAR = 365.0
MAX_EXTRAPOLATE_YRS = 100.0

def days360(d1, d2):
    """
    Calculate the number of days between two dates using the 360-day year convention.
    
    Parameters:
    d1 (datetime): The first date.
    d2 (datetime): The second date, which should be later than or equal to the first date.
    
    Returns:
    int: The number of days between the two dates, using the 30/360 day count convention.
    """
    assert d1 <= d2, "The first date must be before or equal to the second date."
    
    # Adjust day for 30/360 convention
    d1_day = min(d1.day, 30)
    d2_day = min(d2.day, 30) if d1_day < 30 else d2.day  # Adjust d2 only if d1 < 30

    # Calculate the number of days using 360-day year convention
    return (d2.year - d1.year) * 360 + (d2.month - d1.month) * 30 + (d2_day - d1_day)

def create_regular_dates_grid(start_date, end_date, frequency='m'):
    """
    Create a finer grid of dates (e.g., daily, weekly, monthly) from the market close date 
    to the bond maturity date.

    Parameters:
    -----------
    start_date : datetime
        The market close date (start date for the grid).
    end_date : datetime
        The bond maturity date (end date for the grid).
    frequency : str, default 'm'
        The interval for the grid. Options are:
        - 'd': daily
        - 'w': weekly
        - 'm': monthly
        - 'q': quarterly
        - 's': semi-annually
        - 'a': annually
    
    Returns:
    --------
    dates_grid : np.ndarray
        Array of dates from market close to bond maturity at the specified interval.

    Raises:
    -------
    ValueError
        If the input frequency string is not recognized.
    """
    # Ensure the frequency string is lowercase
    freq = frequency.lower()

    # Set the DateOffset for the grid
    if freq == 'd':
        freq = pd.DateOffset(days=1)
    elif freq == 'w':
        freq = pd.DateOffset(weeks=1)
    elif freq == 'm':
        freq = pd.DateOffset(months=1)
    elif freq == 'q':
        freq = pd.DateOffset(months=3)
    elif freq == 's':
        freq = pd.DateOffset(months=6)
    elif freq == 'a':
        freq = pd.DateOffset(years=1)
    else:
        raise ValueError('Input frequency string is not recognized. Use "d", "w", "m", "q", "s", or "a".')

    # Generate the date range
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)

    return date_range

def years_from_reference(ref_date, date_grid):
    """
    Calculate the number of years from a reference date for each date in a given date grid.

    Parameters:
    -----------
    ref_date : datetime
        The reference date from which the year difference is calculated.
    
    date_grid : np.ndarray
        An array of dates for which the year differences are computed.
    
    Returns:
    --------
    np.ndarray
        An array of the number of years between the reference date and each date in the date grid,
        calculated by dividing the number of days by the number of days in a year.

    Notes:
    ------
    The calculation assumes that the number of days in a year is defined by the constant
    `DISC_DAYS_IN_YEAR`. Adjust this constant as necessary for leap years or other definitions
    of a year if needed.

    Examples:
    ---------
    >>> import numpy as np
    >>> from datetime import datetime
    >>> ref = datetime(2020, 1, 1)
    >>> dates = np.array([datetime(2021, 1, 1), datetime(2022, 1, 1)])
    >>> years_from_reference(ref, dates)
    array([1.0, 2.0])
    """
    # Convert the ref_date and date_grid to Pandas datetime to allow .days to be applied in a vectorized fashion
    ref_date = pd.to_datetime(ref_date)
    date_grid = pd.to_datetime(date_grid)

    # Calculate the difference in days between each date in date_grid and the reference date
    # and convert it to years by dividing by the number of days in a year.
    return np.array((date_grid - ref_date).days / DISC_DAYS_IN_YEAR)

def integer_months_from_reference(start_date, end_date):
    """
    Calculate the integer number of months between two dates.

    Parameters
    ----------
    start_date : datetime
        The starting date.
    end_date : datetime
        The ending date.

    Returns
    -------
    int
        The number of whole months from start_date to end_date.
    """
    # Calculate the difference in years and months between the two dates
    delta = relativedelta(end_date, start_date)
    
    # Convert the delta to total months
    term_in_months = delta.years * 12 + delta.months

    return term_in_months

def step_interpolate(dates_step, rates, query_dates):
    """
    Perform step interpolation to find rates corresponding to the query_dates.

    Parameters:
    - dates_step (array-like or pd.DatetimeIndex): Array of dates representing the step function's change points (must be sorted).
    - rates (array-like): Array of rates associated with each date in dates_step.
    - query_dates (array-like or pd.DatetimeIndex): Array of dates for which to find the associated rates.

    Returns:
    - interpolated_rates: Array of rates corresponding to the query_dates.

    Raises:
    - ValueError: 
        If there are duplicated dates in dates_step.
        If dates_step is not sorted in ascending order.
        If any query date is before the first element in dates_step.
    """
    # Convert to DatetimeIndex if inputs are not already in datetime format
    dates_step = pd.to_datetime(dates_step)
    query_dates = pd.to_datetime(query_dates)
    
    # Ensure rates are in a numpy array
    rates = np.array(rates)
    
    # Check if dates_step is sorted and that there are no duplicate dates
    if not dates_step.is_monotonic_increasing or dates_step.has_duplicates:
        raise ValueError("dates_step must be sorted in ascending order without duplicate entries.")

    # Convert DatetimeIndex to numpy datetime64 for efficient processing
    dates_step_np = dates_step.values
    query_dates_np = query_dates.values

    # Use searchsorted to find indices of the step dates less than or equal to query dates
    indices = np.searchsorted(dates_step_np, query_dates_np, side='right') - 1

    if np.any(indices < 0):
        raise ValueError("No query date should be before the first element in dates_step")

    # Return the corresponding rates
    interpolated_rates = rates[indices]

    return interpolated_rates

def integral_knots(date_grid, rate_grid):
    """
    Calculate the integral knots and their corresponding integral values 
    based on the provided date grid and rate grid.

    Parameters:
    -----------
    date_grid : pd.DatetimeIndex
        An array of dates representing the time periods. This must be sorted in 
        ascending order for the calculations to be valid.
    rate_grid : np.ndarray
        An array of rates corresponding to the time periods. The length of this 
        array must match the length of the date_grid.

    Returns:
    --------
    yrs : np.ndarray
        An array of years calculated from the reference date, where each entry 
        represents the number of years from the start date to the respective date 
        in the date grid.
    integral_vals : np.ndarray
        An array of cumulative integral values calculated based on the rates. The 
        first value is initialized to 0.0 to represent the integral at the start.

    Raises:
    -------
    ValueError
        If the lengths of date_grid and rate_grid are not matching
        If the date_grid is not sorted in ascending order with no duplicates.

    Notes:
    ------
    The integral values are calculated as the cumulative sum of the product of 
    the time differences (in years) and the rates. The last value is extrapolated 
    using a predefined constant to allow future interpolation with dates that occur 
    after the maximum date in date_grid.
    """
    # Check if date_grid and rate_grid have the same length; raise an error if not
    if len(date_grid) != len(rate_grid):
        raise ValueError("date_grid and rate_grid must have the same length.")

    # Calculate the number of years from the reference date for each date in date_grid
    yrs_from_reference = years_from_reference(date_grid[0], date_grid)

    # Extrapolate an "end date" using a predefined constant to allow future interpolation
    # with dates that occur after the maximum date in date_grid.
    yrs_from_reference = np.append(yrs_from_reference, yrs_from_reference[-1] + MAX_EXTRAPOLATE_YRS)

    # Calculate the time differences between consecutive years in the yrs_from_reference array
    time_deltas = np.diff(yrs_from_reference)

    # Check if the date_grid is sorted; raise an error if it is not
    # The date_grid needs to be sorted to ensure integral values are properly calucalted with np.cumsum
    # The date_grid is only sorted with no duplicates iff all time_deltas are strictly psitive
    if np.any(time_deltas <= 0):
        raise ValueError("date_grid is unsorted or duplicate dates exist.")

    # Calculate the cumulative integral values based on time deltas and rate grid
    integral_vals = np.cumsum(time_deltas * rate_grid)

    # Insert the initial value for the integral (0.0) at the start of the array
    integral_vals = np.insert(integral_vals, 0, 0.0)

    return yrs_from_reference, integral_vals

def zcbs_from_deltas(time_deltas, integral_vals, integral_time_deltas):
    """
    Calculate the zero-coupon bond values from time deltas and integral values.

    This function uses linear interpolation to find the integral values corresponding to the 
    provided time deltas and then computes the zero-coupon bond values by applying 
    the exponential function.

    Parameters:
    -----------
    time_deltas : np.ndarray
        An array of time deltas for which the zero-coupon bond values are to be computed. 
    integral_vals : np.ndarray
        An array of integral values corresponding to the integral time deltas. 
    integral_time_deltas : np.ndarray
        An array of integral time deltas used for interpolation.

    Returns:
    --------
    np.ndarray
        An array of zero-coupon bond values calculated from the interpolated integral values.

    Raises:
    -------
    ValueError
        If integral_vals and integral_time_deltas are not the same length.

    Notes:
    ------
    The zero-coupon bond values are calculated using the formula:
    ZCB = exp(-I) where I is the interpolated integral value.
    This represents the present value of receiving $1 at maturity.
    """
    # Check if integral_vals and integral_time_deltas have the same length
    if len(integral_vals) != len(integral_time_deltas):
        raise ValueError("integral_vals and integral_time_deltas must have the same length.")
    
    # Interpolate the integral values corresponding to the given time deltas
    interepolated_integral_vals = np.interp(time_deltas, integral_time_deltas, integral_vals)
    
    # Calculate the zero-coupon bond values using the exponential function
    return np.exp(-interepolated_integral_vals)

def zcbs_from_dates(dates, rate_vals, rate_dates):
    """
    Calculate zero-coupon bond (ZCB) values based on the provided dates, 
    rate values, and corresponding rate dates.

    Parameters:
    -----------
    dates : np.ndarray or pd.DatetimeIndex
        An array of dates for which ZCB values are calculated.
    rate_vals : np.ndarray
        An array of rate values corresponding to the rate dates.
    rate_dates : np.ndarray or pd.DatetimeIndex
        An array of dates corresponding to the rate values.

    Returns:
    --------
    np.ndarray
        An array of calculated ZCB values.

    Raises:
    -------
    ValueError
        If any element in dates precedes the reference date (the earliest rate date).
    """
    # Define the reference date as the earliest date in the rate_dates
    reference_date = rate_dates[0]

    # Calculate the time deltas (in years) from the reference date to the provided dates
    time_deltas = years_from_reference(reference_date, dates)

    # Check if any date is before the reference date; if so, raise an error
    if np.any(time_deltas < 0):
        raise ValueError('No element in dates may precede the reference date (the earliest rate date)')

    # Calculate the integral time deltas and corresponding integral values
    integral_time_deltas, integral_vals = integral_knots(rate_dates, rate_vals)

    # Calculate ZCB values using the time deltas and the integral values
    return zcbs_from_deltas(time_deltas, integral_vals, integral_time_deltas)
