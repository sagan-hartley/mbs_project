import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

DISC_DAYS_IN_YEAR = 365.0

def convert_to_datetime(date):
    """
    Convert a string, numpy.datetime64, date, or datetime object to a Python datetime object.

    Parameters:
    - date (str, numpy.datetime64, or datetime): The date to be converted. 
                                                 If it is already a datetime object, it will be returned as-is.

    Returns:
    - datetime: The corresponding datetime object.
    
    Raises:
    - ValueError: If the input type cannot be converted to datetime.
    """
    if isinstance(date, str):
        # Convert a string to datetime, assuming a specific format (e.g., mm/dd/yyyy)
        return datetime.strptime(date, "%m/%d/%Y")
    elif isinstance(date, np.datetime64):
        # Convert numpy.datetime64 to a datetime object
        date_dt = date.astype(datetime)
        # Combine the date with the minimum time to ensure it's a full datetime
        date = datetime.combine(date_dt, datetime.min.time())  # Adds the HMS 00:00:00
        return date
    elif isinstance(date, datetime):
        # Return the input if it's already a datetime object
        return date
    else:
        raise ValueError("Input type could not be converted to datetime")

def convert_to_datetime64_array(dates):
    """
    Convert an array of dates to the 'datetime64[D]' format if it is not already.

    Parameters:
    dates (array-like): An array of date values to be converted.

    Returns:
    numpy.ndarray: The input converted to a 'datetime64[D]' array if needed.
    """
    # Check if the input is a numpy array
    if not isinstance(dates, np.ndarray):
        # Convert the input to a numpy array if it isn't already
        dates = np.array(dates)

    # Check if the array dtype is not 'datetime64[D]'
    if not np.issubdtype(dates.dtype, np.datetime64) or dates.dtype != 'datetime64[D]':
        # Convert the array to 'datetime64[D]' if it isn't already
        dates = np.array(dates, dtype='datetime64[D]')

    return dates

def get_ZCB_vector(payment_dates, rate_vals, rate_dates):
    """
    Calculate the discount factors for each payment date using a piecewise constant forward rate curve.

    The function integrates the given step function of rates over time to compute the cumulative discount factor 
    for each payment date, assuming that the rates remain constant between the corresponding rate_dates.

    Parameters:
    ----------
    payment_dates : list of datetime or datetime64[D]
        A list of future payment dates for which discount factors need to be calculated.
    
    rate_vals : list of float
        A list of discount rates (in decimal form) corresponding to the rate_dates.
        The rates are applied in a piecewise manner between the rate_dates.
    
    rate_dates : list of datetime or datetime64[D]
        A list of dates where the rates change. The first entry represents the market close date (i.e., the 
        starting point for the discounting process). Rates apply between consecutive dates.

    Returns:
    -------
    numpy.ndarray
        An array of discount factors for each payment date. If a payment date occurs before the first rate date, 
        the discount factor will be 0.0 for that payment.
    
    Assumptions:
    ------------
    - The rates are assumed to be constant between rate_dates.
    - If the payment date falls beyond the last rate_date, the last rate in rate_vals is used for discounting the 
      remaining period.
    - If the payment date is before the market close date, it returns a discount factor of 0.0 for that date.
    - If the payment date is on the market close date, it returns a discount factor of 1.0 for that date.
    """
    # Check to make sure the lengths of rate_vals and rate_dates are compatible
    if len(rate_vals) != len(rate_dates) and len(rate_vals) != len(rate_dates) - 1:
        raise ValueError("Rate_vals should be the same length or one index less than rate_dates")

    # If a rate is not user input for the last rate date, concatenate with the last rate value
    if len(rate_vals) == len(rate_dates) - 1:
        rate_vals = np.concatenate([rate_vals, [rate_vals[-1]]])

    # Initialize the result array
    ZCB_vector = np.zeros(len(payment_dates))

    # Define the market close date and convert rate_dates and payment_dates to numpy arrays for efficient operations
    # If inputs are datetime objects, convert them to type datetime64[D] for vectorization
    rate_dates = convert_to_datetime64_array(rate_dates)
    payment_dates = convert_to_datetime64_array(payment_dates)
    market_close_date = rate_dates[0]

    # Calculate time deltas (in years) from the first rate_date
    rate_time_deltas = (rate_dates - market_close_date).astype(float) / DISC_DAYS_IN_YEAR
    payment_time_deltas = (payment_dates - market_close_date).astype(float) / DISC_DAYS_IN_YEAR

    # Calculate the max payment date
    max_payment_date = np.max(payment_dates)

    # If the max payment date is beyond the last rate date add another rate value equal to the last value in rate_vals
    if max_payment_date > rate_dates[-1]:
        rate_time_deltas = np.concatenate((rate_time_deltas, [(max_payment_date - market_close_date).astype(float) / DISC_DAYS_IN_YEAR]))
        rate_vals = np.concatenate([rate_vals, [rate_vals[-1]]])

    # Calculate time differences between consecutive rate dates
    time_diffs = np.diff(rate_time_deltas)

    # Calculate the cumulative integral using the rate step function
    # Multiply rate_vals[:-1] by the time differences and compute the cumulative sum
    integral_values = np.concatenate(([0], np.cumsum(rate_vals[:-1] * time_diffs)))

    # Interpolate the integral values at the payment dates
    interpolated_integrals = np.interp(payment_time_deltas, rate_time_deltas, integral_values)

    # Calculate the discount factors (ZCB values)
    ZCB_vector = np.exp(-interpolated_integrals)

    # Handle cases where payment dates are before the market close date
    before_market_close = payment_time_deltas < 0
    if any(before_market_close):
        ZCB_vector[before_market_close] = 0

    return ZCB_vector

def discount_cash_flows(payment_dates, cash_flows, discount_rate_vals, discount_rate_dates):
    """
    Discounts a series of cash flows to their present value using variable discount rates.

    Parameters:
    -----------
    payment_dates : np.ndarray
        Array of datetime or datetime64[D} objects representing the payment dates.
    cash_flows : np.ndarray
        Array of cash flows corresponding to each payment date.
    discount_rates : np.ndarray
        Array of discount rates (as decimals) corresponding to each discount rate date.
    discount_rate_dates : np.ndarray
        Array of datetime or datetime64[D] objects representing the dates on which the discount rates apply.

    Returns:
    --------
    float:
        The present value of the cash flows.
    """
    # Check that payment dates and cash flows are the same length so the dot product computes correctly
    if len(payment_dates) != len(cash_flows):
        raise ValueError("Payment_dates and cash_flows should have the same length")
    
    # Calculate the ZCB vector using the market close date, payment dates, and discount rates
    zcb_values = get_ZCB_vector(payment_dates, discount_rate_vals, discount_rate_dates)

    # Discount the cash flows to their present value using the dot product of the cash flows and ZCB vectors
    present_value = np.dot(cash_flows, zcb_values)

    return present_value

def create_fine_dates_grid(market_close_date, maturity_years, interval_type='monthly'):
    """
    Create a finer grid of dates (monthly or weekly) from the market close date 
    to the bond maturity date.
    
    Parameters:
    -----------
    market_close_date : datetime
        The market close date (start date for the grid).
    maturity_years : float
        The number of years until bond maturity.
    interval_type : str
        The interval for the grid, either 'monthly' or 'weekly'.
    
    Returns:
    --------
    dates_grid : np.ndarray
        Array of dates from market close to bond maturity at the specified interval.
    """
    # Set the interval for the grid
    if interval_type == 'monthly':
        delta = relativedelta(months=1)
    elif interval_type == 'weekly':
        delta = relativedelta(weeks=1)
    else:
        raise ValueError("Invalid interval_type. Choose 'monthly' or 'weekly'.")

    # Calculate the maturity date
    maturity_date = market_close_date + relativedelta(years=maturity_years)

    # Initialize the dates grid starting from the market close date
    dates_grid = []
    current_date = market_close_date

    while current_date <= maturity_date:
        dates_grid.append(current_date)
        current_date += delta  # Increment by the chosen interval (monthly or weekly)
    
    # Convert the list to a numpy array
    return np.array(dates_grid)

def step_interpolate(dates_step, rates, query_dates):
    """
    Perform step interpolation to find rates corresponding to the query_dates.
    
    Parameters:
    - dates_step (array-like): Array of dates representing the step function's change points (must be sorted).
    - rates (array-like): Array of rates associated with each date in dates_step.
    - query_dates (array-like): Array of dates for which to find the associated rates.

    Returns:
    - interpolated_rates: Array of rates corresponding to the query_dates.

    Raises:
    - ValueError: 
        If there are duplicated dates in dates_step
        If dates_step is not sorted in ascending order
        If any query date is before the first element in dates_step
    """
    # Ensure inputs are numpy arrays, converting datetime objects to numpy datetime64
    dates_step = np.array(dates_step, dtype='datetime64[D]')
    rates = np.array(rates)
    query_dates = np.array(query_dates, dtype='datetime64[D]')

    # Check if dates_step is sorted and that there are no duplicate dates
    if not np.all(np.diff(dates_step) > np.timedelta64(0)):
        raise ValueError("dates_step must be sorted in ascending order without duplicate entires.")

    # Use searchsorted to find indices of the step dates less than or equal to query dates
    indices = np.searchsorted(dates_step, query_dates, side='right') - 1

    if np.any(indices < 0):
        raise ValueError("No query date should be before the first element in dates_step")

    # Return the corresponding rates
    interpolated_rates = rates[indices]
    
    return interpolated_rates

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

def calculate_dv(value_up, value_down, delta):
    """
    Calculate the dollar value (DV) for a given value change, typically used for DV01 calculations.

    Parameters:
    - value_up (float): value of the bond or instrument after a small yield increase.
    - value_down (float): value of the bond or instrument after a small yield decrease.
    - delta (float): The change in yield between the two values, typically in decimal form (e.g., 0.0001 for 1 bp).

    Returns:
    - float: The dollar value representing the value sensitivity per unit change in yield.

    Example:
    --------
    If value_up and value_down represent the bond values after a ±1 basis point change, and delta is 0.0001, 
    this function returns the DV01 (value sensitivity to a 1 bp yield change).
    
    Usage:
    ------
    dv01 = calculate_dv(value_up, value_down, 0.0001)
    """
    # Calculate the dollar value by finding the value difference and dividing by yield change
    dv = (value_up - value_down) / delta

    # Return the result as the DV, which represents sensitivity to yield changes
    return dv

def calculate_convexity(value, value_up, value_down, delta):
    """
    Calculate convexity using value changes for a small yield change.

    Convexity is a measure of the sensitivity of the instrument's duration to changes in interest rates.
    It accounts for the non-linear relationship between values and yields, providing
    a more accurate reflection of interest rate risk.

    Parameters:
    - value (float): Current value.
    - value_up (float): value if yields decrease by delta.
    - value_down (float): value if yields increase by delta.
    - delta (float): Small yield change in decimal form (e.g., 0.0001 for 1 basis point).

    Returns:
    - float: The convexity measure, a higher value suggests higher interest rate sensitivity.
    """
    # Calculate the convexity using the difference between the values when the yield is increased and decreased,
    # and the original value. This helps measure curvature in the value-yield relationship.
    convexity = (value_up + value_down - 2 * value) / (value * (delta ** 2))
    
    return convexity
