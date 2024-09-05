import numpy as np
from datetime import datetime

def get_ZCB_vector(payment_dates, rate_vals, rate_dates):
    """
    Calculate the discount factors for each payment date using a piecewise constant forward rate curve.

    The function integrates the given step function of rates over time to compute the cumulative discount factor 
    for each payment date, assuming that the rates remain constant between the corresponding rate_dates.

    Parameters:
    ----------
    payment_dates : list of datetime
        A list of future payment dates for which discount factors need to be calculated.
    
    rate_vals : list of float
        A list of discount rates (in decimal form) corresponding to the rate_dates.
        The rates are applied in a piecewise manner between the rate_dates.
    
    rate_dates : list of datetime
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
    assert len(rate_vals) == len(rate_dates) or len(rate_vals) == len(rate_dates) - 1, \
        "Rate_vals is not the same length or one index less than rate_dates"

    # If a rate is not user input for the last rate date, concatenate with the last rate value
    if len(rate_vals) == len(rate_dates) - 1:
        rate_vals = np.concatenate([rate_vals, [rate_vals[-1]]])

    # Initialize the result array
    ZCB_vector = np.zeros(len(payment_dates))

    # Define the market close date
    market_close_date = rate_dates[0] 

    # Convert rate_dates and payment_dates to numpy arrays for efficient operations
    rate_dates = np.array(rate_dates)
    payment_dates = np.array(payment_dates)

    # Calculate time deltas (in years) from the first rate_date
    rate_time_deltas = np.array([(rd - market_close_date).days / 365.0 for rd in rate_dates])
    payment_time_deltas = np.array([(pd - market_close_date).days / 365.0 for pd in payment_dates])

    # Calculate the max payment date
    max_payment_date = np.max(payment_dates)

    # If the max payment date is beyond the last rate date add another rate value equal to the last value in rate_vals
    if max_payment_date > rate_dates[-1]:
        rate_time_deltas = np.concatenate((rate_time_deltas, [(max_payment_date - market_close_date).days /365]))
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