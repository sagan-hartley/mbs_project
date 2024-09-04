import numpy as np
from datetime import datetime

def calculate_ZCB_values(rates, start_dates, end_dates):
    """
    Calculate the discount factors over periods given rates.
    Parameters:
    rates (np.ndarray): The discount rates (as decimals).
    start_dates (np.ndarray): The start dates.
    end_dates (np.ndarray): The end dates.
    Returns:
    np.ndarray: The discount factors.
    """
    # Ensure that rates are a ndarray so that np.exp will not throw a TypeError
    rates = np.array(rates)

    # Calculate the time difference in years as a float and return the exponential using the given rates
    time_diffs = np.array([(end_date - start_date).days / 365.0 for end_date, start_date in zip(end_dates, start_dates)])

    return np.exp(-rates * time_diffs)

def get_ZCB_vector(settle_date, payment_dates, rate_vals, rate_dates):
    """
    Calculate the discount factors for each payment date given a set of rates and rate dates.
    Parameters:
    settle_date (datetime): The settlement date.
    payment_dates (list of datetime): A list of payment dates.
    rate_vals (list of float): A list of discount rates (as decimals).
    rate_dates (list of datetime): A list of the corresponding dates to rate_vals.
    Returns:
    numpy.ndarray: An array of discount factors for the given payment dates.
    """
    # Convert rate_vals and rate_dates to numpy arrays for efficient operations
    rate_vals = np.array(rate_vals)
    rate_dates = np.array(rate_dates)

    # Initialize the result array
    zcb_vector = np.zeros(len(payment_dates))

    for i, payment_date in enumerate(payment_dates):
        discount_factor = 1.0

        # Iterate over the rate dates
        for j, rate_date in enumerate(rate_dates):
            # Calculate the time period to use
            if j == 0:
                # Special case when j equals 0, the previous rate date is set to the settle date
                # and the end date is the current rate date
                prev_date = settle_date
                end_date = rate_date
            elif payment_date < rate_date:
                # For the end date, use the payment date if it's before the current rate date
                end_date = payment_date
            else:
                # Otherwise, use the rate date
                end_date = rate_date

            # Calculate the time difference in years
            time_period = (end_date - prev_date).days / 365.0

            # Update the discount factor
            discount_factor *= np.exp(-rate_vals[j] * time_period)

            # If the end date is the payment date, break out, else update the previous date
            if end_date == payment_date:
                break
            else: 
                prev_date = rate_date

        # Store the discount factor for the current payment date
        zcb_vector[i] = discount_factor

    return zcb_vector
