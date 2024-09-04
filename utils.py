import numpy as np
from datetime import datetime

def get_ZCB_vector(payment_dates, rate_vals, rate_dates):
    """
    Calculate the discount factors for each payment date given a set of rates and rate dates.

    Parameters:
    payment_dates (list of datetime): A list of payment dates.
    rate_vals (list of float): A list of discount rates (as decimals).
    rate_dates (list of datetime): A list of the dates corresponding to rate_vals where the first entry is the market close date.

    Returns:
    numpy.ndarray: An array of discount factors for the given payment dates.
    """
    # Convert rate_vals and rate_dates to numpy arrays for efficient operations
    rate_vals = np.array(rate_vals)
    rate_dates = np.array(rate_dates)

    # Initialize the result array
    discount_factors = np.zeros(len(payment_dates))

    for i, payment_date in enumerate(payment_dates):
        discount_factor = 1.0

        # Iterate over the rate dates
        for j, rate_date in enumerate(rate_dates):
            # Calculate the end date to use
            if payment_date < rate_date:
                # Use the payment date if it's before the current rate date
                end_date = payment_date
            else:
                # Otherwise, use the rate date
                end_date = rate_date
            
            # Calculate the time difference in years
            time_period = (end_date - rate_date).days / 365.0
            
            # Update the discount factor
            discount_factor *= np.exp(-rate_vals[j] * time_period)
            
            # If the end date is the payment date, break out
            if end_date == payment_date:
                break

        # Store the discount factor for the current payment date
        discount_factors[i] = discount_factor

    return discount_factors