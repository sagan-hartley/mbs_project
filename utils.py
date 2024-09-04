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
        for j in range(1, len(rate_dates)):
            # Determine the time period, starting from the previous rate date
            prev_date = rate_dates[j - 1]
            curr_date = min(rate_dates[j], payment_date)
            
            # Calculate the time difference in years
            time_period = (curr_date - prev_date).days / 365.0
            
            # Update the discount factor with the rate from the previous period
            discount_factor *= np.exp(-rate_vals[j - 1] * time_period)
            
            # Stop if we reach the payment date
            if curr_date == payment_date:
                break

         # If the payment date is beyond the last rate_date, discount using the last rate
        if payment_date > rate_dates[-1]:
            time_period = (payment_date - rate_dates[-1]).days / 365.0
            discount_factor *= np.exp(-rate_vals[-1] * time_period)

        # Store the discount factor for the current payment date
        discount_factors[i] = discount_factor

    return discount_factors