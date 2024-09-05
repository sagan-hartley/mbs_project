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
    # Initialize the result array
    ZCB_vector = np.zeros(len(payment_dates))

    for i, payment_date in enumerate(payment_dates):
        integral = 0.0

        # Iterate over the rate dates
        for j in range(1, len(rate_dates)):
            # Determine the time period, starting from the previous rate date
            prev_date = rate_dates[j - 1]
            curr_date = min(rate_dates[j], payment_date)
            
            # Calculate the time difference in years
            time_period = (curr_date - prev_date).days / 365.0
            
            # Update the integral with the rate from the previous period
            integral += rate_vals[j - 1] * time_period
            
            # Stop if we reach the payment date
            if curr_date == payment_date:
                break

         # If the payment date is beyond the last rate_date, discount using the last rate
        if payment_date > rate_dates[-1]:
            time_period = (payment_date - rate_dates[-1]).days / 365.0
            integral += rate_vals[-1] * time_period

        # Store the discount factor for the current payment date
        ZCB_vector[i] = np.exp(-integral)

    return ZCB_vector