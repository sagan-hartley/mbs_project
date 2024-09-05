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
    - If the payment date is before the first rate_date, it returns a discount factor of 0 for that date.
    """
    # Initialize the result array
    ZCB_vector = np.zeros(len(payment_dates))

    # Define the market close date
    market_close_date = rate_dates[0] 

    for i, payment_date in enumerate(payment_dates):

        # If the payment date is before the market close date, it cannot be discounted, so return 0 
        if payment_date < market_close_date:
            ZCB_vector[i] = 0

        # If the payment date is the market close date just return 1.0 and break
        elif payment_date == market_close_date:
            ZCB_vector[i] = 1.0

        else:
            # Initialize the integral of the rate value step function
            integral = 0.0

            # Iterate over the rate dates
            for j in range(1, len(rate_dates)):
                # Determine the time period, starting from the previous rate date
                prev_date = rate_dates[j - 1]
                curr_date = min(rate_dates[j], payment_date)
            
                # Calculate the time difference in years
                time_delta = (curr_date - prev_date).days / 365.0
            
                # Update the integral with the rate from the previous period
                integral += rate_vals[j - 1] * time_delta
            
                # Stop if we reach the payment date
                if curr_date == payment_date:
                    break

            # If the payment date is beyond the last rate_date, discount using the last rate
            if payment_date > rate_dates[-1]:
                time_delta = (payment_date - rate_dates[-1]).days / 365.0
                integral += rate_vals[-1] * time_delta

            # Store the discount factor for the current payment date
            ZCB_vector[i] = np.exp(-integral)

    return ZCB_vector