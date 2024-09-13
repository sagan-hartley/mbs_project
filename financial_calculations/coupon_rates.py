import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from utils import (
    get_ZCB_vector
)
from financial_calculations.bond_cash_flows import (
    PMTS_PER_YEAR
)

def calculate_coupon_rate(start_date, maturity_years, forward_curve):
    """
    Calculate the coupon rate required to produce a par price for a bond.
    
    Parameters:
    -----------
    start_date : datetime or datetime64[D]
        The start date of the bond.
    maturity_years : int
        The maturity years of the bond.
    forward_curve : tuple
        A tuple containing two elements:
        - disc_rate_dates (np.ndarray): Array of discount rate dates.
        - disc_rates (np.ndarray): Array of discount rates corresponding to each discount rate date.

    Returns:
    --------
    float:
        The coupon rate required to produce a par price for the bond.
    
    Raises:
    -------
    ValueError:
        If the start date is before the market close date.
    """
    disc_rate_dates, disc_rates = forward_curve
    market_close_date = disc_rate_dates[0]

    # If the market close date is input as a datetime64[D] type, convert to datetime to ensure compatibility with the forward curve data
    if isinstance(market_close_date, np.datetime64): 
        market_close_date = market_close_date.astype(datetime)

    # Validate start date
    if start_date < market_close_date:
        raise ValueError("Start date is not on or after the market close date.")

    else:
        # Generate the bond payment dates by adding multiples of 6-month periods
        payment_dates = [start_date + relativedelta(months=6 * i) for i in range(PMTS_PER_YEAR * maturity_years + 1)]

        # Calculate the discount factors for the bond
        discount_factors = get_ZCB_vector(payment_dates, disc_rates, disc_rate_dates)

        # Define some quantities useful for the coupon rate calculationb
        annuity = (1 / PMTS_PER_YEAR) * np.sum(discount_factors[1:])
        initial_discount = discount_factors[0] 
        final_discount = discount_factors[-1]   

        # Calculate the coupon rate
        coupon_rate = (initial_discount - final_discount) / annuity

        return coupon_rate
