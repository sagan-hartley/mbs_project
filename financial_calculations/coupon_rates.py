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
    forward_curve : ForwardCurve
        An instance of the ForwardCurve class that contains discount rate dates and rates.

    Returns:
    --------
    float:
        The coupon rate required to produce a par bond.
    
    Raises:
    -------
    ValueError:
        If the start date is before the market close date.
    """
    # Access the market close date from the forward curve
    market_close_date = forward_curve.market_close_date

    # Validate start date
    if start_date < market_close_date:
        raise ValueError("Start date is not on or after the market close date.")

    # Generate the bond payment dates by adding multiples of 6-month periods
    payment_dates = [start_date + relativedelta(months=6 * i) for i in range(PMTS_PER_YEAR * maturity_years + 1)]

    # Calculate the discount factors for the bond
    discount_factors = get_ZCB_vector(payment_dates, forward_curve.rates, forward_curve.dates)

    # Define some quantities useful for the coupon rate calculation
    annuity = (1 / PMTS_PER_YEAR) * np.sum(discount_factors[1:])
    initial_discount = discount_factors[0]
    final_discount = discount_factors[-1]

    # Calculate the coupon rate
    coupon_rate = (initial_discount - final_discount) / annuity

    return coupon_rate
