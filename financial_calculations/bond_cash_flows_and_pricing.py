import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from utils import (
    get_ZCB_vector
)

SPOT_SEMI_BOND_HALF_YEAR = 183

def create_spot_semi_bond_cash_flows(market_close_date, balance, coupon, maturity_years):
    """
    Creates the cash flow schedule for a spot semiannual bond with a bullet repayment structure.

    Parameters:
    -----------
    market_close_date : str or datetime  or datetime64[D]
        The market close date when the bond starts. Can be a string in 'YYYY-MM-DD' format or a datetime object.
    balance : float
        The face amount or principal of the bond.
    coupon : float
        The annual coupon rate in decimal form (e.g., 5% should be input as 0.05).
    maturity_years : float
        The number of years until the bond matures.

    Returns:
    --------
    payment_dates : np.ndarray
        Array of datetime64[D] objects representing the payment dates.
    cash_flows : np.ndarray
        Array of cash flows corresponding to each payment date.

    Raises:
    -------
    ValueError:
        If the market close date is beyond the 28th of the month, to avoid end-of-month issues.

    ValueError:
        If the coupon is greater than 1, then it needs to be input as a decimal.
    """
    
    # Ensure the coupon is in decimal and not a percentage
    if coupon > 1:
        raise ValueError("Coupon should not be greater than 1 as it should be a decimal and not a percentage.")
    
    # Convert market close date to a datetime64[D] object if it's a string
    if isinstance(market_close_date, str):
        market_close_date = np.datetime64(market_close_date, 'D')

    # Convert market close date to a datetime64[D] object if it's a datetime object
    if isinstance(market_close_date, datetime):
        market_close_date = np.datetime64(market_close_date, 'D')

    # Get the day value as a float
    py_date = np.datetime_as_string(market_close_date)  # Convert to a string so the date value can be stripped
    market_close_day = int(py_date[8:10])
    
    # Die if the day of the month is greater than 28
    if market_close_day > 28:
        raise ValueError("Day of the month should not be greater than 28 to avoid end-of-month issues.")

    # Calculate the number of payments (semiannual payments)
    num_payments = int(2*maturity_years)

    # Generate the payment dates by adding multiples of 183 days (approx 6-month periods)
    six_months = np.timedelta64(SPOT_SEMI_BOND_HALF_YEAR, 'D')  # A 1-day offset
    payment_dates = [market_close_date + np.arange(1, num_payments + 1) * six_months]

    # Initialize the cash flows array with values balance * (coupon /2) and add the balancce to the final payment
    cash_flows = np.ones(num_payments) * balance * (coupon / 2)
    cash_flows[-1] += balance
    
    return payment_dates, cash_flows

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