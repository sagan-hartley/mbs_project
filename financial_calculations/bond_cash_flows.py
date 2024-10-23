import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from utils import convert_to_datetime

PMTS_PER_YEAR = 2

def create_semi_bond_cash_flows(effective_date, balance, coupon, maturity_years):
    """
    Creates the cash flow schedule for a spot semiannual bond with a bullet repayment structure.

    Parameters:
    -----------
    effective_date : str or datetime or datetime64[D]
        The effective date when the bond starts. Can be a string in 'YYYY-MM-DD' format or a datetime object.
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
        If the effective date is beyond the 28th of the month, to avoid end-of-month issues.

    ValueError:
        If the coupon is greater than 1, then it needs to be input as a decimal.
    """
    
    # Ensure the coupon is in decimal and not a percentage
    if coupon > 1:
        raise ValueError("Coupon should not be greater than 1 as it should be a decimal and not a percentage.")
    
    # Convert effective date to a datetime object if it's a string
    if isinstance(effective_date, str):
        effective_date = datetime.strptime(effective_date, "%Y-%m-%d")

    # Convert effective date to a datetime object if it's a datetime64[D] object
    effective_date = convert_to_datetime(effective_date)
    
    # Die if the day of the month is greater than 28
    if effective_date.day > 28:
        raise ValueError("Day of the month should not be greater than 28 to avoid end-of-month issues.")

    # Calculate the number of payments (semiannual payments)
    num_payments = int(PMTS_PER_YEAR * maturity_years)

    # Generate the payment dates by adding multiples of 6-month periods and change type to datetime64[D] object
    payment_dates = [np.datetime64(effective_date + relativedelta(months=6 * i), 'D') for i in range(1, num_payments + 1)]

    # Initialize the cash flows array with values balance * (coupon /2) and add the balance to the final payment
    cash_flows = np.ones(num_payments) * balance * (coupon / PMTS_PER_YEAR)
    cash_flows[-1] += balance
    
    return payment_dates, cash_flows
