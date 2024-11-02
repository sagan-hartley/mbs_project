import numpy as np
import pandas as pd

from utils import create_regular_dates_grid
from .cash_flows import CashFlowData

class SemiBondContract:
    def __init__(self, origination_date, term_in_months: int, coupon: float, balance=100.0):
        self.origination_date = pd.to_datetime(origination_date)
        self.term_in_months = term_in_months
        self.coupon = coupon
        self.balance = balance  # Corrected attribute name

def create_semi_bond_cash_flows(semi_bond_contract: SemiBondContract):
    """
    Creates the cash flow schedule for a spot semiannual bond with a bullet repayment structure.

    Parameters:
    -----------
    semi_bond_contract : SemiBondContract
        An instance of SemiBondContract containing origination_date, term_in_months, coupon, and balance.

    Returns:
    --------
    CashFlowData
        An instance of CashFlowData containing balances, accrual dates, payment dates, 
        payments, and interest payments.
    
    Raises:
    -------
    ValueError:
        If the coupon is greater than 1 (should be a decimal) or if the origination date is beyond the 28th of the month.
    """

    # Ensure the coupon is in decimal and not a percentage
    if semi_bond_contract.coupon > 1:
        raise ValueError("Coupon should not be greater than 1 as it should be a decimal and not a percentage.")

    # Generate the payment dates by adding multiples of 6-month periods
    accrual_dates = create_regular_dates_grid(
        semi_bond_contract.origination_date,
        semi_bond_contract.origination_date + pd.DateOffset(months=semi_bond_contract.term_in_months),
        's'
    )

    # Define the payment dates equal to the accrual dates
    payment_dates = accrual_dates

    # Initialize an ndarray of balances to ensure consistent size between inputs for the returned CashFlowData
    balances = semi_bond_contract.balance * np.ones(accrual_dates.size)

    # Calculate the coupon payment for a semi bond
    cpn_payment = semi_bond_contract.balance * semi_bond_contract.coupon / 2.0
    payments = cpn_payment * np.ones(accrual_dates.size)

    # Set first payment to 0 and add final principal repayment to the last payment
    payments[0] = 0.0
    payments[-1] += semi_bond_contract.balance

    # Interest payments (initially zero)
    # We initialize this seemingly useless ndarray so we can return the semi bond as an instance of CashFlowData
    interest_payments = np.zeros(accrual_dates.size)

    return CashFlowData(balances, accrual_dates, payment_dates, payments, interest_payments)
