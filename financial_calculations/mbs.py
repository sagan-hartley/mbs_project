import numpy as np
import pandas as pd
from utils import (
    create_regular_dates_grid
)
from .cash_flows import (
    CashFlowData
)

def calculate_monthly_payment(principal, num_months, annual_interest_rate):
    """
    Calculate the monthly payment for a loan.

    Parameters:
    principal (float): Principal amount.
    num_months (int): Number of months.
    annual_interest_rate (float): Annual interest rate (as a decimal).

    Returns:
    float: The monthly payment.

    Raises:
    ValueError: If `num_months` is less than or equal to 0.
    ValueError: If `principal` is less than or equal to 0.

    Notes:
    - If the `annual_interest_rate` is 0, the monthly payment is simply the principal divided by the number of months.
    """
    if num_months <= 0:
        raise ValueError("Number of months must be greater than zero.")
    if principal <= 0:
        raise ValueError("Principal must be greater than zero.")
    
    monthly_interest_rate = annual_interest_rate / 12
    
    # If the interest rate is zero, use simple division to find the monthly payment
    if monthly_interest_rate == 0:
        return principal / num_months

    # Calculate the monthly payment using the formula for an amortizing loan
    payment = (principal * monthly_interest_rate * (1 + monthly_interest_rate) ** num_months) / \
              ((1 + monthly_interest_rate) ** num_months - 1)
    
    return payment

def calculate_scheduled_balances(principal, origination_date, num_months, annual_interest_rate, monthly_payment=None, payment_delay=24):
    """
    Calculate the scheduled balances, principal paydowns, and interest paid for an amortizing loan.
    
    Parameters
    ----------
    principal : float
        The initial principal amount of the loan.
    origination_date : Pandas DateTime
        The date on which the loan originates; should be the first day of a month.
    num_months : int
        The total number of months over which the loan is amortized.
    annual_interest_rate : float
        The annual interest rate as a decimal (e.g., 0.05 for 5%).
    monthly_payment : float, optional
        The fixed monthly payment. If None, the function will calculate it based on the provided terms.
    payment_delay : float, optional
        The payment delay in terms of days. Default is 24 days.

    Returns
    -------
    CashFlowData
        An instance of CashFlowData containing arrays of balances, accrual dates, payment_dates, principal paydowns, and interest paid.

    Raises
    ------
    ValueError
        If the origination_date is not the first of the month.
    """
    
    # Convert origination_date to a pandas Timestamp if it isn't already
    origination_date = pd.to_datetime(origination_date)

    # Ensure origination date is on the first of the month
    if origination_date.day != 1:
        raise ValueError("The origination date must be the first of the month.")

    # Calculate monthly payment if not provided
    if monthly_payment is None:
        monthly_payment = calculate_monthly_payment(principal, num_months, annual_interest_rate)

    # Create an array for each month over the loan term
    months = np.arange(num_months + 1)

    # Generate accrual dates based on the origination date and loan term
    accrual_dates = create_regular_dates_grid(origination_date, origination_date + pd.DateOffset(months=num_months), 'm')

    # Generate payment_dates using the payment delay
    payment_dates = accrual_dates + pd.DateOffset(days=payment_delay)

    # Monthly interest rate
    monthly_rate = annual_interest_rate / 12

    # Calculate remaining balances using amortization formula
    growth_factor = (1 + monthly_rate) ** months
    balances = principal * growth_factor - (monthly_payment / monthly_rate) * (growth_factor - 1)
    balances = np.maximum(balances, 0)  # Prevent negative balances

    # Calculate monthly interest and principal paydowns
    interest_paid = balances[:-1] * monthly_rate
    principal_paydowns = monthly_payment - interest_paid

    # Adjust last payment to pay off any remaining balance
    principal_paydowns[-1] = balances[-2]

    # Include a zero at the start of both interest and principal paydowns arrays
    interest_paid = np.insert(interest_paid, 0, 0)
    principal_paydowns = np.insert(principal_paydowns, 0, 0)

    return CashFlowData(balances, accrual_dates, payment_dates, principal_paydowns, interest_paid)

def calculate_actual_balances(cash_flow_data, smms, net_annual_interest_rate):
    """
    Calculate actual balances with prepayment using `CashFlowData`.

    Parameters
    ----------
    cash_flow_data : CashFlowData
        An instance of `CashFlowData` that includes:
        - scheduled_balances (array-like)
        - accrual_dates (array-like)
        - payment_dates (array-like)
        - scheduled_principal_paydowns (array-like)
        - gross_interest_paid (array-like)
    smms : array-like
        Single Monthly Mortality rates for calculating prepayments.
    net_annual_interest_rate : float
        Net annual interest rate (as a decimal).

    Returns
    -------
    CashFlowData
        A new `CashFlowData` instance containing:
        - actual_balances (array-like): Calculated balances after prepayments.
        - accrual_dates (array-like): Original accrual dates.
        - payment_dates (array-like): Payment dates adjusted by `payment_delay_days`.
        - actual_principal_paydowns (array-like): Principal paydowns based on prepayments.
        - net_interest_paid (array-like): Net interest paid per period.
    """
    # Extract properties from the CashFlowData instance
    scheduled_balances = cash_flow_data.balances
    accrual_dates = cash_flow_data.accrual_dates
    payment_dates = cash_flow_data.payment_dates

    # Derive the number of months from the size of CashFlowData
    num_months = cash_flow_data.get_size()

    # Initialize arrays for actual balances and net interest paid
    actual_balances = np.zeros(num_months)
    net_interest_paid = np.zeros(num_months)

    # Vectorized calculation of pool factors based on SMMS
    pool_factors = np.ones(num_months)
    pool_factors[1:] = np.cumprod(1 - smms)
    pool_factors[-1] = 0  # Ensure the last factor is zero to fully pay down the balance

    # Calculate actual balances, actual principal paydowns, and net interest paid
    actual_balances = scheduled_balances * pool_factors
    actual_principal_paydowns = np.insert(-np.diff(actual_balances), 0, 0)
    net_interest_paid = actual_balances * net_annual_interest_rate / 12
    net_interest_paid = np.concatenate(([net_interest_paid[-1]], net_interest_paid[:-1]))

    # Create and return a new instance of CashFlowData with updated values
    return CashFlowData(
        balances=actual_balances,
        accrual_dates=accrual_dates,
        payment_dates=payment_dates,
        principal_payments=actual_principal_paydowns,
        interest_payments=net_interest_paid
    )
