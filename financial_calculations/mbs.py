from dataclasses import dataclass
import numpy as np
import pandas as pd
from utils import (
    create_regular_dates_grid,
    calculate_antithetic_variance
)
from .cash_flows import (
    CashFlowData,
    StepDiscounter,
    evaluate_cash_flows
)
from financial_models.prepayment import (
    calculate_pccs,
    calculate_smms
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

@dataclass
class MbsContract:
    mbs_id: str
    balance: float
    origination_date: pd.Timestamp
    num_months: int
    gross_annual_coupon: float
    net_annual_coupon: float
    payment_delay: int
    settle_date: pd.Timestamp

    def __post_init__(self):
        # Validate origination_date and settle_date to be pandas Timestamps
        if not isinstance(self.origination_date, pd.Timestamp):
            raise ValueError(f"Invalid origination_date: {self.origination_date}. Must be a pandas Timestamp.")
        
        if not isinstance(self.settle_date, pd.Timestamp):
            raise ValueError(f"Invalid settle_date: {self.settle_date}. Must be a pandas Timestamp.")

import numpy as np

def pathwise_evaluate_mbs(mbs_list, short_rates, short_rate_dates, store_vals=True, store_expecteds=True, store_stdevs=True, antithetic=False):
    """
    Calculate the expected weighted average life (WAL), value, price, as well as path standard deviations 
    for the Mortgage-Backed Securities (MBS) based on simulated short-rate paths.

    Parameters:
    - mbs_list (list): A list of MBS objects. Each object should contain the attributes used in the function logic, 
      including `mbs_id`, `balance`, `origination_date`, `num_months`, `gross_annual_coupon`, `net_annual_coupon`, 
      `payment_delay`, and `settle_date`.
    - short_rates (list or ndarray): A 2D array (or list) representing the simulated short rates for each path. If a 
      1D array or list is provided, it will be converted to a 2D array with a single row.
    - short_rate_dates (list or ndarray): An array of dates corresponding to each short rate path.
    - store_vals (bool): If True, store individual path values in the results. Default is True.
    - store_expecteds (bool): If True, store expected (mean) values for WAL, value, and price. Default is True.
    - store_stdevs (bool): If True, store standard deviations for WAL, value, and price. Default is True.
    - antithetic (bool): If True, calculate the antithetic variance. If False, calculate the simple variance. Default is False.

    Returns:
    - results (list): A list of dictionaries, each containing the results for one MBS.
    """
    # Ensure short_rates is a numpy array and handle 1D case by converting to 2D
    short_rates = np.asarray(short_rates)
    if short_rates.ndim == 1:
        short_rates = short_rates[np.newaxis, :]
    
    results = []  # To store results for each MBS
    market_close_date = short_rate_dates[0]  # Extract the market close date from the short rate dates

    # Loop through each MBS in the provided list
    for mbs in mbs_list:
        # Initialize lists to hold pathwise results for the current MBS
        wals, vals, prices = [], [], []  

        # Calculate the necessary scheduled balance data to value and price the cash flows
        scheduled_balances = calculate_scheduled_balances(
            mbs.balance, mbs.origination_date, mbs.num_months, mbs.gross_annual_coupon, 
            payment_delay=mbs.payment_delay
        )

        # Calculate the Primary Current Coupons (PCCs) and SMMs based on the original short rates
        pccs = calculate_pccs(short_rates, short_rate_dates, scheduled_balances.accrual_dates[:-1])
        smms = calculate_smms(pccs, mbs.gross_annual_coupon, scheduled_balances.accrual_dates[:-1])

        # Initialize a StepDiscounter instance with a placeholder rate, to be updated within each path iteration
        discounter = StepDiscounter(short_rate_dates, short_rates[0, :])

        # Loop through each short rate path and corresponding SMM path
        for index, smm_path in enumerate(smms):
            # Calculate the actual scheduled balances based on the current SMM path
            actual_balances = calculate_actual_balances(scheduled_balances, smm_path, mbs.net_annual_coupon)

            # Update the discounter rates to the current short rate path
            discounter.set_rates(short_rates[index, :])

            # Evaluate cash flows for the current path
            wal, val, price = evaluate_cash_flows(
                actual_balances, discounter, mbs.settle_date, mbs.net_annual_coupon
            )
            
            # Append results to respective lists
            wals.append(wal)
            vals.append(val)
            prices.append(price)

        # Prepare the dictionary for storing results
        mbs_result = {'mbs_id': mbs.mbs_id}

        # Store pathwise results based on the store_vals flag
        if store_vals:
            mbs_result.update({
                'wals': wals,
                'vals': vals,
                'prices': prices
            })

        # Store expected values if store_expecteds is True
        if store_expecteds:
            mbs_result.update({
                'expected_wal': np.mean(wals),
                'expected_value': np.mean(vals),
                'expected_price': np.mean(prices)
            })

        # Store standard deviations if store_stdevs is True
        if store_stdevs:

            # If antithetic is true calculate the standard deviation 
            # as the square root of the antithetic variance
            if antithetic:
                wal_stdev = np.sqrt(calculate_antithetic_variance(wals))
                value_stdev = np.sqrt(calculate_antithetic_variance(vals))
                price_stdev = np.sqrt(calculate_antithetic_variance(prices))

            # If not just calculate the standard deviation using numpy.std()
            else:
                wal_stdev = np.std(wals),
                value_stdev = np.std(vals),
                price_stdev = np.std(prices)

            mbs_result.update({
                'wal_stdev': wal_stdev,
                'value_stdev': value_stdev,
                'price_stdev': price_stdev
            })

        # Append the result for this MBS to the results list
        results.append(mbs_result)

    return results  # Return the list of results for all MBS
