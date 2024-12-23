import numpy as np
import pandas as pd
from utils import create_regular_dates_grid
from .cash_flows import (
    CashFlowData,
    StepDiscounter
)

class SemiBondContract:
    """
    A class to represent a semiannual bond contract with a bullet repayment structure.

    Attributes
    ----------
    origination_date : pd.Timestamp
        The date on which the bond is issued.
    term_in_months : int
        The bond's term length in months.
    coupon : float
        The annual coupon rate of the bond, in decimal form (e.g., 0.05 for 5%).
    balance : float
        The initial principal balance or face value of the bond.

    Methods
    -------
    None
    """

    def __init__(self, origination_date, term_in_months: int, coupon: float, balance=100.0):
        """
        Initialize a new SemiBondContract instance.

        Parameters
        ----------
        origination_date : str or pd.Timestamp
            The date on which the bond is issued. This can be in 'YYYY-MM-DD' format or as a pd.Timestamp.
        term_in_months : int
            The bond's term length in months.
        coupon : float
            The annual coupon rate of the bond, represented as a decimal (e.g., 0.05 for 5%).
        balance : float, optional
            The initial principal balance or face value of the bond. Default is 100.0.
        """
        
        # Convert the origination date to a pandas Timestamp for consistency
        self.origination_date = pd.to_datetime(origination_date)
        
        # Set the bond term in months
        self.term_in_months = term_in_months
        
        # Set the coupon rate, which represents the annual interest rate
        self.coupon = coupon
        
        # Set the initial principal balance of the bond
        self.balance = balance

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

    # Set the last balance to zero, representing its repayment
    balances[-1] = 0

    # Calculate the coupon payment for a semi bond
    cpn_payment = semi_bond_contract.balance * semi_bond_contract.coupon / 2.0

    # Define the interest and principal payment arrays
    # We will set the interest payments equal to the coupon payment and the principal payments to zero
    interest_payments = cpn_payment * np.ones(accrual_dates.size)
    principal_payments = np.zeros(accrual_dates.size)

    # Set first element in interest_payments to 0 and add the final principal repayment to principal_payments
    interest_payments[0] = 0.0
    principal_payments[-1] += semi_bond_contract.balance

    return CashFlowData(balances, accrual_dates, payment_dates, principal_payments, interest_payments)

def calculate_coupon_rate(start_date, maturity_years, discounter):
    """
    Calculate the coupon rate required to produce a par price for a bond.
    
    Parameters:
    -----------
    start_date : datetime or datetime64[D]
        The start date of the bond.
    maturity_years : int
        The maturity years of the bond.
    discounter : StepDiscounter
        An instance of a discounting class that provides zero-coupon bond discount factors
        for given dates based on its stored rates and dates.

    Returns:
    --------
    float:
        The coupon rate required to produce a par bond.
    
    Raises:
    -------
    ValueError:
        If the start date is before the market close date.
    """
    # Access the market close date from the discounter
    market_close_date = discounter.market_close_date

    # Validate start date
    if start_date < market_close_date:
        raise ValueError("Start date must be on or after the market close date.")

    # Generate bond payment dates from start date until maturity with semiannual intervals
    end_date = start_date + pd.DateOffset(years=maturity_years)
    payment_dates = create_regular_dates_grid(start_date, end_date, 's')

    # Calculate discount factors for the bond payment dates
    discount_factors = discounter.zcbs_from_dates(payment_dates)

    # Calculate annuity (present value of coupon payments)
    annuity = 0.5 * np.sum(discount_factors[1:])  # Exclude the first discount factor

    # Initial and final discount factors
    initial_discount = discount_factors[0]
    final_discount = discount_factors[-1]

    # Calculate the coupon rate to produce a par bond
    coupon_rate = (initial_discount - final_discount) / annuity

    return coupon_rate

def pathwise_zcb_eval(maturity_dates, short_rate_paths, short_rate_dates):
    """
    Price a Zero-Coupon Bond (ZCB) using short rate paths by computing the discount factors
    and averaging across paths.

    Parameters:
    -----------
    maturity_dates : array-like
        Array of maturity dates for the ZCBs.
    short_rate_paths : array-like
        Array of short rate paths with shape (num_paths, num_steps).
    short_rate_dates : array-like
       Array of dates corresponding to each short rate path.

    Returns:
    --------
    zcb_vals : np.ndarray
        Array of zcb values based on the maturity dates for each short rate path.
        Each row corresponds to a different short rate path and each column
        represents the different zcb values at that maturity rate.
    """
    # Ensure short_rates is a numpy array and handle 1D case by converting to 2D
    # The one_d_array_bool will be used later to determine return type
    short_rate_paths = np.asarray(short_rate_paths)
    if short_rate_paths.ndim == 1:
        one_d_array_bool = True
        short_rate_paths = short_rate_paths[np.newaxis, :]
    else:
        one_d_array_bool = False
    
    # Get the number of short rate paths and initialize a list to store ZCB present values
    num_paths = short_rate_paths.shape[0]
    present_values = []

    # Initialize a StepDiscounter instance with a placeholder rate, to be updated within each path iteration
    discounter = StepDiscounter(short_rate_dates, short_rate_paths[0, :])

    # Loop over each path to calculate present value using the discount_cash_flows function
    for i in range(num_paths):
        # Update the discounter rates to the current short rate path
        discounter.set_rates(short_rate_paths[i, :])
        
        # Discount the cash flows for this path
        vals = discounter.zcbs_from_dates(maturity_dates)

        # Append the current values to the present_values list
        present_values.append(vals)

    # If only one path, return the values directly as a 1D array
    if one_d_array_bool:
        return present_values[0]
    
    return np.array(present_values)
