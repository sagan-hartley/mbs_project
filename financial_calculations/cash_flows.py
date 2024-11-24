import pandas as pd
import numpy as np
from scipy.optimize import minimize
from utils import (
    days360,
    years_from_reference,
    integral_knots,
    zcbs_from_deltas
)

CASH_DAYS_IN_YEAR = 360

class CashFlowData:
    """
    A class to represent cash flow data associated with a financial product,
    including balances, dates for accrual and payment, and payment amounts.

    Attributes:
    -----------
    balances : np.ndarray
        Array of balances at each accrual or payment date.
    accrual_dates : np.ndarray or pd.DatetimeIndex
        Array of accrual dates for the cash flows.
    payment_dates : np.ndarray or pd.DatetimeIndex
        Array of payment dates for the cash flows.
    principal_payments : np.ndarray
        Array of principal payment amounts.
    interest_payments : np.ndarray
        Array of interest payment amounts.

    Methods:
    --------
    get_size():
        Returns the number of elements in each input array, assuming all inputs are the same length.
    get_total_payments():
        Returns the sum of principal and interest payments.
    """

    def __init__(self, balances, accrual_dates, payment_dates, principal_payments, interest_payments):
        """
        Initializes the CashFlowData class with balances, accrual dates, payment dates, 
        principal payments, and interest payments. Validates that all inputs have the same length.

        Parameters:
        -----------
        balances : np.ndarray
            Array of balances at each accrual or payment date.
        accrual_dates : np.ndarray or pd.DatetimeIndex
            Array of accrual dates for the cash flows.
        payment_dates : np.ndarray or pd.DatetimeIndex
            Array of payment dates for the cash flows.
        principal_payments : np.ndarray
            Array of principal payment amounts.
        interest_payments : np.ndarray
            Array of interest payment amounts.

        Raises:
        -------
        ValueError:
            If any input array has a different length from the others.
        """
        # Check that all inputs have the same length
        input_lengths = {len(balances), len(accrual_dates), len(payment_dates), 
                         len(principal_payments), len(interest_payments)}
        if len(input_lengths) != 1:
            raise ValueError("All input arrays must have the same length.")
        
        # Initialize attributes if all inputs are of the same length
        # If inputs are of the wrong type, convert them if possible
        self.balances = np.array(balances)
        self.accrual_dates = pd.to_datetime(accrual_dates)
        self.payment_dates = pd.to_datetime(payment_dates)
        self.principal_payments = np.array(principal_payments)
        self.interest_payments = np.array(interest_payments)

    def get_size(self):
        """
        Returns the number of elements in the cash flow arrays.

        Returns:
        --------
        int
            Number of elements in the input arrays (assuming all are the same length).
        """
        return len(self.balances)

    def get_total_payments(self):
        """
        Computes the total payments by summing principal and interest payments.

        Returns:
        --------
        np.ndarray
            Array of total payments, calculated as the sum of principal and interest payments.
        """
        return self.principal_payments + self.interest_payments

class StepDiscounter:
    """
    StepDiscounter calculates zero-coupon bond (ZCB) discount factors from given dates and rates.

    Attributes:
    -----------
    market_close : pd.Timestamp
        The initial reference date (first date in `dates`).
    integral_time_deltas : np.ndarray
        Array of year deltas calculated from `dates`.
    integral_vals : np.ndarray
        Cumulative integral values calculated from `rates`.
    dates : pd.DatetimeIndex
        Original array of dates representing the time periods for discounting.
    rates : np.ndarray
        Array of rates associated with each period in `dates`.

    Methods:
    --------
    zcbs_from_dates(zcb_dates):
        Calculates ZCB discount factors for the specified `zcb_dates` based on initial rates and dates.
    """

    def __init__(self, dates, rates):
        """
        Initialize the StepDiscounter with a reference date and rates, computing integral values
        to later use for ZCB discount factors.

        Parameters:
        -----------
        dates : pd.DatetimeIndex
            An array of dates representing the time periods for discounting.
        rates : np.ndarray
            An array of rates corresponding to each date in `dates`.

        Raise:
        ------
        ValueError
            If the legnth of dates and rates are not equal
        """
        # Check that dates and rates have the same length: raise an error if not
        if len(dates) != len(rates):
            raise ValueError("dates and rates must be of the same length.")
        
        # Calculate year deltas from the reference date and their cumulative integral values.
        yrs_from_reference, integral_vals = integral_knots(dates, rates)
        
        # Set the market close as the reference date for year calculations.
        self.market_close_date = pd.to_datetime(dates[0])
        
        # Store the calculated year deltas and integral values for later use.
        self.integral_time_deltas = yrs_from_reference
        self.integral_vals = integral_vals

        # Store the input dates and rates for potential future calculations
        self.dates = dates
        self.rates = rates

    def zcbs_from_dates(self, zcb_dates):
        """
        Calculate zero-coupon bond (ZCB) discount factors for specified dates.

        Parameters:
        -----------
        zcb_dates : pd.DatetimeIndex
            Array of dates for which ZCB discount factors are to be calculated.

        Returns:
        --------
        np.ndarray
            An array of ZCB discount factors corresponding to `zcb_dates`.
        """
        # Calculate time deltas in years from the reference date to each ZCB date.
        zcb_deltas = years_from_reference(self.market_close_date, zcb_dates)
        
        # Return the ZCB discount factors by applying the discounting function.
        return zcbs_from_deltas(zcb_deltas, self.integral_vals, self.integral_time_deltas)
    
    def set_rates(self, new_rates):
        """
        Update the rates and recompute the integral values based on the new rates.

        Parameters:
        -----------
        new_rates : np.ndarray
            An updated array of rates with the same length as the original `rates`.

        Raises:
        -------
        ValueError
            If `new_rates` has a different length than the original `rates`.
        """
        # Ensure the new rates array has the same length as the original rates array
        if len(new_rates) != len(self.rates):
            raise ValueError("The length of `new_rates` must match the length of `rates`.")

        # Update the rates and recompute the integral values with the new rates
        self.rates = new_rates
        _, self.integral_vals = integral_knots(self.dates, self.rates)

def filter_cash_flows(cash_flows, settle_date):
    """
    Filters cash flows occurring after a specified settlement date.

    Parameters:
    -----------
    cash_flows : CashFlowData
        An instance of CashFlowData containing balances, accrual dates, payment dates, 
        and payments.
    settle_date : str or pd.Timestamp
        The settlement date after which cash flows are to be retained. 
        This is converted to a pd.Timestamp if provided as a string.

    Returns:
    --------
    CashFlowData
        A new CashFlowData instance containing only the cash flows that occur after 
        the settlement date.
    """
    # Convert settle_date to a Timestamp if it's not already
    settle_date = pd.to_datetime(settle_date)

    # Get indices of cash flows that occur after the settlement date
    post_settle_indices = np.where(cash_flows.accrual_dates > settle_date)[0]

    # Return filtered cash flows as a new instance of CashFlowData
    return CashFlowData(
        balances=cash_flows.balances[post_settle_indices],
        accrual_dates=cash_flows.accrual_dates[post_settle_indices],
        payment_dates=cash_flows.payment_dates[post_settle_indices],
        principal_payments=cash_flows.principal_payments[post_settle_indices],
        interest_payments=cash_flows.interest_payments[post_settle_indices]
    )

def value_cash_flows(discounter, cash_flows, settle_date):
    """
    Calculate the present value of cash flows using a given discounter.

    Parameters:
    -----------
    discounter : StepDiscounter
        An instance of StepDiscounter used to calculate zero-coupon bond discount factors.
    cash_flows : CashFlowData
        An instance of CashFlowData containing balances, accrual dates, payment dates, 
        and payments.
    settle_date : str or pd.Timestamp
        The settlement date from which cash flows are to be valued. 
        This is converted to a pd.Timestamp if provided as a string.

    Returns:
    --------
    float
        The present value of the cash flows discounted to the settlement date.
    """
    # Filter cash flows that occur after the settlement date
    filtered_cfs = filter_cash_flows(cash_flows, settle_date)

    # Get the ZCB discount factors for the filtered payment dates
    filtered_zcbs = discounter.zcbs_from_dates(filtered_cfs.payment_dates)

    # Get the initial ZCB discount factor for the settlement date
    initial_zcb = discounter.zcbs_from_dates(settle_date)

    # Calculate the present value of the cash flows
    value = np.dot(filtered_cfs.get_total_payments(), filtered_zcbs) / initial_zcb

    return value

def price_cash_flows(present_value, balance_at_settle, settle_date, last_coupon_date, annual_interest_rate, par_balance=100):
    """
    Calculate the clean price of a bond from its present value, accrued interest, and settlement details.

    Parameters
    ----------
    present_value : float
        The present value of the bond's cash flows.
    balance_at_settle : float
        The bond's outstanding balance at the settlement date.
    settle_date : datetime
        The date on which the bond is settled.
    last_coupon_date : datetime
        The date of the last coupon payment.
    annual_interest_rate : float
        The annual interest rate as a decimal (e.g., 0.05 for 5%).
    par_balance : float, optional
        The par balance for the bond, by default 100.

    Returns
    -------
    float
        The calculated clean price of the bond.

    Notes
    -----
    - The clean price is derived by subtracting accrued interest from the dirty price.
    - If `balance_at_settle` is zero, the dirty price is set to zero.
    """  
    # Calculate the dirty price
    # if balance at settle is zero, go ahead and return 0 as it is the dirty an clean price
    if balance_at_settle == 0:
        return 0
    else:
        # Normalize the present value by the balance at settlement
        dirty_price = present_value * par_balance / balance_at_settle

    # Calculate the days between the last coupon and settlement dates
    days_between = days360(pd.to_datetime(last_coupon_date), pd.to_datetime(settle_date))
    
    # Compute accrued interest
    accrued_interest = (annual_interest_rate / CASH_DAYS_IN_YEAR) * days_between * par_balance
    
    # Derive clean price by subtracting accrued interest from the dirty price
    clean_price = dirty_price - accrued_interest

    return clean_price

def get_balance_at_settle(cash_flows, filtered_cfs):
    """
    Calculate the balance at settlement based on cash flows and filtered cash flows.

    The balance at settlement is determined dynamically by the difference 
    between the first value in filtered cash flows and the last element that 
    was filtered out from the original cash flows.

    If the cash flows have not been filtered, the balance at settlement is 
    the first element in cash_flows.balances.

    Parameters:
    ----------
    cash_flows : object
        An object that contains `balances` (array-like) and `payment_dates` (array-like).
    
    filtered_cfs : object
        An object that contains `balances` (array-like) and `payment_dates` (array-like).
    
    Returns:
    -------
    float
        The balance at settlement.

    Raises:
    ------
    ValueError
        If the first payment date in filtered cash flows is not found in cash flows.
    """
    # Check if the first balance in filtered cash flows is the same as the first in cash flows
    if filtered_cfs.balances[0] == cash_flows.balances[0]:
        balance_at_settle = cash_flows.balances[0]  # Same start point
    else:
        # Find the index of the first payment date in filtered_cfs
        index = np.where(cash_flows.payment_dates == filtered_cfs.payment_dates[0])[0]
        if index.size == 0:
            raise ValueError("The first payment date in filtered cash flows is not found in cash flows.")
        
        # Use the index to get the balance just before this payment date
        balance_at_settle = cash_flows.balances[index[0] - 1]

    return balance_at_settle

def calculate_weighted_average_life(cash_flows, settle_date):
    """
    Calculate the Weighted Average Life (WAL) of a CashFlowData instance.

    Parameters:
    -----------
    cash_flows : CashFlowData
        An instance of CashFlowData containing payment_dates and payments.
    settle_date : str or pd.Timestamp
        The date from which to calculate the time until cash flows are received.
        This is converted to a pd.Timestamp if provided as a string.

    Returns:
    --------
    float
        The Weighted Average Life of the cash flows.

    Notes:
    ------
    The WAL is calculated by the difference in balances rather than the principal paydowns
    as this attribute is much easier to work with in cases where prepayment affects the
    cash flows.
    """
    # Filter cash flows that occur after the settlement date
    filtered_cfs = filter_cash_flows(cash_flows, settle_date)

    # Check if there are no payment dates after the settle date
    if len(filtered_cfs.payment_dates) == 0:
        return 0  # No payments after the settle date

    # Calculate the number of years between each payment date and the settle date
    filtered_years = years_from_reference(settle_date, filtered_cfs.payment_dates)

    # Get the filtered balances
    filtered_balances = filtered_cfs.balances
    
    # Determine the initial paydown by the difference between the balance at settle and the first value in filtered_cfs
    initial_paydown = get_balance_at_settle(cash_flows, filtered_cfs) - filtered_cfs.balances[0]
    
    # Create paydown array by appending the initial paydown and the negative differences in balances
    filtered_paydowns = np.append([initial_paydown], -np.diff(filtered_balances))

    # Compute the numerator and denominator for WAL calculation
    wal_numerator = np.dot(filtered_years, filtered_paydowns)
    wal_denominator = np.sum(filtered_paydowns)

    # Calculate WAL
    wal = wal_numerator / wal_denominator if wal_denominator != 0 else 0

    return wal

def get_last_accrual_date(cash_flows, settle_date):
    """
    Get the last accrual date from the cash flows based on the settle date.
    If no valid accrual dates are found, return the settle date.
    
    Parameters
    ----------
    cash_flows : np.ndarray
        An array of all cash flow payment dates, sorted in ascending order.

    settle_date : datetime
        The date to return if no valid coupon dates are found after filtering.

    Returns
    -------
    datetime
        The last accrual date, which is the accrual date from cash_flows right before the settle_date,
        or settle_date if no valid accrual dates are found.
    """
    # Find the index of the accrual date right before the settle date
    last_accrual_index = np.searchsorted(cash_flows.accrual_dates, settle_date)

    # If the index is 0, it means settle_date is before the first date
    if last_accrual_index == 0:
        return settle_date  # Return settle_date if no valid accrual dates found

    # Return the last valid accrual date before the settle date
    last_accrual_date = cash_flows.accrual_dates[last_accrual_index - 1]

    return last_accrual_date

def evaluate_cash_flows(cash_flows, discounter, settle_date, net_annual_interest_rate):
    """
    Evaluate the key metrics of a set of cash flows, including the weighted average life (WAL), 
    present value, and price of the cash flows.

    This function calculates the following metrics:
    - Weighted Average Life (WAL): The average time it takes for a principal amount to be repaid,
      weighted by the amounts of the cash flows.
    - Present Value: The present value of the cash flows as discounted using the provided discounter.
    - Price: The price of the cash flows at the given settlement date, considering the balance at settle
      and applying the net annual interest rate.

    Parameters
    ----------
    cash_flows : object
        An object containing cash flow information, including payment dates, balances, and amounts.
    discounter : function
        A function to discount the cash flows to their present value.
    settle_date : datetime-like
        The settlement date for the evaluation.
    net_annual_interest_rate : float
        The net annual interest rate to be used in pricing the cash flows.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        - WAL (float): Weighted Average Life of the cash flows.
        - value (float): The present value of the cash flows.
        - price (float): The price of the cash flows based on the balance at settle and net interest rate.
    """
    # Calculate the present value of the cash flows by using the provided discounter function
    value = value_cash_flows(discounter, cash_flows, settle_date)

    # Filter the cash flows based on the settle date
    filtered_cfs = filter_cash_flows(cash_flows, settle_date)
    
    # Determine the balance at settle based on the filtered cash flows and calculate the price
    balance_at_settle = get_balance_at_settle(cash_flows, filtered_cfs)
    
    # Determine the last accrual date based on the cash flows and settle date
    last_accrual_date = get_last_accrual_date(cash_flows, settle_date)
    
    # Calculate the price of the cash flows, considering the net annual interest rate and other parameters
    price = price_cash_flows(value, balance_at_settle, settle_date, last_accrual_date, net_annual_interest_rate)
    
    # Calculate the weighted average life (WAL) of the cash flows using the settle date
    wal = calculate_weighted_average_life(cash_flows, settle_date)

    # Return the WAL, present value, and price of the cash flows as a tuple
    return wal, value, price

def oas_search(cash_flows, discounter, settle_date, target=100, initial_guess=0.03, tolerance=1e-4):
    """
    Perform an Option-Adjusted Spread (OAS) search to align the computed bond value with the target price.

    Parameters:
    - cash_flows (CashFlowData): An instance of CashFlowData.
    - discounter (StepDiscounter): An instance of StepDiscounter representing a forward curve 
    - settle_date (Timestamp): The settlement date for the cash flows.
    - target (float, optional): The target price of the bond/security. Default is 100 (par value).
    - initial_guess (float, optional): Initial guess for the OAS (in decimal form, e.g., 0.03 for 3%). 
                                       Default is 0.03.
    - tolerance (float, optional): How far off the squared difference between the value using 
            discount rates plus the oas and the target can be from zero.

    Returns:
    - oas (float): The computed Option-Adjusted Spread (in decimal form).

    Raises:
    - ValueError: If the minimization process fails to converge.
    """
    # Extract the starting rates to be used for the objective function
    start_rates = discounter.rates

    def objective(oas):
        """
        Objective function to minimize. Adjusts the discount rates using the OAS and computes the squared error
        between the present value of cash flows and the target price.

        Parameters:
        - oas (float): The Option-Adjusted Spread to test (in decimal form).

        Returns:
        - float: The squared difference between the computed price and the target.
        """
        # Adjust discount rates by adding the current OAS value to the starting rates
        discounter.set_rates(start_rates + oas)
        
        # Compute the present value of cash flows using the updated discounter
        value = value_cash_flows(discounter, cash_flows, settle_date)
        
        # Return the squared difference from the target price
        return (value - target) ** 2

    # Perform the optimization using the L-BFGS-B method
    result = minimize(
        objective, 
        x0=initial_guess, 
        method='L-BFGS-B', 
        bounds=[(-1, 1)],
        options={'ftol': target * 1e-7}  # Tolerance proportional to the target
    )

    # Reset the discounter back to its original rates
    discounter.set_rates(start_rates)

    # Check for convergence
    if not result.success:
        raise ValueError("Minimization did not converge. Try adjusting the initial guess or checking input data.")
    
    # Check that result satisfies the input error tolerance
    if result.fun > tolerance:
        raise ValueError(f"Optimization result is not within tolerance: {result.fun:.6f}")
    
    # Return the computed OAS
    return result.x[0]

def calculate_dv01(up_val, down_val, bump_amount):
    """
    Calculate the dollar value of one basis point (DV01), 
    which represents the price change for a 1 basis point shift in yield.

    Parameters:
    -----------
    up_val : float
        The value of a security when rates are bumped up by a specified amount.
    down_val : float
        The value of a security when rates are bumped down by a specified amount.
    bump_amount : float
        The amount by which rates were bumped to obtain `bumped_vals`.

    Returns:
    --------
    float
        The DV01.

    Raises:
    -------
    ZeroDivisionError
        If 'bump_amount' is 0.
    """ 
    # Check bump_amount is nonzero
    if bump_amount == 0:
        raise ZeroDivisionError("bump_amount cannot be zero.")

    # Calculate the DV01   
    dv01 = (np.mean(up_val) - np.mean(down_val)) / (2*bump_amount)

    return dv01

def calculate_convexity(val, bumped_up_val, bumped_down_val, bump_amount):
    """
    Calculate the convexity of a security based on the the original,
    bumped-up, and bumped-down values.

    Convexity is calculated as:
        (bumped_up_val - 2 * val + bumped_down_vals) / (bump_amount ** 2)

    Parameters
    ----------
    val : float
        Original value of the security
    bumped_up_val : float
        The value of a security when rates are bumped up by a specified amount.
    bumped_down_val : float
        The value of a security when rates are bumped down by a specified amount.
    bump_amount : float
        The amount of the bump applied to generate `bumped_up_val` and `bumped_down_val`.

    Returns
    -------
    float
        The convexity measure.

    Raises
    ------
    ZeroDivisionError
        If `bump_amount` is zero.
    """
    # Raise ZeroDivisionError for zero bump amount
    if bump_amount == 0:
        raise ZeroDivisionError("Bump amount must be non-zero.")

    # Calculate convexity
    convexity = (bumped_up_val - 2 * val + bumped_down_val) / (bump_amount ** 2)

    return convexity
