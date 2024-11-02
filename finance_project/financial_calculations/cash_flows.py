import pandas as pd
import numpy as np
from utils import (
    years_from_reference,
    integral_knots,
    zcbs_from_deltas
)

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
        self.balances = balances
        self.accrual_dates = accrual_dates
        self.payment_dates = payment_dates
        self.principal_payments = principal_payments
        self.interest_payments = interest_payments

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
        self.market_close = pd.to_datetime(dates[0])
        
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
        zcb_deltas = years_from_reference(self.market_close, zcb_dates)
        
        # Return the ZCB discount factors by applying the discounting function.
        return zcbs_from_deltas(zcb_deltas, self.integral_vals, self.integral_time_deltas)

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
    """
    # Filter cash flows that occur after the settlement date
    filtered_cfs = filter_cash_flows(cash_flows, settle_date)

    # Calculate the number of years between each payment date and the settle date
    filtered_years = years_from_reference(settle_date, filtered_cfs.payment_dates)

    # Calculate the total paydown for each period
    filtered_paydowns = filtered_cfs.principal_payments

    # Compute the numerator and denominator for WAL calculation
    wal_numerator = np.sum(filtered_years * filtered_paydowns)
    wal_denominator = np.sum(filtered_paydowns)
    print(wal_numerator, wal_denominator)

    # Calculate WAL
    wal = wal_numerator / wal_denominator if wal_denominator != 0 else 0

    return wal
