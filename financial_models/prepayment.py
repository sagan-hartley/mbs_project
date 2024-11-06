import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from utils import integer_months_from_reference

SEASONAL_FACTORS_ARRAY = np.array([
        0.75,  # January (month 1)
        0.84,  # February (month 2)
        0.93,  # March (month 3)
        1.02,  # April (month 4)
        1.11,  # May (month 5)
        1.20,  # June (month 6)
        1.20,  # July (month 7)
        1.11,  # August (month 8)
        1.02,  # September (month 9)
        0.93,  # October (month 10)
        0.84,  # November (month 11)
        0.75   # December (month 12)
    ])

def calculate_pccs(short_rates, spread=0.04):
    """
    Calculate the Primary Cuurent Coupons (PCCs) by adding a fixed spread to the short rates.
    
    Parameters:
    - short_rates (ndarray): Array of short rates from the Hull-White model simulation.
    - spread (float): Fixed spread to add to short rates. Default is 0.04 (4%).
    
    Returns:
    - ndarray: Array of PCC values.
    """
    return short_rates + spread

def refi_strength(spreads):
    """
    Calculate the refinancing strength based on an array of spread values.

    The function applies a piecewise linear function to each spread value:
    - Returns 0 if the spread is less than or equal to 0.
    - Increases linearly from 0 to 0.0425 for spreads between 0 and 0.015.
    - Returns a constant value of 0.0425 for spreads greater than or equal to 0.015.

    Parameters
    ----------
    spreads : array-like
        Array of spread values between the gross coupon rate and the primary
        current coupon rate (PCC). Each value in this array represents
        the incentive for a borrower to refinance.

    Returns
    -------
    np.ndarray
        An array of refinancing strength values corresponding to the input spreads,
        with values bounded between 0 and 0.0425.

    Examples
    --------
    >>> spreads = np.array([-0.01, 0, 0.0075, 0.015, 0.02])
    >>> refi_strength(spreads)
    array([0.    , 0.    , 0.02125, 0.0425 , 0.0425 ])
    """
    return np.clip(0.0425 / 0.015 * spreads, 0.0, 0.0425)

def demo(origination_date, num_months, base_smm = 0.005):
    """
    Calculate the demographic factors for each month of a loan's term.

    Demographic factors are calculated as:
        Demo(age) = seasoning(age) * seasonal(month)

    where:
    - seasoning(age) = max(1, age / 18) * base_smm
    - seasonal(month_of_year) represents a seasonal adjustment based on the month of the year.

    Parameters
    ----------
    origination_date : datetime
        The date when the loan was originated.
    num_months : int
        The number of months for the loan term.
    base_smm : float
        The base demographic single month mortality rate

    Returns
    -------
    np.ndarray
        An array of demographic factors for each month in the loan term.
    """
    # Generate ages in months up to num_months
    ages = np.arange(num_months)

    # Seasoning factors based on age
    seasoning_factors = np.maximum(1, ages / 18) * base_smm

    # Calculate month of year for each month in the loan term
    start_month = origination_date.month
    months_of_year = (start_month + ages) % 12

    # Use months_of_year to index directly into the precomputed seasonal factors array
    # Subtract 1 from months_of_year because array indices are 0-based
    seasonal_factors = SEASONAL_FACTORS_ARRAY[months_of_year - 1]

    # Calculate demographic factors
    demo_factors = seasoning_factors * seasonal_factors

    return demo_factors

def calculate_smms(pccs, coupon, market_close_date, origination_date, num_months):
    """
    Calculate the Single Monthly Mortality (SMM) rates based on the PCC and the MBS coupon.
    
    SMM is influenced by refinancing incentives and demographic factors over time.

    Parameters
    ----------
    pccs : ndarray
        A 2D array of Primary Current Coupon (PCC) values, where rows represent scenarios and columns
        represent months.
    coupon : float
        The coupon rate of the MBS.
    market_close_date : datetime or datetime64
        The market close date for the MBS, indicating the start of the evaluation.
    origination_date : datetime or datetime64
        The origination date for the MBS.
    num_months : int
        The length of the MBS in months.
    
    Returns
    -------
    ndarray
        A 2D array of SMM values for each scenario and month.
    
    Raises
    ------
    ValueError
        If the length of `pccs` is less than `num_months` or if the PCC array has unexpected dimensions.
    """
    if pccs.shape[1] < num_months:
        raise ValueError("Length of PCCs is less than the specified number of months.")

    # Ensure dates are in datetime format
    market_close_date = pd.to_datetime(market_close_date)
    origination_date = pd.to_datetime(origination_date)

    # Calculate integer months difference between origination and market close
    total_months_diff = integer_months_from_reference(market_close_date, origination_date)

    # Adjust PCCs for the relevant term window
    pccs = pccs[:, total_months_diff:total_months_diff + num_months]

    # Calculate the refinancing incentive and demographic factors
    spread = coupon - pccs
    refi_factors = refi_strength(spread)
    demo_factors = demo(origination_date, num_months)

    # SMM calculation as the sum of refinancing and demographic components
    smms = refi_factors + demo_factors

    return smms
