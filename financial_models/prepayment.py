import numpy as np
import pandas as pd
from utils import (
    create_regular_dates_grid,
    lag_2darray
)

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
    Calculate the Primary Current Coupons (PCCs) by adding a fixed spread to the short rates.

    Parameters:
    - short_rates (array-like): 1D or 2D array of short rates from the Hull-White model simulation.
    - spread (float): Fixed spread to add to short rates. Default is 0.04 (4%).

    Returns:
    - ndarray: PCC values.
    """
    # Convert short_rates to a numpy array and broadcast the spread addition to get the PCCs
    pccs = np.asarray(short_rates) + spread

    return pccs

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
    return (0.0425 / 0.015) * np.clip(spreads, 0.0, 0.015)

def demo(smm_dates, base_smm=0.005):
    """
    Calculate the demographic factors for each date in the loan's term using a DatetimeIndex.

    Demographic factors are calculated as:
        Demo(age) = seasoning(age) * seasonal(month)

    where:
    - seasoning(age) = min(1, age / 18) * base_smm
    - seasonal(month_of_year) represents a seasonal adjustment based on the month of the year.

    Parameters
    ----------
    smm_dates : pd.DatetimeIndex or array-like
        A Pandas DatetimeIndex representing each date in the loan term. 
        This must be a regular monthly grid.
    base_smm : float
        The base demographic single month mortality rate

    Returns
    -------
    np.ndarray
        An array of demographic factors for each date in the loan term.

    Raises
    ------
    ValueError
        If smm_dates is not a regular monthly grid from start to end.
    """
    # Ensure smm_dates is a Pandas DatetimeIndex
    if not isinstance(smm_dates, pd.DatetimeIndex):
        smm_dates = pd.to_datetime(smm_dates)

    # Check that smm_dates is a monthly date grid from start to end date; raise an error if not
    if np.any(smm_dates != create_regular_dates_grid(smm_dates[0], smm_dates[-1], 'm')):
        raise ValueError(f"smm_dates must be a monthly grid. This grid was input: {smm_dates}")

    # Calculate ages in months from the reference date
    ages_in_months = np.arange(len(smm_dates))

    # Seasoning factors based on age in months
    seasoning_factors = np.minimum(1, ages_in_months / 18) * base_smm

    # Calculate month of year for each date in the term
    months_of_year = smm_dates.month

    # Use months_of_year to index directly into the precomputed seasonal factors array
    # Subtract 1 from months_of_year because array indices are 0-based
    seasonal_factors = SEASONAL_FACTORS_ARRAY[months_of_year - 1]

    # Calculate demographic factors
    demo_factors = seasoning_factors * seasonal_factors

    return demo_factors

def calculate_smms(pccs, pcc_dates, smm_dates, coupon, lag_months=0):
    """
    Calculate Single Monthly Mortality (SMM) rates based on refinancing incentives and demographic factors.

    Parameters
    ----------
    pccs : np.ndarray or list
        Primary Current Coupon values (1D or 2D array).
    pcc_dates : pd.DatetimeIndex, list, or np.ndarray
        Dates corresponding to the columns of pccs
    smm_dates : pd.DatetimeIndex, list, or np.ndarray
        Dates for the loan term as a regular monthly grid.
    coupon : float
        The coupon rate of the MBS.
    lag_months : int, optional
        Number of months to lag PCC values. Default is 0 (no lag).

    Returns
    -------
    np.ndarray
        A 2D array of SMM rates.
    """
    # Ensure `smm_dates` is a Pandas DatetimeIndex
    if not isinstance(smm_dates, pd.DatetimeIndex):
        smm_dates = pd.to_datetime(smm_dates)

    # Convert `pccs` to a numpy array and handle dimensionality
    pccs = np.asarray(pccs)
    is_1d = pccs.ndim == 1

    if is_1d == 1:
        pccs = pccs.reshape(1, -1)  # Temporarily make it 2D for processing
    elif pccs.ndim != 2:
        raise ValueError("pccs must be a 1D or 2D array.")
    
    # Validate dimensions
    if pccs.shape[1] != len(pcc_dates):
        raise ValueError("The number of columns in `pccs` must match the length of `pcc_dates`.")
    
    # Use searchsorted to find indices for each smm date
    indexes = np.searchsorted(pcc_dates, smm_dates, side='right') - 1
    pccs = pccs[:, indexes]

    # Apply lag
    if lag_months > 0:
        if lag_months >= pccs.shape[1]:
            raise IndexError("Lag exceeds the number of available months in `pccs`.")
        pccs = lag_2darray(pccs, lag_months)

    # Calculate refinancing and demographic factors
    spread = coupon - pccs
    refi_factors = refi_strength(spread)  # Refinancing incentives
    demo_factors = demo(smm_dates)  # Demographic factors

    # Final SMM calculation
    smms = refi_factors + demo_factors

    return smms.flatten() if is_1d else smms # Return smms to original shape if input was 1D else return smms
