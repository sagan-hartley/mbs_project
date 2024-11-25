import numpy as np
import pandas as pd
from utils import create_regular_dates_grid

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

def calculate_pccs(short_rates, short_rate_dates, cash_flow_dates, spread=0.04):
    """
    Calculate the Primary Current Coupons (PCCs) by adding a fixed spread to the short rates.

    Parameters:
    - short_rates (array-like): 1D or 2D array of short rates from the Hull-White model simulation.
    - short_rate_dates (DatetimeIndex or array-like): Dates corresponding to each short rate in the simulation.
    - cash_flow_dates (DatetimeIndex or array-like): Cash flow dates for which PCCs need to be calculated.
    - spread (float): Fixed spread to add to short rates. Default is 0.04 (4%).

    Returns:
    - ndarray: PCC values. If `short_rates` was 1D, returns a 1D array; otherwise, returns a 2D array.
    """
    # Convert inputs to numpy arrays or DatetimeIndex as appropriate
    short_rates = np.asarray(short_rates)
    short_rate_dates = pd.to_datetime(short_rate_dates) if not isinstance(short_rate_dates, pd.DatetimeIndex) else short_rate_dates
    cash_flow_dates = pd.to_datetime(cash_flow_dates) if not isinstance(cash_flow_dates, pd.DatetimeIndex) else cash_flow_dates

    # Flag to remember if input was originally 1D
    was_1d = short_rates.ndim == 1
    if was_1d:
        short_rates = short_rates[np.newaxis, :]  # Convert to 2D for processing

    # Use searchsorted to find indices for each cash flow date
    indexes = np.searchsorted(short_rate_dates, cash_flow_dates, side='right') - 1
    result = short_rates[:, indexes] + spread

    # If input was originally 1D, return the first row as a 1D array
    return result[0] if was_1d else result

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

def calculate_smms(pccs, coupon, smm_dates, lag_months=0):
    """
    Calculate the Single Monthly Mortality (SMM) rates based on the Primary Current Coupon (PCC) values,
    the MBS coupon, and demographic factors over time.

    The SMM calculation considers refinancing incentives influenced by the spread between the PCC and the
    coupon rate, as well as demographic factors that vary with time.

    Parameters
    ----------
    pccs : ndarray or list
        A 2D array or list of Primary Current Coupon (PCC) values, where rows represent different scenarios 
        and columns represent months. Each element corresponds to a PCC value for a specific month in a given scenario.
    coupon : float
        The coupon rate of the MBS, representing the fixed interest rate paid to bondholders.
    smm_dates : pd.DatetimeIndex, ndarray, or list
        A Pandas DatetimeIndex or array-like of dates representing the loan's term. This must be a regular monthly grid.
        These dates are used to calculate demographic factors (e.g., age and seasonal adjustments).
    lag_months : int, optional
        The number of months by which to lag the PCC values. The lag introduces delay into the SMM calculation,
        which can be useful to model delayed refinancing effects. Default is 0 (no lag).

    Returns
    -------
    ndarray
        A 2D array of SMM values for each scenario and month. The shape of the array corresponds to the input `pccs` 
        array, with rows representing scenarios and columns representing months.
    
    Raises
    ------
    ValueError
        If the dimensions of `pccs` do not match the length of `smm_dates`.
        If 'pccs' is not a 1d or 2d array.
    IndexError
        If lag_months causes smms_dates[0] + lag_months to be out of bounds of smm_dates.
    """
    
    # Ensure `smm_dates` is a Pandas DatetimeIndex for compatibility with date calculations
    if not isinstance(smm_dates, pd.DatetimeIndex):
        smm_dates = pd.to_datetime(smm_dates)
    
    # Convert `pccs` to a numpy array if it is not already one
    pccs = np.asarray(pccs)
    
    # Validate that the number of columns in `pccs` matches the length of `smm_dates`
    if pccs.ndim == 1:
        # If pccs is a 1D array, check if its length matches the length of smm_dates
        if len(smm_dates) != len(pccs):
            raise ValueError("The length of smm_dates must match the length of pccs.")
    elif pccs.ndim == 2:
        # If pccs is a 2D array, check if the number of columns matches the length of smm_dates
        if len(smm_dates) != pccs.shape[1]:
            raise ValueError("The length of smm_dates must match the number of months in pccs.")
    else:
        raise ValueError("pccs must be a 1D or 2D array.")
    
    # Apply the lag to PCCs if `lag_months` is specified
    if lag_months != 0:
        # Check that `lag_months + 1` does not go out of bounds
        if lag_months + 1 > pccs.shape[1]:
            raise IndexError("Lag index exceeds the number of available months in PCCs.")
        
        # Initialize the lagged PCC array
        # Note that log_months corresponds to the correct pcc index since smm_dates is a regular monthly grid
        lagged_pccs = np.zeros_like(pccs)
        lagged_pccs[:, :lag_months+1] = pccs[:, [0]]  # Repeat the first column for lagged months
        lagged_pccs[:, lag_months:] = pccs[:, :-lag_months]  # Shift the remaining PCCs by lag_months
    else:
        lagged_pccs = pccs
    
    # Calculate the refinancing incentive based on the spread between the coupon and lagged PCCs
    spread = coupon - lagged_pccs
    refi_factors = refi_strength(spread)  # Example function that models refinancing incentives
    
    # Calculate the demographic factors based on `smm_dates`
    demo_factors = demo(smm_dates)  # Example function that applies demographic factors based on time
    
    # Final SMM calculation as the sum of refinancing and demographic factors
    smms = refi_factors + demo_factors

    return smms
