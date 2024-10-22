import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

def calculate_pccs(short_rates, spread=0.04):
    """
    Calculate the Primary Credit Curve (PCC) by adding a fixed spread to the short rates.
    
    Parameters:
    - short_rates (ndarray): Array of short rates from the Hull-White model simulation.
    - spread (float): Fixed spread to add to short rates. Default is 0.04 (4%).
    
    Returns:
    - ndarray: Array of PCC values.
    """
    return short_rates + spread

def calculate_smms(pccs, coupon, market_close_date, settle_date, num_months, alpha=50):
    """
    Calculate the Single Monthly Mortality (SMM) rates based on the PCC and the MBS coupon.
    
    If the PCC is less than the coupon rate, SMM is set to `alpha`. Otherwise, it is 0.
    
    Parameters:
    - pccs (ndarray): Array of PCC values.
    - coupon (float): The coupon rate of the MBS.
    - num_months (int): The length of the MBS in months.
    - alpha (float): The turnover constant to apply when the PCC is less than the coupon. Default is 1.
    
    Returns:
    - ndarray: Array of SMM values.
    
    Raises:
    - ValueError: If the length of `pccs` is less than `num_months`.
    """
    if len(pccs[0]) < num_months:
        raise ValueError("Length of PCCs is less than the specified number of months.")
    
     # Convert to datetime if the dates are input as a numpy datetime64 object
    if isinstance(market_close_date, np.datetime64):
        market_close_date = market_close_date.astype(datetime)
    if isinstance(settle_date, np.datetime64):
        settle_date = settle_date.astype(datetime)

    # calculate the total number of months between the market close date and the settle date
    delta = relativedelta(settle_date, market_close_date)
    total_months_diff = delta.years * 12 + delta.months
    
    # Shorten the PCCs to the specified number of months, starting at the index representing the settle date (the beginning of the accrual dates)
    pccs = pccs[:, total_months_diff:total_months_diff + num_months]  
    
    # Calculate SMMs based on the conditions
    smms = alpha*(coupon - pccs) # Vectorized condition
    smms = np.clip(smms, 0, 1)  # Clip the SMM values to the range [0, 1]
    
    return smms

def calculate_turnover(alpha, beta_1, theta, loan_age):
    """
    Calculate the turnover rate.

    Parameters:
    - alpha (float): The base turnover rate.
    - beta_1 (float): The sensitivity of turnover to loan age.
    - theta (float): The decay factor for loan age.
    - loan_age (float): The current age of the loan in months.

    Returns:
    - float: The calculated turnover rate.
    """
    turnover = alpha - beta_1 * np.exp(-theta * loan_age)
    return turnover

def calculate_seasonality(alpha, month, theta):
    """
    Calculate the seasonality factor based on the month.

    Parameters:
    - alpha (float): The base seasonal factor.
    - month (int): The current month (1-12).
    - theta (float): The seasonal adjustment factor.

    Returns:
    - float: The calculated seasonality factor.
    """
    seasonality = alpha * np.sin((np.pi / 2) * (month + theta - 3) / (3 - 1))
    return seasonality

def calculate_borrower_incentive(chi, beta_1, nu, x):
    """
    Calculate the borrower incentive.

    Parameters:
    - chi (float): The borrower behavior adjustment factor.
    - beta_1 (float): The sensitivity of borrower incentive to changes.
    - nu (float): A factor representing borrower behavior.
    - x (float): A variable affecting borrower incentive.

    Returns:
    - float: The calculated borrower incentive.
    """
    borrower_incentive = np.arctan(chi * np.pi * beta_1 * (nu - np.arctan(x) / np.pi))
    return borrower_incentive

def calculate_burnout(beta_1, loan_age, beta_2, incentive, start_value):
    """
    Calculate the burnout factor.

    Parameters:
    - beta_1 (float): The burnout sensitivity parameter.
    - loan_age (float): The current age of the loan in months.
    - beta_2 (float): The additional burnout adjustment factor.
    - incentive (float): The borrower incentive.
    - start_value (float): The starting value for comparison.

    Returns:
    - float: The calculated burnout factor.
    """
    burnout = np.exp(beta_1 * loan_age + beta_2 * max(incentive, start_value))
    return burnout

def calculate_smm(alpha, beta_1, theta, chi, nu, x, beta_2, loan_age, month, start_value):
    """
    Calculate the SMM (Single Monthly Mortality) based on various factors.

    Parameters:
    - alpha (float): The base turnover rate.
    - beta_1 (float): The sensitivity of turnover and burnout to loan age.
    - theta (float): The decay factor for turnover.
    - chi (float): The borrower behavior adjustment factor.
    - nu (float): A factor representing borrower behavior.
    - x (float): A variable affecting borrower incentive.
    - beta_2 (float): The burnout adjustment factor.
    - loan_age (float): The current age of the loan in months.
    - month (int): The current month (1-12).
    - start_value (float): The starting value for comparison in burnout calculation.

    Returns:
    - float: The calculated SMM.
    """
    # Calculate individual components
    turnover = calculate_turnover(alpha, beta_1, theta, loan_age)
    seasonality = calculate_seasonality(alpha, month, theta)
    borrower_incentive = calculate_borrower_incentive(chi, beta_1, nu, x)
    burnout = calculate_burnout(beta_1, loan_age, beta_2, borrower_incentive, start_value)

    # Calculate the SMM using the components
    smm = (turnover * seasonality) + (borrower_incentive * burnout)
    return smm
