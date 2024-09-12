import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize
from utils import (
    discount_cash_flows
)
from financial_calculations.bond_cash_flows import (
    create_semi_bond_cash_flows
)

def calculate_coupon_rate(start_date, maturity_years, par_value, forward_curve, initial_guess=0.04):
    """
    Calculate the coupon rate required to produce a par price for a bond.
    
    Parameters:
    -----------
    start_date : datetime or datetime64[D]
        The start date of the bond.
    maturity_years : int
        The maturity years of the bond.
    par_value : float
        The par value of the bond.
    forward_curve : tuple
        A tuple containing two elements:
        - payment_dates (np.ndarray): Array of payment dates as datetime objects.
        - spot_rates (np.ndarray): Array of spot rates corresponding to each payment date.
    initial_guess : float, optional
        Initial guess for the coupon rate. Default is 0.04 (4%).

    Returns:
    --------
    float:
        The coupon rate required to produce a par price for the bond.
    
    Raises:
    -------
    ValueError:
        If the start date is before the market close date.
        If the minimization process does not converge to a solution for the coupon rate.
    """
    spot_rate_dates, spot_rates = forward_curve
    market_close_date = spot_rate_dates[0]

    # If the market close date is input as a datetime64[D] type, convert to datetime to ensure compatibility with the forward curve data
    if isinstance(market_close_date, np.datetime64): 
        market_close_date = market_close_date.astype(datetime)

    # Validate start date
    if start_date < market_close_date:
        raise ValueError("Start date is not on or after the market close date.")
    
    # If start_date is one of the spot rate dates, return the corresponding rate
    elif np.isin(start_date, spot_rate_dates):
        rate_index = np.where(start_date == spot_rate_dates)[0][0]
        return spot_rates[rate_index]

    else:
        # Objective function to minimize the squared difference between bond price and par value
        def objective(coupon_rate):
            # Create semiannual bond cash flows
            payment_dates, cash_flows = create_semi_bond_cash_flows(start_date, par_value, coupon_rate, maturity_years)
            
            # Calculate the bond price by discounting cash flows
            price = discount_cash_flows(payment_dates, cash_flows, spot_rates, spot_rate_dates)
            
            # Objective: the squared difference between bond price and par value
            return (price - par_value)**2

        # Minimize the objective function to find the coupon rate that produces par price
        result = minimize(objective, x0=initial_guess, method='L-BFGS-B', bounds=[(0, 1)], options={'ftol': 1e-4})
        
        if result.success:
            return result.x[0]  # Return the calculated coupon rate
        else:
            raise ValueError("Minimization did not converge for the coupon rate calculation.")
