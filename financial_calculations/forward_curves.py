import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize
from utils import (
    discount_cash_flows,
    create_fine_dates_grid
)
from financial_calculations.bond_cash_flows import (
    create_semi_bond_cash_flows
)

def bootstrap_forward_curve(cmt_data, market_close_date, par_value, initial_guess=0.04):
    """
    Bootstrap the spot curve using bond data.
    
    Parameters:
    -----------
    cmt_data : list of tuples
        Each tuple contains (maturity_years, coupon_rate) for bonds.
        Coupon rate should be expressed as a decimal (e.g., 5% as 0.05).
    market_close_date : datetime or datetime64[D]
        The market close date to which the cash flows are discounted.
    par_value : float
        The par value of the bond.
    initial_guess : float, optional
        The initial guess for each spot rate, default is 0.04.
        
    Returns:
    --------
    tuple:
        A tuple containing two elements:
        - spot_rate_dates (np.ndarray): Array of spot rate dates.
        - spot_rates (np.ndarray): Array of spot rates corresponding to each spot rate date.
    
    Raises:
    -------
    ValueError:
        If minimization fails to converge for a specific bond.
    """

    # If the market close date is input as a datetime64[D] type, convert to datetime for relativedelta operations in the future
    if isinstance(market_close_date, np.datetime64): 
        market_close_date_dt = market_close_date.astype(datetime)
        market_close_date = datetime.combine(market_close_date_dt, datetime.min.time()) # This adds the HMS 00:00:00 to the datetime object 
            # which makes it identical to a datetime object initialized with the month, day, and year of the given numpy datetime64[D]
        
    spot_rate_dates = np.array([market_close_date])
    spot_rates = np.array([])

    for maturity_years, coupon in cmt_data:
        # Append the spot rate date associated with the current maturity year
        spot_rate_dates = np.append(spot_rate_dates, market_close_date + relativedelta(years=maturity_years))

        # Generate cash flows for the bond
        payment_dates, cash_flows = create_semi_bond_cash_flows(market_close_date, par_value, coupon, maturity_years)

        # Objective function to minimize the squared difference between the bond price and the par value
        def objective(rate: float):
            discount_rates = np.append(spot_rates, rate) # Append the spot rate date associated with the current maturity year
            price = discount_cash_flows(payment_dates, cash_flows, discount_rates, spot_rate_dates)  # Discount the cash flows using the new spot rate

            return (price - par_value)**2 # Return the difference squared as the quantity to be minimized

        # Minimize the objective function using a 'L-BFGS-B' method to find the best spot rate
        # We set a tolerance level based on the par value to make sure the minimizer converges
        result = minimize(objective, x0=initial_guess, method='L-BFGS-B', bounds=[(0, 1)], options={'ftol': par_value * 1e-7}) 

        if result.success:
            spot_rates = np.append(spot_rates, result.x[0])
        else:
            raise ValueError(f"Minimization did not converge for payment date {payment_dates[-1]}.")

    return spot_rate_dates, spot_rates

def bootstrap_finer_forward_curve(cmt_data, market_close_date, par_value, frequency='monthly', initial_guess=0.04, penalty=1.0):
    """
    Bootstraps a finer forward curve using a specified grid frequency (monthly/weekly).
    This method penalizes large rate jumps to ensure smoother transitions between rates.

    Parameters:
    -----------
    cmt_data : list of tuples
        A list of (maturity_years, coupon_rate) tuples representing bond data.
        e.g. [(1, 0.03), (2, 0.04), (3, 0.05)] for 1, 2, 3 year bonds with 3%, 4%, 5% coupons.
    market_close_date : datetime or np.datetime64
        The market close date for the bonds. If the date is a numpy datetime64[D], it's converted to a datetime object.
    par_value : float
        The par value of the bonds.
    frequency : str, optional
        The frequency of the date grid for the forward curve. Choices are 'monthly' (default) or 'weekly'.
    initial_guess : float, optional
        The initial guess for the spot rate in the optimization routine. Default is 0.04 (4%).
    penalty : float, optional
        A penalty parameter that penalizes large jumps in the discount rates to ensure smoother rate transitions.
    
    Returns:
    --------
    tuple:
        A tuple containing two elements:
        - spot_rate_dates (np.ndarray): Array of finer spot rate dates.
        - spot_rates (np.ndarray): Array of spot rates corresponding to each spot rate date.

    Raises:
    -------
    ValueError:
        If the frequency is not 'monthly' or 'weekly'
        If the optimization process fails to converge for a specific bond.
    """

    # If market_close_date is numpy datetime64, convert to datetime for use with relativedelta
    if isinstance(market_close_date, np.datetime64): 
        market_close_date_dt = market_close_date.astype(datetime)
        market_close_date = datetime.combine(market_close_date_dt, datetime.min.time())  # Ensure time is set to 00:00:00

    # Create a finer grid of dates (monthly/weekly) up to the longest bond maturity
    spot_rate_dates = create_fine_dates_grid(market_close_date, cmt_data[-1][0], frequency)

    # Initialize an empty array for spot rates
    spot_rates = np.array([])

    # Determine frequency multiplier (e.g., 12 for monthly, 52 for weekly)
    if frequency == 'monthly':
        intervals_per_year = 12
    elif frequency == 'weekly':
        intervals_per_year = 52
    else:
        raise ValueError("Invalid frequency. Choose 'monthly' or 'weekly'.")

    prev_years = 0  # Track the previous bond's maturity year

    # Loop through the CMT data (maturity years and coupon rates)
    for maturity_years, coupon in cmt_data:

        # Calculate the time difference between the current bond and the previous bond in years
        curr_time_diff = maturity_years - prev_years
        prev_years = maturity_years

        # Find the index right after the current maturity date
        curr_date_stop_index = np.searchsorted(spot_rate_dates, market_close_date + relativedelta(years=maturity_years))

        # Index through finer grid dates until the current bond's maturity
        curr_spot_rate_dates = spot_rate_dates[:curr_date_stop_index]

        # Generate cash flow dates and amounts for the current bond
        payment_dates, cash_flows = create_semi_bond_cash_flows(market_close_date, par_value, coupon, maturity_years)

        # Define the objective function to minimize the squared difference between bond price and par value
        def objective(rates):
            # Extend spot rates with the rates being optimized
            discount_rates = np.concatenate([spot_rates, np.ones(curr_time_diff * intervals_per_year) * rates])
            
            # Calculate the bond price by discounting the cash flows with the new discount rates
            price = discount_cash_flows(payment_dates, cash_flows, discount_rates, curr_spot_rate_dates)
            
            # Apply a penalty to discourage large jumps in the discount rates
            penalty_term = penalty * np.sum(np.diff(discount_rates)**2)

            # Return the sum of the squared price difference and the penalty
            return (price - par_value)**2 + penalty_term

        # Minimize the objective function using L-BFGS-B method
        # x0 is the initial guess for the spot rate, bounds ensure rates remain between 0 and 1
        result = minimize(objective, x0=np.full(curr_time_diff * intervals_per_year, initial_guess), 
                          method='L-BFGS-B', bounds=[(0, 1)], 
                          options={'ftol': par_value * 1e-7})

        # If the optimization converges, append the found rate to the spot rates array
        if result.success:
            spot_rates = np.concatenate([spot_rates, result.x])
            print(result.x)
        else:
            raise ValueError(f"Minimization did not converge for payment date {payment_dates[-1]}.")

    return spot_rate_dates, spot_rates

