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
    Bootstrap the disc curve using bond data.
    
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
        The initial guess for each discount rate. Default is 0.04.
        
    Returns:
    --------
    tuple:
        A tuple containing two elements:
        - disc_rate_dates (np.ndarray): Array of discount rate dates.
        - disc_rates (np.ndarray): Array of discount rates corresponding to each discount rate date.
    
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
        
    disc_rate_dates = np.array([market_close_date])
    disc_rates = np.array([])

    for maturity_years, coupon in cmt_data:
        # Append the disc rate date associated with the current maturity year
        disc_rate_dates = np.append(disc_rate_dates, market_close_date + relativedelta(years=maturity_years))

        # Generate cash flows for the bond
        payment_dates, cash_flows = create_semi_bond_cash_flows(market_close_date, par_value, coupon, maturity_years)

        # Objective function to minimize the squared difference between the bond price and the par value
        def objective(rate: float):
            discount_rates = np.append(disc_rates, rate) # Append the disc rate date associated with the current maturity year
            price = discount_cash_flows(payment_dates, cash_flows, discount_rates, disc_rate_dates)  # Discount the cash flows using the new disc rate

            return (price - par_value)**2 # Return the difference squared as the quantity to be minimized

        # Minimize the objective function using a 'L-BFGS-B' method to find the best disc rate
        # We set a tolerance level based on the par value to make sure the minimizer converges
        result = minimize(objective, x0=initial_guess, method='L-BFGS-B', bounds=[(0, 1)], options={'ftol': par_value * 1e-7}) 

        if result.success:
            disc_rates = np.append(disc_rates, result.x[0])
        else:
            raise ValueError(f"Minimization did not converge for payment date {payment_dates[-1]}.")
    
    return disc_rate_dates, disc_rates

def bootstrap_finer_forward_curve(cmt_data, market_close_date, par_value, frequency='monthly', initial_guess=0.04, penalty=100.0):
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
        The frequency of the date grid for the forward curve. Choices are 'monthly' or 'weekly'. Default is 'monthly'
    initial_guess : float, optional
        The initial guess for the discount rate in the optimization routine. Default is 0.04 (4%).
    penalty : float, optional
        A penalty parameter that penalizes large jumps in the discount rates to ensure smoother rate transitions. Default is 100.0
    
    Returns:
    --------
    tuple:
        A tuple containing two elements:
        - disc_rate_dates (np.ndarray): Array of finer discount rate dates.
        - disc_rates (np.ndarray): Array of discount rates corresponding to each discount rate date.

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
    disc_rate_dates = create_fine_dates_grid(market_close_date, cmt_data[-1][0], frequency)

    # Initialize an empty array for disc rates
    disc_rates = np.array([])

    # Define the objective function to minimize the squared difference between bond price and par value
    def objective(rates):
        # Initialize the price squared error
        price_error_sq = 0

        # Loop through the CMT data (maturity years and coupon rates)
        for maturity_years, coupon in cmt_data:
            # Generate cash flow dates and amounts for the current bond
            payment_dates, cash_flows = create_semi_bond_cash_flows(market_close_date, par_value, coupon, maturity_years)
            
            # Calculate the bond price by discounting the cash flows with the new discount rates
            price = discount_cash_flows(payment_dates, cash_flows, rates, disc_rate_dates)

            price_error_sq += (price - par_value) ** 2 # sum the price squared error
            
        # Apply a penalty to discourage large jumps in the discount rates
        smoothing_error_sq = penalty * np.sum(np.diff(rates)**2)

        # Return the sum of the squared price difference and the penalty
        return price_error_sq + smoothing_error_sq

    # Minimize the objective function using L-BFGS-B method
    # x0 is the initial guess for the disc rate, bounds ensure rates remain between 0 and 1
    result = minimize(objective, x0=np.ones(len(disc_rate_dates))*initial_guess, 
                          method='L-BFGS-B', bounds=[(0, 1)], 
                          options={'ftol': par_value * 1e-7})

    # If the optimization converges, append the found rate to the disc rates array
    if result.success:
        disc_rates = result.x
    else:
        raise ValueError("Minimization did not converge.")

    return disc_rate_dates, disc_rates
