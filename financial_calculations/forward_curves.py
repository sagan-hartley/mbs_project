import numpy as np
import pandas as pd
from scipy.optimize import minimize
from financial_calculations.bonds import (
    SemiBondContract,
    create_semi_bond_cash_flows
)
from financial_calculations.cash_flows import (
    StepDiscounter,
    value_cash_flows
)
from utils import (
    create_regular_dates_grid
)

def bootstrap_forward_curve(market_close_date, cmt_data, balance=100, initial_guess=0.04):
    """
    Bootstraps a forward curve by calibrating discount rates for a series of semiannual bonds 
    using cash flow data and market rates.

    Parameters
    ----------
    market_close_date : str or pd.Timestamp
        The date at which the market is closed and the forward curve begins.
    cmt_data : list of tuples
        A list of bond data tuples, each containing:
        (effective_date, maturity_years, coupon).
    balance : float
        The face value or principal balance of the bonds. Default is 100.
    initial_guess : float, optional
        The initial guess for the discount rate, by default 0.04 (4%).

    Returns
    -------
    StepDiscounter
        A StepDiscounter instance representing the forward curve based on the bootstrapped rates.

    Raises
    ------
    ValueError
        If cmt_data contains duplicate maturity dates.
        If the minimization process fails to converge for any bond in the data.
    """
    # Sort cmt_data by effective date + maturity years, if not already sorted
    sorted_cmt_data = sorted(cmt_data, key=lambda x: x[0] + pd.DateOffset(years=x[1]))

    # Initialize rate dates with the market close date as the starting point
    rate_dates = np.array([pd.to_datetime(market_close_date)])
    rate_vals = np.array([])  # Holds the bootstrapped rate values

    # Loop through each bond's data to calibrate and add to the forward curve
    for effective_date, maturity_years, coupon in sorted_cmt_data:

        # If duplicate maturity dates exist, raise an error as the corresponding rate has already been bootstrapped
        maturity_date = effective_date + pd.DateOffset(years=maturity_years)
        if np.any(rate_dates == maturity_date):
            raise(ValueError("Duplicate maturity dates cannot exist for this bootstrapping method."))

        # Create a bond instance and generate its cash flows
        semi_bond = SemiBondContract(effective_date, maturity_years * 12, coupon, balance)
        semi_bond_flows = create_semi_bond_cash_flows(semi_bond)

        # Define the objective function for minimization
        def objective(rate: float):
            # Temporarily append the rate to calculate discount factors
            temp_rate_vals = np.append(rate_vals, rate)
            discounter = StepDiscounter(rate_dates, temp_rate_vals)
            value = value_cash_flows(discounter, semi_bond_flows, market_close_date)
            return (value - balance) ** 2

        # Minimize the objective function to find the rate that matches the bond's market value
        result = minimize(objective, x0=initial_guess, method='L-BFGS-B', options={'ftol': balance * 1e-7})

        # If minimization is successful, add the current rate and to rate_vals and rate_dates
        if result.success:
            rate_vals = np.append(rate_vals, result.x[0])
            rate_dates = np.append(rate_dates, maturity_date)
        else:
            raise ValueError("Minimization did not converge.")

    # Extend the last rate to the end of the curve to ensure the curve is complete
    rate_vals = np.append(rate_vals, rate_vals[-1])

    return StepDiscounter(rate_dates, rate_vals)

def calibrate_fine_curve(market_close_date, cmt_data, balance=100, frequency='m', initial_guess=0.04, smoothing_error_weight=100.0):
    """
    Calibrates a fine forward curve by bootstrapping discount rates for a series of bonds with regular intervals, 
    minimizing the squared error between bond prices and a target balance while applying a smoothing penalty to 
    reduce large rate jumps.

    Parameters
    ----------
    market_close_date : str or pd.Timestamp
        The date at which the market closes and the forward curve begins.
    cmt_data : list of tuples
        A list of bond data tuples, each containing:
        (effective_date (pd.Timestamp), maturity_years (int), coupon (float)).
        Each bond's maturity is calculated as effective_date + maturity_years.
    balance : float
        The face value or principal balance of the bonds. Default is 100.
    frequency : str, optional
        The frequency of the rate grid, by default 'm' for monthly intervals.
    initial_guess : float, optional
        The initial guess for the discount rate, by default 0.04 (4%).
    smoothing_error_weight : float, optional
        A penalty weight applied to discourage large jumps in discount rates, by default 100.0.

    Returns
    -------
    StepDiscounter
        A StepDiscounter instance representing the calibrated forward curve based on the bootstrapped rates.

    Raises
    ------
    ValueError
        If there are duplicate maturity dates in `cmt_data` or if the minimization process fails to converge.
    """
    # Sort the bond data by effective date + maturity years
    sorted_cmt_data = sorted(cmt_data, key=lambda x: x[0] + pd.DateOffset(years=x[1]))

    # Calculate maturity dates for each bond in cmt_data
    maturity_dates = [
        effective_date + pd.DateOffset(years=maturity_years)
        for effective_date, maturity_years, _ in sorted_cmt_data
    ]

    # Check for duplicate maturity dates and raise an error if found
    if len(maturity_dates) != len(set(maturity_dates)):
        raise ValueError("Duplicate maturity dates found in the input data. Ensure each bond has a unique maturity date.")

    # Determine the latest maturity date
    max_maturity_date = max(maturity_dates)

    # Create a grid of rate dates from market close date to the latest maturity date
    rate_dates = create_regular_dates_grid(market_close_date, max_maturity_date, frequency)

    # Define the objective function for optimization: minimizes squared errors between bond prices and balance
    def objective(rates):
        # Initialize the squared error of the bond price
        price_error_sq = 0

        # Loop through the sorted bond data (effective date, maturity years, and coupon rates)
        for effective_date, maturity_years, coupon in sorted_cmt_data:
            # Create bond instance and generate its cash flows
            semi_bond = SemiBondContract(effective_date, maturity_years * 12, coupon, balance)
            semi_bond_flows = create_semi_bond_cash_flows(semi_bond)

            # Discount the bond's cash flows using the given rates
            discounter = StepDiscounter(rate_dates, rates)
            price = value_cash_flows(discounter, semi_bond_flows, market_close_date)

            # Add the squared difference between calculated price and target balance
            price_error_sq += (price - balance) ** 2
        
        # Apply a penalty to discourage large jumps in consecutive discount rates
        smoothing_error_sq = smoothing_error_weight * np.sum(np.diff(rates)**2)

        # Return the total of price squared error and smoothing penalty
        return price_error_sq + smoothing_error_sq

    # Perform the minimization using the L-BFGS-B method, with an initial guess for each rate
    rates_length = len(rate_dates)
    result = minimize(objective, x0=np.ones(rates_length) * initial_guess, method='L-BFGS-B',
                      options={'ftol': balance * rates_length * 1e-7})

    # Raise an error if optimization did not converge
    if result.success:
        rates = result.x
    else:
        raise ValueError("Minimization did not converge. Try adjusting the initial guess or checking input data.")

    # Return the StepDiscounter with calibrated dates and rates
    return StepDiscounter(rate_dates, rates)
