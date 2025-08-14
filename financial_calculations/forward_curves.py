import numpy as np
import pandas as pd
from scipy.optimize import minimize
from financial_calculations.bonds import (
    SemiBondContract,
    create_semi_bond_cash_flows,
    calculate_coupon_rate
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

        # Create an instance of StepDiscounter with an additional date and rate
        # to be modified in the objective function
        temp_rate_vals = np.append(rate_vals, initial_guess)
        discounter = StepDiscounter(rate_dates, temp_rate_vals)

        # Define the objective function for minimization
        def objective(rate):
            # Set the last temp rate val to the current rate
            temp_rate_vals[-1] = rate[0]
            discounter.set_rates(temp_rate_vals)
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

def calibrate_fine_curve(market_close_date, cmt_data, balance=100, frequency='m', initial_guess=0.04, smoothing_error_weights=[10.0, 10.0]):
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
    smoothing_error_weights : list, optional
        A list of two penalty weighties applied to discourage large first and second order jumps in discount rates.
        Default [10.0, 10.0].

    Returns
    -------
    StepDiscounter
        A StepDiscounter instance representing the calibrated forward curve based on the bootstrapped rates.

    Raises
    ------
    ValueError
        If the length of smoothing_error_weights != 2.
        If there are duplicate maturity dates in `cmt_data`.
        If the minimization process fails to converge.
    """
    # Check that the correct amount of smoothing error weights are input
    if len(smoothing_error_weights) != 2:
        raise ValueError(f"The length of smoothing_error_weights is not 2. {len(smoothing_error_weights)} was input instead.")
    
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

    # Loop through the sorted bond data (effective date, maturity years, and coupon rates) to compute and store cash flows
    semi_bond_flows_list = []
    for effective_date, maturity_years, coupon in sorted_cmt_data:
        # Create bond instance and generate its cash flows
        semi_bond = SemiBondContract(effective_date, maturity_years * 12, coupon, balance)
        semi_bond_flows = create_semi_bond_cash_flows(semi_bond)
        semi_bond_flows_list.append(semi_bond_flows)

    # Define an instance of Stepdiscounter to use for bond pricing
    discounter = StepDiscounter(rate_dates, np.zeros(len(rate_dates)))

    def calculate_price_errors(forward_rates):
        # Initialize the price error list
        price_errors = []

        # Loop through the pre-computed semi bond cash flows
        for semi_bond_flows in semi_bond_flows_list:
            # Discount the bond's cash flows using the given rates
            discounter.set_rates(forward_rates)
            price = value_cash_flows(discounter, semi_bond_flows, market_close_date)

            # Calculate the difference between calculated price and target balance and append it to the price errors list
            price_error = np.abs(price - balance)
            price_errors.append(price_error)

        return np.asarray(price_errors)
    
    def calculate_coupon_errors(forward_rates):
        # Initialize the coupon error list
        coupon_errors = []

        # Loop through the sorted CMT data
        for effective_date, maturity_years, coupon in sorted_cmt_data:
            # Calculate the theoretical coupon based on the current set of rates
            discounter.set_rates(forward_rates)
            theoretical_coupon = calculate_coupon_rate(effective_date, maturity_years, discounter)

            # Calculate the current coupon error and append it to the coupon errors list
            coupon_error = np.abs(coupon - theoretical_coupon)
            coupon_errors.append(coupon_error)

        return np.asarray(coupon_errors)

    # Define the objective function for optimization
    def objective(rates):
        # Calculate the sum of the squared price and coupon errors
        # Note that the couppn error is calculated as a percent as to be more relative to the other numbers
        coupon_error_pct_sq = np.sum((100*calculate_coupon_errors(rates))**2)
        price_error_sq = np.sum(calculate_price_errors(rates)**2)

        # Define a penalty to discourage large jumps in consecutive discount rates
        first_order_jumps = np.diff(rates)
        second_order_jumps = np.diff(first_order_jumps)
        smoothing_error_sq = smoothing_error_weights[0] * np.sum(first_order_jumps**2) + smoothing_error_weights[1] * np.sum(second_order_jumps**2)

        # Return the sum of the coupon, price, and smoothing error terms
        return coupon_error_pct_sq + price_error_sq + smoothing_error_sq

    # Perform the minimization using the scipy.optimize minimize function
    rates_length = len(rate_dates)
    result = minimize(objective, x0=np.ones(rates_length) * initial_guess)

    # Raise an error if optimization did not converge
    if result.success:
        rates = result.x
    else:
        raise ValueError("Minimization did not converge. Try adjusting the initial guess or checking input data.")

    # Return the StepDiscounter with calibrated dates and rates
    return StepDiscounter(rate_dates, rates)
