import numpy as np
import pandas as pd
from utils import (
    years_from_reference,
    step_interpolate,
    DISC_DAYS_IN_YEAR
)

def calculate_theta(forward_curve, alpha, sigma, sim_short_rate_dates):
    """
    Calculate the drift term theta(t) in the Hull-White model given a forward curve, and return 
    both the simulated short rate dates and the calculated theta values.

    Parameters
    ----------
    forward_curve : StepDiscounter
        An instance of StepDiscounter that holds the forward rate data with associated dates.
    alpha : float
        The mean-reversion rate in the Hull-White model. A special case is handled if alpha is zero.
    sigma : float
        Volatility of the short rate in the Hull-White model.
    sim_short_rate_dates : array-like
        Array of dates for which the theta term is calculated, usually corresponding to 
        simulation dates for short rates.

    Returns
    -------
    sim_short_rate_dates : ndarray
        The simulated short rate dates, converted to datetime64, where theta is evaluated.
    theta_vals : ndarray
        The calculated theta values at each simulated short rate date, representing the drift term 
        in the Hull-White model.

    Raises
    ------
    ValueError
        If any date in sim_short_rate_dates is before the forward curve's market_close_date.
    """
    # Handle cases where forward_curve.rates is one element short by extending the last rate
    # This final rate is assumed to extend from the last date to infinity
    if len(forward_curve.rates) == len(forward_curve.dates) - 1:
        forward_curve.rates = np.concatenate([forward_curve.rates, [forward_curve.rates[-1]]])

    # Reference start date for calculations
    market_close_date = forward_curve.market_close_date

    # Convert sim_short_rate_dates to datetime64 format for consistency with forward_curve.dates
    sim_short_rate_dates = pd.to_datetime(sim_short_rate_dates)

    # Check that all simulation dates are on or after the market close date
    if np.any(sim_short_rate_dates < market_close_date):
        raise ValueError("Each short rate date must be on or after the market close date")

    # Calculate time deltas in years from the market close date to each simulation date
    sim_short_rate_deltas = years_from_reference(market_close_date, sim_short_rate_dates)

    # Interpolate forward rates at each simulated short rate date using step interpolation
    forward_rates = step_interpolate(forward_curve.dates, forward_curve.rates, sim_short_rate_dates)

    # Compute the derivative of forward rates with respect to time (df/dt) using finite differences
    dfdt = np.gradient(forward_rates, sim_short_rate_deltas)

    # Calculate theta values based on whether alpha (mean reversion) is zero
    if alpha == 0:
        # Special case: no mean reversion, theta equals df/dt
        theta_vals = dfdt
    else:
        # General case: mean-reversion is present
        theta_vals = dfdt + alpha * forward_rates + (sigma**2 / (2 * alpha)) * (1 - np.exp(-2 * alpha * sim_short_rate_deltas))

    # Return the simulation dates and their corresponding theta values
    return sim_short_rate_dates, theta_vals

def hull_white_simulate(alpha, sigma, theta, start_rate, iterations=1000, antithetic=True):
    """
    Simulate short rate paths using the Hull-White model with optional antithetic variates.
    
    Parameters:
    - alpha (float): The mean-reversion rate.
    - sigma (float): Volatility of the short rate.
    - theta (tuple): Tuple of (dates, theta_values) from the Hull-White model.
    - start_rate (float): The initial rate for the model.
    - iterations (int): Number of iterations for the Monte Carlo simulation. Default is 1000.
    - antithetic (bool): Boolean flag for using antithetic variates for dW. Default is True.
    
    Returns:
    - dates (ndarray): The dates for each short rate.
    - r_all (ndarray): The simulated short rate paths for all iterations.
    - r_avg (ndarray): The average simulated short rate path across iterations.
    - r_var (ndarray): The simulated variance for each short rate step across iterations.
    """
    # Unpack the dates and value from theta and get the number of steps by their length
    dates, vals = theta
    num_steps = len(dates)

    # Convert dates to a Pandas DateTime index if necessary
    dates = pd.to_datetime(dates)

    # Initialize array for storing all simulated short rate paths
    r_all = np.zeros((iterations, num_steps))

    # Set the initial short rate for all iterations
    r_all[:, 0] = start_rate

    if antithetic:
        # Handle odd iterations by creating an extra path for the leftover sample
        half_iterations = iterations // 2

        # Generate random normal samples for antithetic sampling if true
        dW_half = np.random.normal(size=(half_iterations, num_steps - 1))  # shape (half_iterations, num_steps - 1)

        # Create antithetic variates
        dW = np.concatenate((dW_half, -dW_half), axis=0)  # shape (iterations, num_steps - 1)

        # If iterations is odd, add one extra independent path
        if iterations % 2 == 1:
            extra_dW = np.random.normal(size=(1, num_steps - 1))  # One extra normal sample
            dW = np.concatenate((dW, extra_dW), axis=0)

    else:
        # Generate random normal increments for the Wiener process for all iterations and all steps if antithetic is False
        dW = np.random.normal(size=(iterations, num_steps - 1))  # shape (iterations, num_steps - 1)

    # Calculate time increments
    dt = (dates[1:] - dates[:-1]).days / DISC_DAYS_IN_YEAR  # shape (num_steps - 1)

    # Hull-White short rate evolution
    for t in range(1, num_steps):
        # Calculate dr for all iterations at once
        dr = (vals[t - 1] - alpha * r_all[:, t - 1]) * dt[t - 1] + sigma * np.sqrt(dt[t - 1]) * dW[:, t - 1]
        r_all[:, t] = r_all[:, t - 1] + dr  # Add the calculated dr to the previous rates

    # Calculate the average short rate path
    r_avg = r_all.mean(axis=0)

    # If antithetic is used, calculate variance for the antithetic paths
    if antithetic:
        # Split into original and antithetic halves
        half_iterations = iterations // 2
        if iterations % 2 == 0:
            # Even case: Use all original and antithetic paths
            r_original = r_all[:half_iterations, :]
            r_antithetic = r_all[half_iterations:, :]  # All remaining paths are antithetic
        else:
            # Odd case: Use original and antithetic paths, excluding the last odd path
            r_original = r_all[:half_iterations, :]
            r_antithetic = r_all[half_iterations:iterations - 1, :]  # Leave out the last path (odd)

        # Compute the average of the antithetic and original paths
        r_combined = (r_original + r_antithetic) / 2

        # If iterations is odd, include the extra path directly into the combined paths
        if iterations % 2 == 1:
            r_combined = np.vstack((r_combined, r_all[-1, :]))

        # Compute the variance of the combined antithetic variates
        r_var = np.var(r_combined, axis=0)

    else:
        # Calculate variance across all short rate paths (no antithetic variates)
        r_var = r_all.var(axis=0)

    return dates, r_all, r_avg, r_var

def hull_white_simulate_from_curve(alpha, sigma, forward_curve, short_rate_dates,
                                   start_rate, iterations=1000, antithetic=True):
    """
    Simulates short rate paths using the Hull-White model from a given forward curve.

    Parameters:
    ----------
    - alpha (float) : The mean reversion rate in the Hull-White model.
    - sigma (float) : The volatility of the short rate.
    - forward_curve (StepDiscounter): An instance of StepDiscounter that holds the forward rate data with associated dates.
    - short_rate_dates (array-like) : An array of dates for which the short rate will be simulated.
    - start_rate (float) : The initial short rate at the starting date.
    - iterations (int) : The number of simulation paths to generate (default: 1000).
    - antithetic (bool) : If True, applies antithetic variates for variance reduction in simulations (default: False).

    Returns:
    -------
    - dates (numpy.ndarray) : The array of dates corresponding to the short rate steps.
    - r_all (numpy.ndarray) : The simulated short rate paths for each iteration (shape: iterations x steps).
    - r_avg (numpy.ndarray) : The average simulated short rate path across iterations (shape: steps).
    - r_var (numpy.ndarray) : The variance of the short rate path across iterations (shape: steps).
    """
    # Calculate theta values based on the forward curve, alpha, and sigma
    theta = calculate_theta(forward_curve, alpha, sigma, short_rate_dates)
    
    # Run the Hull-White simulation using the calculated theta
    hw_simulation = hull_white_simulate(alpha, sigma, theta, start_rate, iterations, antithetic)
    
    return hw_simulation
