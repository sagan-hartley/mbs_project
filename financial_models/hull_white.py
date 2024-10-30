import numpy as np
from datetime import datetime
from utils import (
    convert_to_datetime,
    convert_to_datetime64_array,
    step_interpolate,
    DISC_DAYS_IN_YEAR
)

def calculate_theta(forward_curve, alpha, sigma, sim_short_rate_dates):
    """
    Calculate the theta(t) term in the Hull-White model given a forward curve and return both 
    the simulated short rate dates and theta.
    
    Parameters:
    ----------
    - forward_curve (ForwardCurve): A ForwardCurve object with rate dates and rate values as attributes.
    - alpha (float) : The mean-reversion rate in the Hull-White model (if 0, a special case is handled).
    - sigma (float) : Volatility of the short rate.
    - sim_short_rate_dates (array-like) : Array of dates for the Monte Carlo simulation.

    Returns:
    -------
    - sim_short_rate_dates (ndarray) : The simulated short rate dates where theta is evaluated.
    - theta_vals (ndarray) : The drift term theta(t) at each simulated short rate date.
    """
    # If the length of forward_curve.rates is short by one, repeat the last rate for the last date
    # This happens in the coarse curve calibration, but not the fine curve calibration
    if len(forward_curve.rates) == len(forward_curve.dates) - 1:
        forward_curve.rates = np.concatenate([forward_curve.rates, [forward_curve.rates[-1]]])

    # Convert forward_curve.dates and sim_short_rate_dates to numpy datetime64 arrays for vectorized operations
    forward_curve.dates = convert_to_datetime64_array(forward_curve.dates)
    sim_short_rate_dates = convert_to_datetime64_array(sim_short_rate_dates)

    # Convert the market_close_date to datetime64 to ensure compatibility with forward_curve.dates
    market_close_date = convert_to_datetime64_array(forward_curve.market_close_date)

    # Check that all simulated short rate dates are after or equal to the market close date
    if np.any(sim_short_rate_dates < market_close_date):
        raise ValueError("Every short rate date must be on or after the market close date")

    # Calculate the time deltas for the simulated short rate dates (in years)
    sim_short_rate_deltas = (sim_short_rate_dates - market_close_date).astype(float) / DISC_DAYS_IN_YEAR

    # Use step interpolation to find forward rates for the simulated short rate dates
    forward_rates = step_interpolate(forward_curve.dates, forward_curve.rates, sim_short_rate_dates)

    # Calculate the numerical derivative of forward rates w.r.t. time (using finite differences)
    dfdt = np.gradient(forward_rates, sim_short_rate_deltas)

    # Calculate theta values based on whether alpha is zero (no mean reversion) or not
    if alpha == 0:
        # Special case: no mean reversion
        theta_vals = dfdt
    else:
        # General case: mean-reversion is present
        theta_vals = dfdt + alpha * forward_rates + (sigma**2 / (2 * alpha)) * (1 - np.exp(-2 * alpha * sim_short_rate_deltas))

    # Return both simulated short rate dates and the corresponding theta values
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

    # Convert dates to numpy datetime64[D] if necessary
    dates = convert_to_datetime64_array(dates)

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
    dt = (dates[1:] - dates[:-1]).astype(float) / DISC_DAYS_IN_YEAR  # shape (num_steps - 1)

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
    - forward_curve (ForwardCurve): A ForwardCurve object with rate dates and rate values as attributes.
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
