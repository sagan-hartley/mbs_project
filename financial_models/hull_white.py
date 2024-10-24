import numpy as np
from datetime import datetime
from utils import (
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
    forward_curve : tuple 
        A tuple containing rate_dates and corresponding forward rates (rate_dates, rate_vals).
    alpha : float 
        The mean-reversion rate in the Hull-White model (if 0, a special case is handled).
    sigma : float 
        Volatility of the short rate.
    sim_short_rate_dates : list or ndarray 
        Array of dates for the Monte Carlo simulation.

    Returns:
    -------
    sim_short_rate_dates : ndarray 
        The simulated short rate dates where theta is evaluated.
    theta : ndarray 
        The drift term theta(t) at each simulated short rate date.
    """
    
    rate_dates, rate_vals = forward_curve  # Unpack the forward curve tuple

    # Ensure rate_vals and rate_dates are aligned in length
    if len(rate_vals) != len(rate_dates) and len(rate_vals) != len(rate_dates) - 1:
        raise ValueError("rate_vals should be the same length or one index less than rate_dates")

    # If rate_vals is shorter by one, repeat the last rate for the last date
    if len(rate_vals) == len(rate_dates) - 1:
        rate_vals = np.concatenate([rate_vals, [rate_vals[-1]]])

    # Convert rate_dates and sim_short_rate_dates to numpy datetime64 for vectorized operations
    rate_dates = convert_to_datetime64_array(rate_dates)
    sim_short_rate_dates = convert_to_datetime64_array(sim_short_rate_dates)
    market_close_date = rate_dates[0]

    # Check that all simulated short rate dates are after or equal to the market close date
    if np.any(sim_short_rate_dates < market_close_date):
        raise ValueError("Every short rate date must be on or after the market close date")

    # Calculate the time deltas for the simulated short rate dates (in years)
    sim_short_rate_deltas = (sim_short_rate_dates - market_close_date).astype(float) / DISC_DAYS_IN_YEAR

    # Use step interpolation to find forward rates for the simulated short rate dates
    forward_rates = step_interpolate(rate_dates, rate_vals, sim_short_rate_dates)

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
    Simulate short rate paths using the Hull-White model.
    
    Parameters:
    - alpha (float) : The mean-reversion rate.
    - sigma (float) : Volatility of the short rate.
    - theta (tuple) : Tuple of (dates, theta_values) from the Hull-White model.
    - start_rate (float) : The initial rate for the model.
    - iterations (int): Number of iterations for the Monte Carlo simulation. Default is 1000.
    - antithetic (bool): Boolean used to determine whether anithetic or regular normal sampling should be used for dW. Default is True.
    
    Returns:
    - dates (ndarray) : The dates for each short rate
    - r_all (ndarray) : The simulated short rate paths for each iterations.
    - r_avg (ndarray) : The average simulated short rate path across iterations.
    - r_var (ndarray) : The simulated variance for each short rate step across iterations.
    """
    dates, vals = theta
    num_steps = len(dates)

    # If inputs are datetime objects, convert them to numpy datetime64[D] for vectorization 
    dates = convert_to_datetime64_array(dates)
    
    # Initialize array for storing all simulated short rate paths
    r_all = np.zeros((iterations, num_steps))

    # Set the initial short rate for all iterations
    r_all[:, 0] = start_rate  

    if antithetic:
        # Generate random normal samples for antithetic sampling if true
        half_iterations = iterations // 2
        dW_half = np.random.normal(size=(half_iterations, num_steps - 1))  # shape (half_iterations, num_steps - 1)
    
        # Create antithetic variates
        dW = np.concatenate((dW_half, -dW_half), axis=0)  # shape (iterations, num_steps - 1)

        if iterations % 2 == 1:
            # Add one additional normal sample if the number of iterations is odd
            dW = np.concatenate((dW, np.random.normal(size=(1, num_steps - 1))), axis=0)

    else:
        # Generate random normal increments for the Wiener process for all iterations and all steps if antithetic is False
        dW = np.random.normal(size=(iterations, num_steps - 1))  # shape (iterations, num_steps - 1)

    # Calculate time increments
    dt = (dates[1:] - dates[:-1]).astype(float) / DISC_DAYS_IN_YEAR  # shape (num_steps - 1)

    # Hull-White short rate evolution
    for t in range(1, num_steps):
        # Calculate dr for all iterations at once, ensuring correct shapes
        dr = (vals[t - 1] - alpha * r_all[:, t - 1]) * dt[t - 1] + sigma * np.sqrt(dt[t - 1]) * dW[:, t - 1]
        r_all[:, t] = r_all[:, t - 1] + dr  # Add the calculated dr to the previous rates

    # Calculate the average short rate path
    r_avg = r_all.mean(axis=0)

    # Calculate variance across the short rate paths
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
    - forward_curve (tuple) : A tuple of (dates, forward_rate_values) from the forward curve.
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
