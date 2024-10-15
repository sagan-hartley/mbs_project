import numpy as np
from datetime import datetime

HW_DAYS_IN_YEAR = 365.25

def calculate_theta(forward_curve, alpha, sigma, time_points):
    """
    Calculate the theta(t) term in the Hull-White model given a forward curve and return both the time points and theta.
    
    Parameters:
    - forward_curve (tuple) : A tuple containing dates and their corresponding forward rates (rate_dates, rate_vals).
    - alpha (float) : The mean-reversion rate (if 0, special case is handled).
    - sigma (float) : Volatility of the short rate.
    - time_points (list) : Array of times to evaluate the forward curve.
    
    Returns:
    - time_points (ndarray) : The time points where theta is evaluated.
    - theta (ndarray): The drift term theta(t) at each time point.
    """
    rate_dates, rate_vals = forward_curve # Extract rate dates and values from the forward curve tuple

    # Check to make sure the lengths of rate_vals and rate_dates are compatible
    if len(rate_vals) != len(rate_dates) and len(rate_vals) != len(rate_dates) - 1:
        raise ValueError("Rate_vals should be the same length or one index less than rate_dates")

    # If a rate is not user input for the last rate date, concatenate with the last rate value
    if len(rate_vals) == len(rate_dates) - 1:
        rate_vals = np.concatenate([rate_vals, [rate_vals[-1]]])

    # Define the market close date and convert rate_dates and time_points to numpy arrays for efficient operations
    # If inputs are datetime objects, convert them to type datetime64[D] for vectorization
    if isinstance(rate_dates[0], datetime):
        market_close_date = np.datetime64(rate_dates[0], 'D')
        rate_dates = np.array(rate_dates, dtype = 'datetime64[D]')
        time_points = np.array(time_points, dtype = 'datetime64[D]')
    else:
        market_close_date = rate_dates[0]
        rate_dates = np.array(rate_dates)
        time_points = np.array(time_points)

    # Filter the time points array to ensure that only positive time point deltas remain
    time_points = time_points[time_points >= market_close_date]

    # calculate the time deltas for the forward curve and time point dates
    rate_time_deltas = (rate_dates - market_close_date).astype(float) / HW_DAYS_IN_YEAR
    time_point_deltas = (time_points - market_close_date).astype(float) / HW_DAYS_IN_YEAR

    # Use np.interp for linear interpolation of forward rates
    forward_rates = np.interp(time_point_deltas, rate_time_deltas, rate_vals)
    
    # Numerical derivative of forward rates w.r.t time (finite differences)
    dfdt = np.gradient(forward_rates, time_point_deltas)
    
    if alpha == 0:
        # Special case when alpha is 0 (no mean reversion)
        theta = dfdt
    else:
        # General case for alpha > 0
        theta = dfdt + alpha * forward_rates + (sigma**2 / (2 * alpha)) * (1 - np.exp(-2 * alpha * time_point_deltas))
    
    # Return both time points and the corresponding theta values
    return time_points, theta

def hull_white_simulate(alpha, sigma, theta, start_rate, iterations=1000, antithetic=True):
    """
    Simulate short rate paths using the Hull-White model.
    
    Parameters:
    - alpha (float) : The mean-reversion rate.
    - sigma (float) : Volatility of the short rate.
    - theta (tuple) : Tuple of (dates, theta_values) from the Hull-White model.
    - start_rate (float) : The initial rate for the model.
    - iterations (int): Number of iterations for the Monte Carlo simulation. Default is 1000.
    
    Returns:
    - dates (ndarray) : The dates for each short rate
    - r_avg (ndarray) : The average simulated short rate path across iterations.
    """
    dates, vals = theta
    num_steps = len(dates)

    # If inputs are datetime objects, convert them to numpy datetime64[D] for vectorization 
    if isinstance(dates[0], datetime):
        dates = np.array(dates, dtype='datetime64[D]')
    
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
    else:
        # Generate random normal increments for the Wiener process for all iterations and all steps if antithetic is False
        dW = np.random.normal(size=(iterations, num_steps - 1))  # shape (iterations, num_steps - 1)

    # Calculate time increments
    dt = (dates[1:] - dates[:-1]).astype(float) / HW_DAYS_IN_YEAR  # shape (num_steps - 1)

    # Hull-White short rate evolution
    for t in range(1, num_steps):
        # Calculate dr for all iterations at once, ensuring correct shapes
        dr = (vals[t - 1] - alpha * r_all[:, t - 1]) * dt[t - 1] + sigma * np.sqrt(dt[t - 1]) * dW[:, t - 1]
        r_all[:, t] = r_all[:, t - 1] + dr  # Add the calculated dr to the previous rates

    # Calculate the average short rate path
    r_avg = r_all.mean(axis=0)  

    # Calculate variance across the short rate paths
    r_var = r_all.var(axis=0)
    
    return dates, r_avg, r_var