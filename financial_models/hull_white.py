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

def delta_x_per_step_ou(alpha, sigma, dt_list):
    """
    Compute lattice spacings Δx_i for each step i using the OU-state variance match:
        Var[x_{i+1} | x_i] = (σ^2 / (2α)) * (1 - e^{-2α Δt_i})
        Δx_i = sqrt(3) * sqrt(Var)

    Parameters
    ----------
    alpha : float
        Mean reversion speed.
    sigma : float
        Short-rate volatility.
    dt_list : array-like
        Time step lengths Δt_i in years.

    Returns
    -------
    dx_list : np.ndarray
        Lattice spacing for each step i.
    """
    dt = np.asarray(dt_list, dtype=float)
    if alpha > 1e-12:
        V = sigma * np.sqrt((1.0 - np.exp(-2.0 * alpha * dt)) / (2.0 * alpha))
    else:
        # α → 0 limit
        V = sigma * np.sqrt(dt)
    return np.sqrt(3.0) * V

def build_rate_lattice(r0, dx_list):
    """
    Build a recombining short-rate lattice with variable spacing per layer.

    At layer i (time index i), nodes are:
        r_{i,j} = r0 + j * Δx_i,  j ∈ {-i, ..., +i}.

    Parameters
    ----------
    r0 : float
        Initial short rate at the root.
    dx_list : array-like (N,)
        Per-step lattice spacing Δx_i for i=0..N-1.

    Returns
    -------
    lattice : list[np.ndarray]
        lattice[i] has shape (2*i+1,) with node rates for time i.
        Length of the list is N+1 (from time 0 to time N).
    """
    lattice = [np.array([r0], float)]
    for i, dx in enumerate(np.asarray(dx_list, float), start=1):
        j = np.arange(-i, i + 1)
        lattice.append(r0 + j * dx)
    return lattice

def probs_from_theta(r_nodes_by_step, theta_vals, alpha, sigma, dt_list, dx_list, eps=1e-12):
    """
    Compute trinomial probabilities with variable Δt_i and Δx_i
    using exact moment matching.
    """
    N = len(dt_list)
    pu_list, pm_list, pd_list = [], [], []

    for i in range(N):
        r_i = r_nodes_by_step[i]                         # node rates at time i
        theta_i = float(theta_vals[i])
        mu = (theta_i - alpha * r_i) * float(dt_list[i]) # mean increment
        a2 = sigma**2 * float(dt_list[i]) + mu**2        # variance term + mu^2
        h = float(dx_list[i])                       # Δx_i

        pu = 0.5 * (a2 / (h*h) + mu / h)
        pd = 0.5 * (a2 / (h*h) - mu / h)
        pm = 1.0 - a2 / (h*h)

        # clamp & renormalize
        pu = np.clip(pu, eps, 1.0-eps)
        pd = np.clip(pd, eps, 1.0-eps)
        pm = 1.0 - pu - pd

        pu_list.append(pu)
        pm_list.append(pm)
        pd_list.append(pd)

    return pu_list, pm_list, pd_list

def backward_price(r_nodes_by_step, pu_list, pm_list, pd_list, dt_list, payoff_T):
    """
    Backward induction to price a payoff at maturity.

    payoff_T : array length 2*N+1 giving terminal payoff at maturity nodes.
    """
    N = len(dt_list)
    V_next = payoff_T.astype(float).copy()

    for i in range(N - 1, -1, -1):
        r_i = r_nodes_by_step[i]
        pu, pm, pd = pu_list[i], pm_list[i], pd_list[i]
        Vi = np.empty_like(r_i)

        for idx, r in enumerate(r_i):
            j = idx - i
            up   = (j+1) + (i+1)
            mid  = (j+0) + (i+1)
            down = (j-1) + (i+1)
            cont = pu[idx]*V_next[up] + pm[idx]*V_next[mid] + pd[idx]*V_next[down]
            Vi[idx] = np.exp(-r * float(dt_list[i])) * cont
        V_next = Vi

    return float(V_next[0])
