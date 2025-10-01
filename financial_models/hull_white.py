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

def phi_from_forward(forward_curve, alpha, sigma):
    """
    Compute φ(t) shift values for the Hull-White model lattice building function.

    Parameters
    ----------
    forward_curve : ForwardCurve
        A ForwardCurve object with rate dates and rate values as attributes.
    alpha : float
        Mean reversion speed of the Hull-White model.
    sigma : float
        Volatility of the short rate.

    Returns
    -------
    times : np.ndarray
        Time grid in years starting from 0.
    phi : np.ndarray
        Array of φ(t) shift values corresponding to the time grid.
    """
    # Calculate times list from the difference in the forward curve dates from the initial
    times = (forward_curve.dates - forward_curve.dates[0]).astype(float) / DISC_DAYS_IN_YEAR

    f = forward_curve.rates.copy()
    adj = (sigma**2)/(2.0*alpha**2) * (1.0 - np.exp(-alpha*times))**2
    phi = f + adj

    return times, phi

def build_rate_lattices(forward_curve, alpha, sigma):
    """
    Build a Hull-White short-rate lattice using OU-state variance per step.

    Parameters
    ----------
    forward_curve : ForwardCurve
        A ForwardCurve object with rate dates and rate values as attributes.
    alpha : float
        Mean reversion speed of the Hull-White model.
    sigma : float
        Volatility of the short rate.

    Returns
    -------
    r_lattice : list of np.ndarray
        Short-rate lattice. At step i, r_lattice[i] is an array of shape (2*i+1,)
        containing the short rates at that time step.
    x_lattice : list of np.ndarray
        Zero-mean lattice. At step i, x_lattice[i] is an array of shape (2*i+1)
        containing the spread of steps from x_0 = 0 at that time step.
    """
    times, phi = phi_from_forward(forward_curve, alpha, sigma)
    dt_list = np.diff(times)
    N = len(dt_list)

    dx_list = delta_x_per_step_ou(alpha, sigma, dt_list)

    # x-lattice centered at 0
    x_lattice = [np.array([0.0])]
    for i, dx in enumerate(dx_list, start=1):
        j = np.arange(-i, i+1)
        x_lattice.append(j * dx)

    # build r-lattice
    r_lattice = [x_lattice[0] + phi[0]]
    for i in range(1, N+1):
        r_lattice.append(x_lattice[i] + phi[i])

    return r_lattice, x_lattice

def probs_from_nu(x_lattice, alpha, dt_list, dx_list):
    """
    Compute trinomial probabilities using Jamshidian/Hull-White drift alignment.

    Parameters
    ----------
    x_lattice : list of np.ndarray
        Zero-mean lattice with step sizes dx_list.
    alpha : float
        Mean reversion parameter.
    dt_list : array-like of float
        Step lengths Δt_i in years.
    dx_list : array-like of float
        Per-step lattice spacing Δx_i (length N).

    Returns
    -------
    pu_list, pm_list, pd_list : lists of np.ndarray
        Probabilities at each step, arrays of length 2*i+1.
    """
    N = len(dt_list)
    pu_list, pm_list, pd_list = [], [], []

    for i in range(N):
        x_i = x_lattice[i]           # x-values at step i
        dx_next = dx_list[i]         # spacing at step i+1
        V = dx_next / np.sqrt(3.0)   # volatility scale
        dt = dt_list[i]

        pu, pm, pd = [], [], []

        for j, x_ij in enumerate(x_i):
            # Projected mean-reverted state
            M = x_ij * np.exp(-alpha * dt)

            # Nearest child index
            k = int(np.round(M / dx_next))
            x_next_k = k * dx_next

            # Offset
            nu = M - x_next_k

            # Probabilities
            pu_val = 1/6.0 + (nu**2)/(6*V**2) + nu/(2*np.sqrt(3.0)*V)
            pm_val = 2/3.0 - (nu**2)/(3*V**2)
            pd_val = 1.0 - pu_val - pm_val

            pu.append(pu_val)
            pm.append(pm_val)
            pd.append(pd_val)

        pu_list.append(np.array(pu))
        pm_list.append(np.array(pm))
        pd_list.append(np.array(pd))

    return pu_list, pm_list, pd_list

def backwards_price(r_lattice, pu_list, pm_list, pd_list, dt_list, target_step):
    """
    Compute discount factors at a given slice of the lattice using backward induction.

    Parameters
    ----------
    r_lattice : list of np.ndarray
        Short-rate lattice (from build_rate_lattice).
    pu_list, pm_list, pd_list : lists of np.ndarray
        Transition probabilities per node (length N).
    dt_list : array-like
        Step lengths Δt_i in years.
    target_step : int
        The time step index at which to extract discount factors (0 = root, N = maturity).

    Returns
    -------
    discounts : np.ndarray
        Array of discount factors at the chosen step, aligned with r_lattice[target_step].
    """
    print(pu_list[:2])
    N = len(dt_list)
    # start with payoff 1 at maturity
    V_next = np.ones(2*N+1, float)

    # backward induction
    for i in range(N-1, -1, -1):
        r_i, pu, pm, pd = r_lattice[i], pu_list[i], pm_list[i], pd_list[i]
        Vi = np.empty_like(r_i)
        for idx, r in enumerate(r_i):
            j = idx - i
            up   = (j+1) + (i+1)
            mid  = (j  ) + (i+1)
            down = (j-1) + (i+1)
            cont = pu[idx]*V_next[up] + pm[idx]*V_next[mid] + pd[idx]*V_next[down]
            Vi[idx] = np.exp(-r * float(dt_list[i])) * cont
            
        V_next = Vi
        if i == target_step:  # capture the slice
            return V_next.copy()

    # if target_step=0, return root node as array
    return np.array([V_next[0]])
