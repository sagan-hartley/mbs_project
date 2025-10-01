import numpy as np
import pandas as pd
from utils import (
    years_from_reference,
    step_interpolate,
    calculate_antithetic_variance
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
    sim_short_rate_dates : Pandas DatetimeIndex
        The simulated short rate dates, converted to a Pandas DatetimeIndex, where theta is evaluated.
    theta_vals : ndarray
        The calculated theta values at each simulated short rate date, representing the drift term 
        in the Hull-White model.

    Raises
    ------
    ValueError
        If any date in sim_short_rate_dates is before the forward curve's market_close_date.
    """
    # Reference start date for calculations
    market_close_date = forward_curve.market_close_date

    # Convert sim_short_rate_dates to datetime64 format for consistency with forward_curve.dates
    sim_short_rate_dates = pd.to_datetime(sim_short_rate_dates)

    # Check that all simulation dates are on or after the market close date
    if np.any(sim_short_rate_dates < market_close_date):
        raise ValueError("Each short rate date must be on or after the market close date")

    # Calculate time differnce in years from the market close date to each simulation date
    sim_short_rate_years = years_from_reference(market_close_date, sim_short_rate_dates)

    # Interpolate forward rates at each simulated short rate date using step interpolation
    forward_rates = step_interpolate(forward_curve.dates, forward_curve.rates, sim_short_rate_dates)

    # Calculate the derivative of the forward rates using the fact that they are a piecewise step function
    dfdt = np.diff(forward_rates) / np.diff(sim_short_rate_years)

    # Calculate theta, note that theta is just dfdt when alpha equals 0
    if alpha == 0:
        theta = dfdt

    else:
        theta = dfdt + alpha * forward_rates[:-1] + (sigma**2 / (2 * alpha)) * (1 - np.exp(-2 * alpha * sim_short_rate_years[:-1]))

    # Append the last element in theta to theta as to match size with sim_short_rate_dates
    theta = np.append(theta, theta[-1])

    return sim_short_rate_dates, theta
    
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
    - dates (Pandas DatetimeIndex): The dates for each short rate.
    - r_all (ndarray): The simulated short rate paths for all iterations.
    - r_avg (ndarray): The average simulated short rate path across iterations.
    - r_var (ndarray): The simulated variance for each short rate step across iterations.
    """
    
    # Check if antithetic is True and iterations is odd
    if antithetic and iterations % 2 != 0:
        raise ValueError("When antithetic=True, the number of iterations must be even.")

    # Unpack the dates and values from theta and get the number of steps by their length
    dates, vals = theta
    num_steps = len(dates)

    # Convert dates to a Pandas DatetimeIndex if necessary
    dates = pd.to_datetime(dates)

    # Initialize array for storing all simulated short rate paths
    r_all = np.zeros((iterations, num_steps))

    # Set the initial short rate for all iterations
    r_all[:, 0] = start_rate

    if antithetic:
        # Handle even iterations by creating pairs of antithetic paths
        half_iterations = iterations // 2

        # Generate random normal samples for antithetic sampling if true
        dW_half = np.random.normal(size=(half_iterations, num_steps - 1))  # shape (half_iterations, num_steps - 1)

        # Create antithetic variates
        dW = np.concatenate((dW_half, -dW_half), axis=0)  # shape (iterations, num_steps - 1)
    else:
        # Generate random normal increments for the Wiener process for all iterations and all steps if antithetic is False
        dW = np.random.normal(size=(iterations, num_steps - 1))  # shape (iterations, num_steps - 1)

    # Calculate time increments
    dt = np.diff(years_from_reference(dates[0], dates))  # shape (num_steps - 1)

    # Hull-White short rate evolution
    for t in range(0, num_steps - 1):
        # Calculate dr for all iterations at once
        dr = (vals[t] - alpha * r_all[:, t]) * dt[t] + sigma * np.sqrt(dt[t]) * dW[:, t]
        r_all[:, t + 1] = r_all[:, t] + dr  # Add the calculated dr to the previous rates

    # Calculate the average short rate path
    r_avg = r_all.mean(axis=0)

    # If antithetic is used, calculate variance for the antithetic paths
    if antithetic:
        # Compute the variance of the combined antithetic variates
        r_var = calculate_antithetic_variance(r_all)

    else:
        # Calculate variance across all short rate paths (no antithetic variates)
        r_var = r_all.var(axis=0)

    return dates, r_all, r_avg, r_var

def hull_white_lattice_simulate(alpha, sigma, theta, start_rate, iterations=1000):
    # Unpack theta dates and values
    dates, vals = theta
    num_steps = len(dates)

    # Convert dates to Pandas datetime if necessary
    dates = pd.to_datetime(dates)

    # Compute year fractions from reference date
    t_years = years_from_reference(dates[0], dates)
    
    # Compute time step sizes dt (ensure correct size)
    dt = np.diff(t_years)  # Shape (num_steps - 1)
    dt = np.insert(dt, 0, dt[0])  # Ensures dt has shape (num_steps)

    # Step size in rate space (vectorized)
    h = sigma * np.sqrt(3 * dt)

    # Expected short rate path r* (vectorized computation)
    exp_decay = np.exp(-alpha * dt)
    r_star = np.cumsum(vals * (1 - exp_decay) / alpha)
    r_star = np.insert(r_star, 0, start_rate)  # Ensures alignment

    # Compute transition probabilities
    p_u = np.zeros(num_steps)
    p_m = np.zeros(num_steps)
    p_d = np.zeros(num_steps)

    for i in range(num_steps):
        r_ij = r_star[i]  # Centered around r*
        sigma_sq_dt = sigma**2 * dt[i]
        alpha_h_sq = (alpha * h[i])**2
        denom = h[i]**2

        p_u[i] = 0.5 * ((sigma_sq_dt + alpha_h_sq) / denom + (alpha * (r_star[i] - r_ij) * dt[i]) / h[i])
        p_m[i] = 1 - (sigma_sq_dt + alpha_h_sq) / denom
        p_d[i] = 0.5 * ((sigma_sq_dt + alpha_h_sq) / denom - (alpha * (r_star[i] - r_ij) * dt[i]) / h[i])

    # Simulate random walks
    r_all = np.zeros((iterations, num_steps))
    r_all[:, 0] = start_rate  # Start each path at r0

    for i in range(1, num_steps):
        # Generate random numbers
        rand_vals = np.random.rand(iterations)

        # Assign movement based on probabilities
        move_up = rand_vals < p_u[i - 1]
        move_mid = (rand_vals >= p_u[i - 1]) & (rand_vals < p_u[i - 1] + p_m[i - 1])
        move_down = rand_vals >= p_u[i - 1] + p_m[i - 1]

        # Update rates based on movements
        r_all[:, i] = r_all[:, i - 1] + move_up * h[i - 1] - move_down * h[i - 1]

    # Calculate the mean and variance of short rate paths
    r_avg = r_all.mean(axis=0)
    r_var = r_all.var(axis=0)

    return (dates, r_all, r_avg, r_var), (p_u, p_m, p_d)

def hull_white_simulate_from_curve(alpha, sigma, forward_curve, short_rate_dates, iterations=1000, antithetic=True):
    """
    Simulates short rate paths using the Hull-White model from a given forward curve.

    Parameters:
    ----------
    - alpha (float) : The mean reversion rate in the Hull-White model.
    - sigma (float) : The volatility of the short rate.
    - forward_curve (StepDiscounter): An instance of StepDiscounter that holds the forward rate data with associated dates.
    - short_rate_dates (array-like) : An array of dates for which the short rate will be simulated.
    - iterations (int) : The number of simulation paths to generate (default: 1000).
    - antithetic (bool) : If True, applies antithetic variates for variance reduction in simulations (default: False).

    Returns:
    -------
    - hw_simulation (tuple) : A tuple containing the results of the Hull-White simulation
    """
    # Calculate theta values based on the forward curve, alpha, and sigma
    theta = calculate_theta(forward_curve, alpha, sigma, short_rate_dates)
    
    # Run the Hull-White simulation using the calculated theta
    hw_simulation = hull_white_lattice_simulate(alpha, sigma, theta, forward_curve.rates[0], iterations)[0]
    
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
    times = years_from_reference(forward_curve.dates[0], forward_curve.dates)

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