import numpy as np
import pandas as pd
from utils import (
    years_from_reference,
    step_interpolate,
    calculate_antithetic_variance
)
from financial_calculations.cash_flows import (
    StepDiscounter
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
    Build a Hull-White short-rate lattice using the k-matching (drift-aligned)
    method for the OU process.

    Parameters
    ----------
    forward_curve : ForwardCurve
        A ForwardCurve object with attributes:
            - dates : list of datetime objects (same length as rates)
            - rates : list of floats (instantaneous forward rates in decimals)
    alpha : float
        Mean reversion speed of the Hull-White model.
    sigma : float
        Volatility of the short rate.

    Returns
    -------
    x_lattice : list of np.ndarray
        Zero-mean OU state lattice. Slice i is an array of x-values at time step i.
        Slice widths can vary (not fixed 2i+1) to ensure all children fit.
    r_lattice : list of np.ndarray
        Short-rate lattice, r = x + φ(t), with the same geometry as x_lattice.
    dx_list : np.ndarray
        Lattice spacings Δx_i for each step (from delta_x_per_step_ou).
    dt_list : np.ndarray
        Time step lengths Δt_i in years.
    """
    # --- Step 1: φ(t) and time grid ---
    times, phi = phi_from_forward(forward_curve, alpha, sigma)
    dt_list = np.diff(times)
    N = len(dt_list)

    # --- Step 2: OU state step sizes ---
    dx_list = delta_x_per_step_ou(alpha, sigma, dt_list)

    # --- Step 3: Build zero-mean x-lattice via k-matching ---
    x_lattice = [np.array([0.0], dtype=float)]  # root node at x=0
    for i in range(1, N + 1):
        prev_x = x_lattice[i - 1]
        dx = dx_list[i - 1]
        dt = dt_list[i - 1]

        # Conditional mean projection for each parent node
        M = prev_x * np.exp(-alpha * dt)

        # Nearest "middle" node in the next slice
        k_raw = np.rint(M / dx).astype(int)

        # Ensure the new slice covers all children (k−1, k, k+1) for every parent
        k_min = k_raw.min()
        k_max = k_raw.max()
        new_k_start = k_min - 1
        new_k_end = k_max + 1

        # Create the next slice centered on 0, wide enough to fit all children
        new_k_vals = np.arange(new_k_start, new_k_end + 1, dtype=int)
        new_slice = new_k_vals.astype(float) * dx

        x_lattice.append(new_slice)

    # --- Step 4: Build r-lattice by adding φ(t) ---
    r_lattice = [x_lattice[0] + phi[0]]
    for i in range(1, N + 1):
        r_lattice.append(x_lattice[i] + phi[i])

    return x_lattice, r_lattice, dx_list, dt_list

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

class HullWhiteLattice:
    """
    Hull-White trinomial lattice for short-rate modeling and pricing.
    """

    def __init__(self, forward_curve, alpha, sigma):
        """
        Initialize the lattice.

        Parameters
        ----------
        forward_curve : ForwardCurve
            Object with .dates and .rates (same length).
        alpha : float
            Mean reversion speed.
        sigma : float
            Short-rate volatility.
        """
        self.forward_curve = forward_curve
        self.alpha = alpha
        self.sigma = sigma

        # build lattice
        (self.x_lattice,
         self.r_lattice,
         self.dx_list,
         self.dt_list) = build_rate_lattices(forward_curve, alpha, sigma)

        # build probabilities
        self.pu_list, self.pm_list, self.pd_list = probs_from_nu(self.x_lattice, alpha, self.dt_list, self.dx_list)

    def backwards_price(self, target_step):
        """
        Compute discount factors at a given slice via backward induction (k-matched).

        Parameters
        ----------
        target_step : int
            Slice index at which to extract discount factors (0 = root, N = maturity).

        Returns
        -------
        discounts : np.ndarray
            Vector of discount factors aligned with r_lattice[target_step].
        """

        N = len(self.dt_list)
        # terminal payoff: 1 at every maturity node (use actual last slice width)
        V_next = np.ones_like(self.r_lattice[-1], dtype=float)

        for i in range(N - 1, -1, -1):
            r_i = self.r_lattice[i]
            dt  = float(self.dt_list[i])
            pu  = self.pu_list[i]
            pm  = self.pm_list[i]
            pd  = self.pd_list[i]

            # recompute k-indices (mean-reversion alignment) for this step
            dx_next = self.dx_list[i]
            M = self.x_lattice[i] * np.exp(-self.alpha * dt)
            k = np.rint(M / dx_next).astype(int)

            # The next slice was built wide enough to include k±1; map to 0-based indices:
            # next slice x-values are a uniform grid: x_{i+1,ℓ} = (ℓ + off) * dx_next
            # We can locate index 0 by finding where x == min(x_{i+1,*}) = k_min*dx_next
            k_min = (self.x_lattice[i+1] / dx_next).round().astype(int).min()
            k0 = -k_min  # 0-based offset
            k_arr = k + k0

            # discount at current nodes
            disc = np.exp(-r_i * dt)

            # continuation from k-1, k, k+1 at the next slice
            Vi = np.empty_like(r_i)
            for idx in range(len(r_i)):
                kc = k_arr[idx]
                cont = (pu[idx] * V_next[kc + 1] +
                        pm[idx] * V_next[kc + 0] +
                        pd[idx] * V_next[kc - 1])
                Vi[idx] = disc[idx] * cont

            V_next = Vi
            if i == target_step:
                return V_next.copy()

        return np.array([V_next[0]], dtype=float)

    def arrow_debreu_with_k_matching(self):
        """
        Forward-propagate Arrow–Debreu state prices Ψ_{i,*} honoring k-node alignment.

        Returns
        -------
        psi_list : list[np.ndarray]
            psi_list[i] sums to the discount factor P(0, t_i).
        """

        N = len(self.dt_list)
        psi_list = [np.array([1.0], dtype=float)]  # Ψ_{0,0} = 1

        for i in range(N):
            r_i  = self.r_lattice[i]
            dt   = float(self.dt_list[i])
            pu   = self.pu_list[i]
            pm   = self.pm_list[i]
            pd   = self.pd_list[i]
            dx_n = self.dx_list[i]

            # size next slice correctly (variable widths)
            psi_next = np.zeros_like(self.r_lattice[i+1], dtype=float)

            # k-indices via mean-reversion alignment
            M = self.x_lattice[i] * np.exp(-self.alpha * dt)
            k = np.rint(M / dx_n).astype(int)

            # map to 0-based indices for the next slice
            k_min = (self.x_lattice[i+1] / dx_n).round().astype(int).min()
            k0 = -k_min
            k_arr = k + k0

            disc = np.exp(-r_i * dt)
            for idx, psi_ij in enumerate(psi_list[i]):
                w = psi_ij * disc[idx]
                kc = k_arr[idx]
                psi_next[kc + 1] += w * pu[idx]
                psi_next[kc + 0] += w * pm[idx]
                psi_next[kc - 1] += w * pd[idx]

            psi_list.append(psi_next)

        return psi_list
    
    def forward_probabilities(self):
        """
        Forward-propagate undiscounted path probabilities q_{i,*} (risk-neutral mass).

        Returns
        -------
        q_list : list[np.ndarray]
            q_list[i] sums to 1 for all i.
        """

        N = len(self.dt_list)
        q_list = [np.array([1.0], dtype=float)]

        for i in range(N):
            pu   = self.pu_list[i]
            pm   = self.pm_list[i]
            pd   = self.pd_list[i]
            dx_n = self.dx_list[i]

            q_next = np.zeros_like(self.r_lattice[i+1], dtype=float)

            # k-indices for routing
            M = self.x_lattice[i] * np.exp(-self.alpha * float(self.dt_list[i]))
            k = np.rint(M / dx_n).astype(int)
            k_min = (self.x_lattice[i+1] / dx_n).round().astype(int).min()
            k0 = -k_min
            k_arr = k + k0

            for idx, mass in enumerate(q_list[i]):
                kc = k_arr[idx]
                q_next[kc + 1] += mass * pu[idx]
                q_next[kc + 0] += mass * pm[idx]
                q_next[kc - 1] += mass * pd[idx]

            q_list.append(q_next)

        return q_list

    def conditional_discount_factors(self):
        """
        Node-wise conditional discount factors DF^{(0->i)}_{i,*} = Ψ_{i,*} / q_{i,*}.

        Returns
        -------
        cond_df : list[np.ndarray]
            For each slice i, an array of conditional discount factors aligned
            with r_lattice[i]. Unreachable nodes (q=0) are np.nan.
        """

        psi = self.arrow_debreu_with_k_matching()
        q   = self.forward_probabilities()

        cond_df = []
        for i in range(len(psi)):
            qi = q[i]
            psii = psi[i]
            out = np.full_like(psii, np.nan, dtype=float)
            nz = qi > 0
            out[nz] = psii[nz] / qi[nz]
            cond_df.append(out)

        return cond_df

    def conditional_forward_curve(self, slice_index, node_index):
        """
        Build a StepDiscounter for the conditional zero curve starting at node (slice_index, node_index).

        This forward-propagates Arrow-Debreu mass from the chosen node through the lattice
        (with k-matching) to obtain conditional discount factors DF(i0->k | j0), then converts
        them to continuously-compounded zero rates Z(τ) = -ln(DF)/τ on the relative maturity axis.
        The returned StepDiscounter is constructed with calendar dates forward_curve.dates[i0:]
        and the corresponding conditional zero rates.

        Parameters
        ----------
        slice_index : int
            Starting slice i0 (0..N). If i0 == N, the curve is a single point with Z(0)=r_{i0,j0}.
        node_index : int
            0-based index of the node on r_lattice[i0].

        Returns
        -------
        StepDiscounter
            StepDiscounter(dates, rates) where:
            - dates = forward_curve.dates[i0:] (pd.DatetimeIndex)
            - rates = conditional zero rates from that node to each future date
        """
        i0 = int(slice_index)
        j0 = int(node_index)
        N = len(self.dt_list)

        if i0 < 0 or i0 > N:
            raise ValueError("slice_index must be between 0 and N inclusive.")
        if j0 < 0 or j0 >= len(self.r_lattice[i0]):
            raise ValueError("node_index out of range for the chosen slice.")

        # Absolute times and relative maturities (years)
        times = np.concatenate(([0.0], np.cumsum(self.dt_list)))
        maturity_years = times[i0:] - times[i0]

        # Calendar dates for i0..N
        dates_slice = pd.DatetimeIndex(self.forward_curve.dates[i0:])

        # Special case: start at maturity
        if i0 == N:
            zero_curve = np.array([self.r_lattice[i0][j0]], dtype=float)
            return StepDiscounter(dates_slice, zero_curve)

        # Precompute mapping offsets for k-matching (s -> s+1): array_index = k(node units) + k0
        k0_offsets = []
        for s in range(N):
            dx_s = self.dx_list[s]
            k_min_next = (self.x_lattice[s+1] / dx_s).round().astype(int).min()
            k0_offsets.append(-k_min_next)

        # Initialize AD mass at (i0, j0)
        psi_curr = np.zeros_like(self.r_lattice[i0], dtype=float)
        psi_curr[j0] = 1.0

        # DF(i0->i0 | j0) = 1
        df_vals = [1.0]

        # Forward propagation from i0 to N
        for s in range(i0, N):
            dt  = float(self.dt_list[s])
            r_s = self.r_lattice[s]
            dx_s = self.dx_list[s]

            # k-matching indices for this step
            M = self.x_lattice[s] * np.exp(-self.alpha * dt)
            k_arr = np.rint(M / dx_s).astype(int) + k0_offsets[s]

            # Discount at current nodes
            disc = np.exp(-r_s * dt)

            # Push discounted mass to next slice
            psi_next = np.zeros_like(self.r_lattice[s+1], dtype=float)
            pu, pm, pd = self.pu_list[s], self.pm_list[s], self.pd_list[s]
            for q, mass in enumerate(psi_curr):
                if mass == 0.0:
                    continue
                w = mass * disc[q]
                kc = k_arr[q]
                psi_next[kc + 1] += w * pu[q]
                psi_next[kc + 0] += w * pm[q]
                psi_next[kc - 1] += w * pd[q]

            psi_curr = psi_next
            df_vals.append(float(psi_curr.sum()))

        # Convert DF -> zero rates without a mask; handle τ=0 explicitly
        df_vals = np.array(df_vals, dtype=float)
        zero_curve = np.empty_like(df_vals)
        zero_curve[0] = self.r_lattice[i0][j0]                 # define Z(0)
        zero_curve[1:] = -np.log(df_vals[1:]) / maturity_years[1:]  # τ>0

        # Construct and return your StepDiscounter(dates, rates)
        return StepDiscounter(dates_slice, zero_curve)
    
    def backward_all_conditional_forwards(self):
        """
        For every slice i = 0..N, compute:
        • node-wise discount factors P(i->k) for all k = i..N (via backward induction),
        • node-wise per-step instantaneous forward rates over [t_{k-1}, t_k].

        Returns
        -------
        discounts_per_slice : list[np.ndarray]
            discounts_per_slice[i] is a 2D array of shape ((N - i + 1), M_i),
            where M_i = number of nodes at slice i.
            Row r = 0..(N-i) contains P(i -> i+r) evaluated at each node j on slice i.
            Row 0 is all ones (P(i->i) = 1).

        forwards_per_slice : list[np.ndarray]
            forwards_per_slice[i] is a 2D array of shape ((N - i), M_i).
            Row r = 0..(N-i-1) contains the instantaneous per-step forward rates over
            [t_{i+r}, t_{i+r+1}] at each node j on slice i:
                f^{(i)}_{i+r -> i+r+1}(j)
            = - ( ln P(i->i+r+1) - ln P(i->i+r) ) / Δt_{i+r}.
            Note: the first row equals the short rates r_{i,*} at slice i.

        Notes
        -----
        • Efficiency: We run one backward pass per target k (k = 0..N). During the pass
        we store P(i->k) simultaneously for all earlier slices i encountered.
        • k-matching:
            M = x_{s-1,*} * exp(-α Δt_{s-1}),  k_node = round(M / Δx_s)
            array index on slice s: k_arr = k_node + k0, where
            k0 = - min(round(x_{s,*} / Δx_s)).
        """

        N = len(self.dt_list)

        # ---- helper: one backward step s -> s-1 (maps V on slice s to V on slice s-1) ----
        def backstep(V_next, s):
            dt   = float(self.dt_list[s-1])
            r_im = self.r_lattice[s-1]
            pu   = self.pu_list[s-1]
            pm   = self.pm_list[s-1]
            pd   = self.pd_list[s-1]
            dx_s = self.dx_list[s-1]

            # k-matching indices for routing into slice s
            M = self.x_lattice[s-1] * np.exp(-self.alpha * dt)                 # projected mean
            k_node = np.rint(M / dx_s).astype(int)                             # node units
            k_min_next = (self.x_lattice[s] / dx_s).round().astype(int).min()  # origin of next grid in node units
            k0 = -k_min_next
            k_arr = k_node + k0                                                # 0-based array indices

            disc = np.exp(-r_im * dt)
            V_im = np.empty_like(r_im, dtype=float)
            for q in range(len(r_im)):
                kc = k_arr[q]
                cont = pu[q] * V_next[kc + 1] + pm[q] * V_next[kc + 0] + pd[q] * V_next[kc - 1]
                V_im[q] = disc[q] * cont
            return V_im

        # ---- allocate output containers for all slices ----
        discounts_per_slice = []
        forwards_per_slice  = []
        for i in range(N + 1):
            Mi = len(self.r_lattice[i])
            D  = np.zeros((N - i + 1, Mi), dtype=float)
            D[0, :] = 1.0  # P(i->i) = 1
            discounts_per_slice.append(D)
            if i < N:
                F = np.zeros((N - i, Mi), dtype=float)
                forwards_per_slice.append(F)
            else:
                forwards_per_slice.append(np.zeros((0, Mi), dtype=float))  # empty for last slice

        # ---- process each target maturity k, one backward pass each ----
        # k = 0 is trivial: only affects slice i=0 row 0 (already set to ones).
        for k in range(1, N + 1):
            # terminal payoff 1 at slice k
            V = np.ones_like(self.r_lattice[k], dtype=float)

            # step back: s goes k, k-1, ..., 1
            # when we land on slice i = s-1, V holds P(i->k) at that slice
            for s in range(k, 0, -1):
                V = backstep(V, s)           # V now lives on slice s-1
                i = s - 1
                discounts_per_slice[i][k - i, :] = V

        # ---- convert discounts to per-step instantaneous forwards at each slice ----
        dt = np.asarray(self.dt_list, dtype=float)
        for i in range(N):
            D = discounts_per_slice[i]  # shape (N - i + 1, Mi)
            F = forwards_per_slice[i]   # shape (N - i,     Mi)
            # rows r = 0..(N-i-1) map to steps [t_{i+r}, t_{i+r+1}]
            for r in range(N - i):
                # f^{(i)}_{i+r -> i+r+1} = - (ln P(i->i+r+1) - ln P(i->i+r)) / Δt_{i+r}
                num = np.log(D[r + 1, :]) - np.log(D[r, :])
                F[r, :] = - num / dt[i + r]
            # sanity: F[0, :] == r_lattice[i] (since D[0,:]=1 and D[1,:]≈exp(-r_i * dt_i))

        return discounts_per_slice, forwards_per_slice
