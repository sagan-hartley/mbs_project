import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize
from utils import (
    discount_cash_flows
)
from financial_calculations.bond_cash_flows import (
    create_semi_bond_cash_flows
)

def bootstrap_forward_curve(cmt_data, market_close_date, par_value, initial_guess=0.04):
    """
    Bootstrap the spot curve using bond data.
    
    Parameters:
    -----------
    cmt_data : list of tuples
        Each tuple contains (maturity_years, coupon_rate) for bonds.
        Coupon rate should be expressed as a decimal (e.g., 5% as 0.05).
    market_close_date : datetime or datetime64[D]
        The market close date from which the cash flows are discounted.
    par_value : float
        The par value of the bond.
    initial_guess : float, optional
        The initial guess for each spot rate, default is 0.03.
        
    Returns:
    --------
    tuple:
        A tuple containing two elements:
        - spot_rate_dates (np.ndarray): Array of spot rate dates.
        - spot_rates (np.ndarray): Array of spot rates corresponding to each spot rate date.
    
    Raises:
    -------
    ValueError:
        If minimization fails to converge for a specific bond.
    """

    spot_rate_dates = np.array([market_close_date])
    spot_rates = np.array([])

    for maturity_years, coupon in cmt_data:
        # Append the spot rate date associated with the current maturity year
        spot_rate_dates = np.append(spot_rate_dates, market_close_date + relativedelta(years=maturity_years))

        # Generate cash flows for the bond
        payment_dates, cash_flows = create_semi_bond_cash_flows(market_close_date, par_value, coupon, maturity_years)

        # Objective function to minimize the squared difference between the bond price and the par value
        def objective(rate: float):
            discount_rates = np.append(spot_rates, rate) # Append the spot rate date associated with the current maturity year
            price = discount_cash_flows(payment_dates, cash_flows, discount_rates, spot_rate_dates)  # Discount the cash flows using the new spot rate

            return (price - par_value)**2 # Return the difference squared as the quantity to be minimized

        # Minimize the objective function using a 'L-BFGS-B' method to find the best spot rate
        result = minimize(objective, x0=initial_guess, method='L-BFGS-B', bounds=[(0, 1)], options={'ftol': 1e-4}) # We set a tolerance level to make sure the minimizer converges

        if result.success:
            spot_rates = np.append(spot_rates, result.x[0])
        else:
            raise ValueError(f"Minimization did not converge for payment date {payment_dates[-1]}.")

    return spot_rate_dates, spot_rates

