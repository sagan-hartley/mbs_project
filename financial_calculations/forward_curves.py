import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize
from utils import (
    convert_to_datetime,
    discount_cash_flows,
    create_fine_dates_grid,
)
from financial_calculations.bond_cash_flows import (
    create_semi_bond_cash_flows
)

class ForwardCurve:
    """
    A class for constructing and calibrating a forward curve for discounting cash flows. 

    The ForwardCurve class enables both basic and fine-grained calibration of forward rates
    based on input bond data and a specified market close date. It supports:
    - Basic bootstrapping of forward rates to match input bond prices.
    - A fine-grained calibration that penalizes large rate jumps, ensuring smoother
      transitions between rates.

    Parameters:
    -----------
    market_close_date : str, datetime, or datetime64
        The date to which cash flows are discounted, representing the start of the forward curve.
    dates : np.ndarray, optional
        Array of discount rate dates; defaults to containing only `market_close_date`.
    rates : np.ndarray, optional
        Array of forward rates corresponding to each date; defaults to an empty array.

    Attributes:
    -----------
    market_close_date : datetime
        Market close date for the forward curve.
    dates : np.ndarray
        Array of dates representing the forward curve.
    rates : np.ndarray
        Array of discount rates corresponding to each date in the forward curve.
    """
    
    def __init__(self, market_close_date, dates=None, rates=None):
        """
        Initializes the ForwardCurve instance with a market close date, and optionally, specific dates and rates.

        Parameters:
        -----------
        market_close_date : str or datetime
            The market close date as a 'YYYY-MM-DD' string or a datetime object, representing the start date for the forward curve.
        dates : array-like of datetime, optional
            An array of dates for the forward curve. Defaults to an array with only the market close date.
        rates : array-like of float, optional
            An array of forward rates corresponding to each date in `dates`. Defaults to an empty array.

        Notes:
        ------
        - The `market_close_date` is converted to a datetime object if provided as a string or datetime64 object.
        - If `dates` or `rates` are not provided, they are initialized to default values: a single market close date for `dates` and an empty array for `rates`.
        """
        # Convert the market close date attribute to a datetime object to ensure compatibility with calibration methods
        self.market_close_date = convert_to_datetime(market_close_date)

        # If dates and rates attribute values are not provided, initialize ndarrays for each,
        # with self.dates containing a single market close date and self.rates being empty
        self.dates = dates if dates is not None else np.array([self.market_close_date])
        self.rates = rates if rates is not None else np.array([])

    def bootstrap_forward_curve(self, cmt_data, balance, initial_guess=0.04):
        """
        Bootstraps the forward curve using bond data to match bond prices at par value.

        For each bond, this method calculates semiannual cash flows and finds the
        discount rate that equates the bond price to the given balance. Discount rates 
        are sequentially appended to create a forward curve.

        Parameters:
        -----------
        cmt_data : list of tuples
            Each tuple contains (maturity_years, coupon_rate) for bonds.
            Coupon rate should be expressed as a decimal (e.g., 5% as 0.05).
        balance : float
            The balance of the bond.
        initial_guess : float, optional
            The initial guess for each discount rate. Default is 0.04.

        Returns:
        --------
        None

        Raises:
        -------
        ValueError
            If minimization fails to converge for a specific bond.
        """

        # Reset the dates and rates attributes
        self.dates = np.array([self.market_close_date])
        self.rates = np.array([])

        for maturity_years, coupon in cmt_data:
            # Append the date associated with the current maturity year
            self.dates = np.append(self.dates, self.market_close_date + relativedelta(years=maturity_years))

            # Generate cash flows for the bond
            payment_dates, cash_flows = create_semi_bond_cash_flows(self.market_close_date, balance, coupon, maturity_years)

            # Objective function to minimize the squared difference between the bond price and the balance
            def objective(rate: float):
                discount_rates = np.append(self.rates, rate) # Append the rate associated with the current maturity year
                price = discount_cash_flows(payment_dates, cash_flows, discount_rates, self.dates)  # Discount the cash flows using the new rate

                return (price - balance)**2 # Return the difference squared as the quantity to be minimized

            # Minimize the objective function using a 'L-BFGS-B' method to find the best disc rate
            # We set a tolerance level based on the balance to make sure the minimizer converges
            result = minimize(objective, x0=initial_guess, method='L-BFGS-B', options={'ftol': balance * 1e-7}) 

            if result.success:
                self.rates = np.append(self.rates, result.x[0])
            else:
                raise ValueError(f"Minimization did not converge for payment date {payment_dates[-1]}.")

    def calibrate_finer_forward_curve(self, cmt_data, balance, frequency='monthly', initial_guess=0.04, smoothing_error_weight=100.0):
        """
        Calibrates a finer forward curve with a penalty for large rate jumps, using a specified grid frequency.

        This method minimizes the squared error between bond prices and balance, while penalizing
        large jumps in discount rates to ensure smoother transitions between dates in the curve.

        Parameters:
        -----------
        cmt_data : list of tuples
            A list of (maturity_years, coupon_rate) tuples representing bond data.
            e.g. [(1, 0.03), (2, 0.04), (3, 0.05)] for 1, 2, 3 year bonds with 3%, 4%, 5% coupons.
        balance : float
            The balance of the bonds.
        frequency : str, optional
            The frequency of the date grid for the forward curve. Choices are 'monthly' or 'weekly'. Default is 'monthly'.
        initial_guess : float, optional
            The initial guess for the discount rate in the optimization routine. Default is 0.04 (4%).
        smoothing_error_weight : float, optional
            A weight parameter that penalizes large jumps in the discount rates to ensure smoother rate transitions. Default is 100.0.

        Returns:
        --------
        None

        Raises:
        -------
        ValueError
            If the frequency is not 'monthly' or 'weekly', or if the optimization process fails to converge.
        """
        # Set the dates attribute to a finer grid of dates (monthly/weekly) up to the longest bond maturity
        self.dates = create_fine_dates_grid(self.market_close_date, cmt_data[-1][0], frequency)

        # Define the objective function to minimize the squared difference between bond price and balance
        def objective(rates):
            # Initialize the price squared error
            price_error_sq = 0

            # Loop through the CMT data (maturity years and coupon rates)
            for maturity_years, coupon in cmt_data:
                # Generate cash flow dates and amounts for the current bond
                payment_dates, cash_flows = create_semi_bond_cash_flows(self.market_close_date, balance, coupon, maturity_years)
            
                # Calculate the bond price by discounting the cash flows with the new discount rates
                price = discount_cash_flows(payment_dates, cash_flows, rates, self.dates)

                price_error_sq += (price - balance) ** 2 # sum the price squared error
            
            # Apply a penalty to discourage large jumps in the discount rates
            smoothing_error_sq = smoothing_error_weight * np.sum(np.diff(rates)**2)

            # Return the sum of the squared price difference and the penalty
            return price_error_sq + smoothing_error_sq

        # Minimize the objective function using L-BFGS-B method
        # x0 is the initial guess for the disc rate
        rates_length = len(self.dates)
        result = minimize(objective, x0=np.ones(rates_length)*initial_guess, method='L-BFGS-B', 
                        options={'ftol': balance * rates_length * 1e-7})

        # If the optimization converges, set the rates attribute to the result
        if result.success:
            self.rates = result.x
        else:
            raise ValueError("Minimization did not converge.")
