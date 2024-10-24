from datetime import datetime
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from utils import (
    discount_cash_flows
)
from financial_calculations.forward_curves import (
    bootstrap_forward_curve, 
    calibrate_finer_forward_curve
)
from financial_calculations.mbs_cash_flows import (
    calculate_scheduled_balances,
    calculate_actual_balances,
    evaluate_cash_flows
)
from financial_models.hull_white import (
    hull_white_simulate_from_curve
)
from financial_models.prepayment import (
    calculate_pccs,
    calculate_smms
)

SETTLE_DATE_IDX = 5
ORIGINATION_DATE_IDX = 6

def parse_datetime(date_str):
    """
    Parse a date string into a Python datetime object.

    Parameters:
    date_str (str): The date string in 'MM/DD/YYYY' format.

    Returns:
    datetime: A Python datetime object representing the parsed date.
    """
    return datetime.strptime(date_str, "%m/%d/%Y")

def load_treasury_rates_data(rates_file):
    """
    Loads treasury data from a CSV file and safely evaluate maturity year strings.

    Parameters:
    rates_file (str): Path to the treasury rates data CSV file.

    Returns:
    date (datetime): The effective date of the treasury rates
    maturity_rate_tuples (list): A list of tuples representing the treasury data.
    """
    # Read the original CSV into a DataFrame
    df = pd.read_csv(rates_file)

    # Extract the date
    date_str = df['Date'].iloc[0]
    date = parse_datetime(date_str)

    # Remove the 'Date' column to focus on maturity years and rates
    maturity_year_columns = df.columns[1:]  # All columns except 'Date'
    
    # Extract the rates from the first row (since it's only one row of data)
    rates = df.iloc[0, 1:].tolist()  # All values in the first row except the date
    
    # Create a list of tuples with maturity year (as int) and corresponding rate (as percentage)
    maturity_rate_tuples = [(int(col.split()[0]), rate/100) for col, rate in zip(maturity_year_columns, rates)]
    
    # Return the date and the list of tuples
    return date, maturity_rate_tuples

def load_mbs_data(mbs_file):
    """
    Loads MBS data from a CSV file and safely evaluate list-like strings.

    Parameters:
    mbs_file (str): Path to the mbs data CSV file.

    Returns:
    mbs_data (list): A list of tuples representing the mbs data.
    """
    # Load MBS data from CSV files
    mbs_data = pd.read_csv(mbs_file, skiprows=1, header=None)

    # Apply the datetime parsing to the Settle Date and Origination Date columns
    mbs_data[SETTLE_DATE_IDX] = mbs_data[SETTLE_DATE_IDX].apply(parse_datetime)
    mbs_data[ORIGINATION_DATE_IDX] = mbs_data[ORIGINATION_DATE_IDX].apply(parse_datetime)

    # Convert values to a list
    mbs_data = mbs_data.values.tolist()

    return mbs_data

def init_smms(peak=60, length=180, zeros=False):
    """
    Initializes an array of SMMs based on the given switch and end points

    Parameters:
    peak (int): the point where the SMMs go from increasing to decreasing. Default is 60.
    length (int): the length of the SMMs array. Default is 180
    zeros (bool): a boolean determining if the SMMs should all be set to zero. Default is false

    Returns:
    smms (ndarray): a numpy array containing the SMMs

    """
    # Check if zeros is True and initialize an array of zeros if so
    if zeros is True:
        smms = np.zeros(length)
    else:
        # Initialize values before the peak
        t_less_peak = np.arange(0, peak)
        smm_less_peak = (t_less_peak / peak) * 0.01

        # Initialize values after the peak
        t_greater_equal_peak = np.arange(peak, length)
        smm_greater_equal_peak = 0.015 - (t_greater_equal_peak / (length - peak)) * 0.01

        # Join the two regions
        smms = np.concatenate([smm_less_peak, smm_greater_equal_peak])

    return smms

def plot_forward_curves(coarse_curve, fine_curve):
    """
    Plots the coarse and fine forward curves.

    Parameters:
    coarse_curve (tuple): A tuple of rate dates and rate values for the coarse curve.
    fine_curve (tuple): A tuple of rate dates and rate values for the fine curve.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.step(coarse_curve[0], np.append(coarse_curve[1], coarse_curve[1][-1]), where='post', label='Coarse Curve', color='blue')
    plt.step(fine_curve[0], fine_curve[1], where='post', label='Fine Curve', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Rate')
    plt.title('Forward Curves')
    plt.legend()
    plt.grid()
    plt.show()

def price_mbs_cash_flows(mbs_data, smms, coarse_curve, fine_curve):
    """
    Prices MBS all cash flows.

    Parameters:
    mbs_data (list): A list containing details of the MBSs including ID, balance, number of months, gross and net annual coupons, accrual dates, single monthly mortality (SMM) rate, and payment delay days.
    smms (ndarray): A ndarray containing prepayment rates.
    coarse_curve (tuple): A tuple containing coarse rate dates and corresponding forward rate values.
    fine_curve (tuple): A tuple containing fine rate dates and corresponding forward rate values.
    
    Returns:
    tuple
        The MBS ID, present value of the cash flows, and weighted average life (WAL) of the security for each the coarse and fine curve
    """

    # Initialize results arrays for coarse and fine curves
    coarse_curve_results = []
    fine_curve_results = []

    # Unpack the coarse and fine forward curve rate dates and values
    coarse_rate_dates, coarse_rate_vals = coarse_curve 
    fine_rate_dates, fine_rate_vals = fine_curve 

    # Loop through each mbs and print the ID, coarse price, fine price, and WAL
    for mbs in mbs_data:
        # Unpack the MBS details
        mbs_id, balance, num_months, gross_annual_coupon, net_annual_coupon, settle_date, origination_date, payment_delay = mbs

        # Calculate the necessary scheduled and actual balance data to price the cash flows
        scheduled_balance_data = calculate_scheduled_balances(balance, origination_date, num_months, gross_annual_coupon)
        actual_balance_data = calculate_actual_balances(scheduled_balance_data, smms, net_annual_coupon, payment_delay)

        coarse_curve_result = evaluate_cash_flows(actual_balance_data, settle_date, net_annual_coupon, coarse_rate_vals, coarse_rate_dates)
        fine_curve_result = evaluate_cash_flows(actual_balance_data, settle_date, net_annual_coupon, fine_rate_vals, fine_rate_dates)
        
        print(f"{mbs[0]}: Coarse Price = {coarse_curve_result[2]}, Fine Price = {fine_curve_result[2]}, WAL = {coarse_curve_result[0]} years")

        # Append the results
        coarse_curve_results.append(coarse_curve_result)
        fine_curve_results.append(fine_curve_result)

    return coarse_curve_results, fine_curve_results

def evaluate_mbs_short_rate_paths(mbs_data, short_rates, short_rate_dates):
    """
    Calculate expected values and prices and path standard deviations for MBS based on short rate paths.

    Parameters:
    - mbs_data (list): A list containing details of the MBSs including ID, balance, number of months, gross and net annual coupons, accrual dates, single monthly mortality (SMM) rate, and payment delay days.
    - short_rates (ndarray): Array of short rates from the simulation.
    - short_rate_dates (tuple): Array of dates corresponding to the short rates

    Returns:
    - results (list): A list of dictionaries containing MBS ID, expected value, expected price, and standard deviations for each MBS.
    """
    results = []  # To store results for each MBS
    market_close_date = short_rate_dates[0] # Extract the market clse date from the short rate dates

    # Loop through each MBS in the provided data
    for mbs in mbs_data:
        wals = []  # Initialize a list to store calculated weight average lifes for the current MBS
        vals = []  # Initialize a list to store calculated values for the current MBS
        prices = []  # Initialize a list to store calculated prices for the current MBS

        # Unpack the MBS details
        mbs_id, balance, num_months, gross_annual_coupon, net_annual_coupon, settle_date, origination_date, payment_delay = mbs

        # Calculate the necessary scheduled balance data to value and price the cash flows
        scheduled_balance_data = calculate_scheduled_balances(balance, origination_date, num_months, gross_annual_coupon)

        # Calculate the Primary Current Coupons (PCCs) based on short rates
        pccs = calculate_pccs(short_rates)

        # Calculate Single Monthly Mortality (SMM) rates
        smms = calculate_smms(pccs, gross_annual_coupon, market_close_date, settle_date, num_months)

        # Loop through each SMM to calculate cash flows
        for index, smm in enumerate(smms):
            # Calculate the actual scheduled balances based on the current SMM
            actual_balance_data = calculate_actual_balances(scheduled_balance_data, smm, net_annual_coupon, payment_delay)

            # Evaluate the cash flows for the current MBS using the actual balances, SMM, and short rates data
            eval_results = evaluate_cash_flows(actual_balance_data, settle_date, net_annual_coupon, short_rates[index], short_rate_dates)
            wal, val, price = eval_results # Extract the WAL, present value, and clean price from eval_results
            
            # Append the calculated wal, present value, and price to their respective lists
            wals.append(wal)
            vals.append(val)
            prices.append(price)

        # Calculate the standard deviation of the collected cash flow WALs, present values, and prices
        wal_std = np.array(wals).std()
        val_std = np.array(vals).std()
        price_std = np.array(prices).std()

        # Calculate the mean of the WALs, present values, and prices for the expected cash flow
        expected_wal = np.mean(wals)
        expected_value = np.mean(vals)
        expected_price = np.mean(prices)

        # Print the results for the current MBS
        print(f"{mbs[0]}, Expected WAL: {expected_wal}, WAL Path STD: {wal_std}, Expected Value: {expected_value}, Value Path STD: {val_std}, Expected Price: {expected_price}, Price Path STD: {price_std}")

        # Store the result in a dictionary for structured output
        results.append({
            'mbs_id': mbs[0],  # MBS identifier
            'expected_wal': expected_wal,  # Expected WAL
            'wal_std': wal_std,  # Standard deviation of the WALs
            'expected_value': expected_value,  # Expected cash flow value
            'value_std': val_std,  # Standard deviation of the cash flow values
            'expected_price': expected_price,  # Expected price of the MBS
            'price_std': price_std  # Standard deviation of the cash flow prices
        })

    return results  # Return the list of results for all MBS

def price_zcbs(rate_dates, rate_vals):
    """
    Price zero-coupon bonds using given rate dates and forward curve values (rate_vals).

    Parameters:
    - rate_dates: Array of dates representing the maturities of the ZCBs.
    - rate_vals: Discount rates or yields corresponding to each maturity date.

    Returns:
    - zcb_prices: List of discounted prices for each ZCB.
    """
    zcb_prices = []  # Initialize the list for ZCB prices
    
    # Loop through each date and calculate the price of a ZCB maturing on that date
    for date in rate_dates:
        # Calculate the price by discounting cash flow of 1 occuring on the maturity date
        price = discount_cash_flows([date], [1], rate_vals, rate_dates)
        zcb_prices.append(price)

    return zcb_prices

def plot_hull_white(hull_white, forward_curve, title='Hull-White Average Path vs Forward Curve'):
    """
    Plots the Hull-White simulation results and forward curve.

    Parameters:
    hull_white (tuple): A tuple of rate dates and rate values for the Hull-White simulation.
    forward_curve (tuple): A tuple of rate dates and rate values for the forward curve.
    title (str): A string representing the title of the grpah. Default is 'Hull-White Average Path vs Forward Curve'.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.step(forward_curve[0], forward_curve[1], where='post', label='Fine Curve', color='orange')
    plt.step(hull_white[0], hull_white[2], where='post', label='Hull-White', color='red')
    plt.xlabel('Date')
    plt.ylabel('Rate')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def plot_hull_white_paths(hull_white, forward_curve, title='Hull-White Paths vs Forward Curve'):
    """
    Plots the Hull-White simulation results and forward curve.

    Parameters:
    hull_white (tuple): A tuple of rate dates and rate values for the Hull-White simulation.
    forward_curve (tuple): A tuple of rate dates and rate values for the forward curve.
    title (str): A string representing the title of the grpah. Default is 'Hull-White Paths vs Forward Curve'.

    Returns:
        None
    """
    num_paths = len(hull_white[1])
    colors = sns.color_palette("husl", num_paths)  # Generate distinct colors
    plt.figure(figsize=(10, 6))
    
    for index, rate in enumerate(hull_white[1]):
        plt.step(hull_white[0], rate, where='post', label=f'Hull-White Path {index + 1}', color=colors[index], alpha=0.6)
    
    plt.step(forward_curve[0], forward_curve[1], where='post', label='Fine Curve', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Rate')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def plot_hull_white_zcb_prices(hull_white):
    """
    Plots the zcb prices based on short rates from a Hull-White simulation.

    Parameters:
    hull_white (tuple): A tuple of rate dates and rate values for the Hull-White simulation.

    Returns:
        None
    """
    dates, _ , rates, _ = hull_white # Extract the dates and rates from the Hull-White simulation

    hw_zcb_prices = price_zcbs(dates, rates) # Calculate the ZCB prices

    plt.figure(figsize=(10, 6))
    plt.scatter(dates, hw_zcb_prices, s=10)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('ZCB Prices')
    plt.grid()
    plt.show()

def main():
    # Define the file paths here
    calibration_file = 'data/daily-treasury-rates.csv'
    mbs_file = 'data/mbs_data.csv'
    
    # Load data
    market_close_date, calibration_data = load_treasury_rates_data(calibration_file)
    mbs_data = load_mbs_data(mbs_file)
    
    # Calculate forward curves
    coarse_curve = bootstrap_forward_curve(calibration_data, market_close_date, 100)
    fine_curve = calibrate_finer_forward_curve(calibration_data, market_close_date, 100, smoothing_error_weight=50000)
    
    # Plot the curves
    plot_forward_curves(coarse_curve, fine_curve)

    # Initialize SMMs
    smms = init_smms()
    
    # Price MBS cash flows
    results = price_mbs_cash_flows(mbs_data, smms, coarse_curve, fine_curve)

    # Define the start rate based on information from the fine forward curve
    start_rate = fine_curve[1][0]

    # Define alpha, sigma, and num_iterations
    alpha = 1
    sigma = 0.01
    num_iterations = 1000

    print(f"Alpha: {alpha}, Sigma: {sigma}, Number of Iterations: {num_iterations}")

    # Define the short rate dates to be used for the Hull-White simulation
    # In this case we will use the already defined monthly grid from the fine curve
    short_rate_dates = fine_curve[0]

    # Use Hull-White to simulate short rates based on the fine forward curve data
    hull_white = hull_white_simulate_from_curve(alpha, sigma, fine_curve, short_rate_dates, start_rate, num_iterations)

    # Create a second simulation with no antithetic sampling to compare to the original Hull-White simulation
    hw_no_antithetic = hull_white_simulate_from_curve(alpha, sigma, fine_curve, short_rate_dates, start_rate, num_iterations, False)

    # Create separate simulations with low number of iterations to plot and compare antithetic vs general sampling paths
    low_path_iterations = 6 # Define the number of paths to be simulated for the low number of iterations simulation
    hw_low_path_1 = hull_white_simulate_from_curve(alpha, sigma, fine_curve, short_rate_dates, start_rate, low_path_iterations, False)
    hw_low_path_2 = hull_white_simulate_from_curve(alpha, sigma, fine_curve, short_rate_dates, start_rate, low_path_iterations)

    # Plot to compare the Hull-White simulations to the fine forward curve
    plot_hull_white(hull_white, fine_curve)
    plot_hull_white(hw_no_antithetic, fine_curve, title='No Antithetic Hull-White Average Path vs Forward Curve')
    plot_hull_white_paths(hw_low_path_1, fine_curve, title='No Antithetic Hull-White Paths vs Forward Curve')
    plot_hull_white_paths(hw_low_path_2, fine_curve, title='Antithetic Hull-White Paths vs Forward Curve')

    # Plot the ZCB prices based on short rates from the Hull-White simulation
    plot_hull_white_zcb_prices(hull_white)

    # Print the 30-yr rate variance of antithetic vs regular sampling Hull-White simmulations
    print(f"Antithetic Sampling 30-yr Rate Variance: {hull_white[3][-1]}, No Antithetic Sampling 30-yr Rate Variance: {hw_no_antithetic[3][-1]}")

    # Extract the short rate paths from the Hull-White simulation
    short_rates = hull_white[1]

    # Simulate expected WALs, values, prices, and their standard deviations from the first set of short rates
    simulated_mbs_values = evaluate_mbs_short_rate_paths(mbs_data, short_rates, fine_curve[0])

if __name__ == '__main__':
    main()