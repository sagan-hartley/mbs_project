import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from utils import (
    convert_to_datetime,
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
    date = convert_to_datetime(date_str)

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
    mbs_data[SETTLE_DATE_IDX] = mbs_data[SETTLE_DATE_IDX].apply(convert_to_datetime)
    mbs_data[ORIGINATION_DATE_IDX] = mbs_data[ORIGINATION_DATE_IDX].apply(convert_to_datetime)

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
    Calculate expected values and prices, path standard deviations, and DV01 for MBS based on short rate paths.

    Parameters:
    - mbs_data (list): A list containing details of the MBSs including ID, balance, number of months, gross and net annual coupons, accrual dates, single monthly mortality (SMM) rate, and payment delay days.
    - short_rates (ndarray): Array of short rates from the simulation.
    - short_rate_dates (tuple): Array of dates corresponding to the short rates.

    Returns:
    - results (list): A list of dictionaries containing MBS ID, expected value, expected price, DV01, and standard deviations for each MBS.
    """
    results = []  # To store results for each MBS
    market_close_date = short_rate_dates[0]  # Extract the market close date from the short rate dates

    # Loop through each MBS in the provided data
    for mbs in mbs_data:
        wals, vals, prices, dv01s = [], [], [], []  # Lists to store results for the current MBS

        # Unpack the MBS details
        mbs_id, balance, num_months, gross_annual_coupon, net_annual_coupon, settle_date, origination_date, payment_delay = mbs

        # Calculate the necessary scheduled balance data to value and price the cash flows
        scheduled_balance_data = calculate_scheduled_balances(balance, origination_date, num_months, gross_annual_coupon)

        # Calculate the Primary Current Coupons (PCCs) and SMMs based on the original short rates
        pccs = calculate_pccs(short_rates)
        smms = calculate_smms(pccs, gross_annual_coupon, market_close_date, origination_date, num_months)

        # Precompute bumped short rates
        bump_amount = 0.0025
        short_rates_up = short_rates + bump_amount
        short_rates_down = short_rates - bump_amount
        bumped_pccs_up = calculate_pccs(short_rates_up)
        bumped_smms_up = calculate_smms(bumped_pccs_up, gross_annual_coupon, market_close_date, origination_date, num_months)
        bumped_pccs_down = calculate_pccs(short_rates_down)
        bumped_smms_down = calculate_smms(bumped_pccs_down, gross_annual_coupon, market_close_date, origination_date, num_months)

        # Loop through each SMM to calculate cash flows
        for index, smm in enumerate(smms):
            # Calculate the actual scheduled balances based on the current SMM
            actual_balance_data = calculate_actual_balances(scheduled_balance_data, smm, net_annual_coupon, payment_delay)

            # Evaluate the cash flows for the current MBS using the actual balances, SMM, and short rates data
            wal, val, price = evaluate_cash_flows(actual_balance_data, settle_date, net_annual_coupon, short_rates[index], short_rate_dates)
            
            # Store the calculated values
            wals.append(wal)
            vals.append(val)
            prices.append(price)

            # Calculate bumped actual balances and cash flows for upward and downward bumped short rates
            actual_balance_data_up = calculate_actual_balances(scheduled_balance_data, bumped_smms_up[index], net_annual_coupon, payment_delay)
            _, _, price_up = evaluate_cash_flows(actual_balance_data_up, settle_date, net_annual_coupon, short_rates_up[index], short_rate_dates)

            actual_balance_data_down = calculate_actual_balances(scheduled_balance_data, bumped_smms_down[index], net_annual_coupon, payment_delay)
            _, _, price_down = evaluate_cash_flows(actual_balance_data_down, settle_date, net_annual_coupon, short_rates_down[index], short_rate_dates)

            # Calculate DV01 as the difference between the up and down bumped prices divided by twice the bump amount times one hundred (to convert to units of percent)
            dv01 = (price_down - price_up) / (2 * 100 * bump_amount)
            dv01s.append(dv01)

        # Calculate means for the WAL, value, price, and DV01 of the MBS
        expected_wal = np.mean(wals)
        expected_value = np.mean(vals)
        expected_price = np.mean(prices)
        expected_dv01 = np.mean(dv01s)

        # Calculate standard deviations
        wal_std = np.std(wals)
        val_std = np.std(vals)
        price_std = np.std(prices)
        dv01_std = np.std(dv01s)

        # Print the results for the current MBS
        print(f"{mbs_id}, Expected WAL: {expected_wal}, WAL Path STD: {wal_std}, Expected Value: {expected_value}, "
              f"Value Path STD: {val_std}, Expected Price: {expected_price}, Price Path STD: {price_std}, "
              f"Expected DV01: {expected_dv01}, DV01 Path STD: {dv01_std}")

        # Store the result in a dictionary for structured output
        results.append({
            'mbs_id': mbs_id,
            'expected_wal': expected_wal,
            'wal_std': wal_std,
            'expected_value': expected_value,
            'value_std': val_std,
            'expected_price': expected_price,
            'price_std': price_std,
            'expected_dv01': expected_dv01,
            'dv01_std': dv01_std
        })

    return results  # Return the list of results for all MBS

def pathwise_zcb_eval(maturity_date, short_rate_paths, discount_rate_dates):
    """
    Price a Zero-Coupon Bond (ZCB) using short rate paths by computing the discount factors
    and averaging across paths.

    Parameters:
    -----------
    maturity_date : np.datetime64
        The maturity date of the ZCB.
    short_rate_paths : np.ndarray
        Array of short rate paths with shape (num_paths, num_steps).
    discount_rate_dates : np.ndarray
        Array of datetime or datetime64[D] objects representing the dates on which the discount rates apply.

    Returns:
    --------
    (zcb_price, zcb_std) : tuple
         The average present value (price) of the ZCB across all paths and the standard deviation 
    """
    num_paths = short_rate_paths.shape[0]
    present_values = np.zeros(num_paths)

    # Set up a single cash flow of 1 at the maturity date (for a ZCB)
    payment_dates = np.array([maturity_date])
    cash_flows = np.array([1.0])  # ZCB has a single cash flow of 1 at maturity

    # Loop over each path to calculate present value using the discount_cash_flows function
    for i in range(num_paths):
        # Use the short rates of the current path as the discount rates
        discount_rate_vals = short_rate_paths[i, :]
        
        # Discount the cash flows for this path
        present_values[i] = discount_cash_flows(payment_dates, cash_flows, discount_rate_vals, discount_rate_dates)
    
    # Average the present values across all paths
    zcb_price = np.mean(present_values)

    # Calculate the standard deviation the present values across all paths
    zcb_std = np.std(present_values)
    
    return zcb_price, zcb_std

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
    dates, rates , _, _ = hull_white # Extract the dates and rates from the Hull-White simulation

    hw_zcb_prices = [pathwise_zcb_eval(date, rates, dates)[0] for date in dates] # Calculate the ZCB prices

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
    small_alpha = 0.5 # Define an small alpha to limit the mean reversion effect. This will allow the difference in sampling paths to be more pronounced.
    hw_low_paths_1 = hull_white_simulate_from_curve(small_alpha, sigma, fine_curve, short_rate_dates, start_rate, low_path_iterations, False)
    hw_low_paths_2 = hull_white_simulate_from_curve(small_alpha, sigma, fine_curve, short_rate_dates, start_rate, low_path_iterations)

    # Plot to compare the Hull-White simulations to the fine forward curve
    plot_hull_white(hull_white, fine_curve)
    plot_hull_white(hw_no_antithetic, fine_curve, title='No Antithetic Hull-White Average Path vs Forward Curve')
    plot_hull_white_paths(hw_low_paths_1, fine_curve, title='No Antithetic Hull-White Paths vs Forward Curve')
    plot_hull_white_paths(hw_low_paths_2, fine_curve, title='Antithetic Hull-White Paths vs Forward Curve')

    # Plot the ZCB prices based on short rates from the Hull-White simulation
    plot_hull_white_zcb_prices(hull_white)

    # Print the 30-yr rate variance of antithetic vs regular sampling Hull-White simmulations
    print(f"Antithetic Sampling 30-yr Rate Variance: {hw_low_paths_2[3][-1]}, No Antithetic Sampling 30-yr Rate Variance: {hw_low_paths_1[3][-1]}")

    # Extract the short rate paths from the Hull-White simulation
    short_rates = hull_white[1]

    # Simulate expected WALs, values, prices, and their standard deviations from the first set of short rates
    simulated_mbs_values = evaluate_mbs_short_rate_paths(mbs_data, short_rates, fine_curve[0])

if __name__ == '__main__':
    main()