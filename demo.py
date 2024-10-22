import pstats
import cProfile
from datetime import datetime
from dateutil.relativedelta import relativedelta
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
    calculate_monthly_payment,
    calculate_scheduled_balances,
    calculate_balances_with_prepayment_and_dates,
    calculate_dirty_price,
    calculate_clean_price,
    calculate_present_value,
    calculate_weighted_average_life
)
from financial_models.hull_white import (
    calculate_theta,
    hull_white_simulate
)
from financial_models.prepayment import (
    calculate_pccs,
    calculate_smms
)

def parse_datetime(date_str):
    # This function parses the datetime string into a Python datetime object
    return datetime.strptime(date_str, "%m/%d/%Y")

def load_treasury_rates_data(rates_file):
    """
    Loads treasury data from a CSV file and safely evaluate list-like strings.

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

    # Apply the datetime parsing to the Accruel Date column
    mbs_data[5] = mbs_data[5].apply(parse_datetime)

    # Convert values to a list
    mbs_data = mbs_data.values.tolist()

    # For each settle date compute accrual dates grid
    for mbs in mbs_data:
        mbs[5] = [mbs[5] + relativedelta(months=i) for i in range(mbs[2] + 1)]

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
    if zeros == True:
        smms = np.zeros(length)
    else:
        # Initialize values before the peak
        t_less_peak = np.arange(0, peak)
        smm_less_peak = (t_less_peak / peak) * 0.01

        # Initialize values after the peak
        t_greater_equal_peak = np.arange(peak, length)
        smm_greater_equal_peak = 0.015 - (t_greater_equal_peak / (length - peak)) * 0.01

        # Concatenate the two arrays
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

def evaluate_cash_flows(scheduled_balance_data, mbs, smms, forward_curve):
    """
    Values and prices MBS cash flows considering prepayments and payment delays.

    Parameters:
    scheduled_balance_data: Scheduled balance data.
    mbs (list): A list containing details of the MBS including ID, balance, number of months, gross and net annual coupons, settle date, single monthly mortality (SMM) rate, and payment delay days.
    smms (ndarray): A ndarray containing prepayment rates.
    forward_curve (tuple): A tuple containing rate dates and corresponding forward rate values.

    Returns:
    tuple: The MBS ID, clean price of the cash flows, present value of the cash flows, weighted average life (WAL), and the cash flow DataFrame.
    """

    # Unpack the MBS details
    mbs_id, balance, num_months, gross_annual_coupon, net_annual_coupon, accrual_dates, payment_delay = mbs

    # Unpack the forward curve into rate dates and rate values
    rate_dates, rate_vals = forward_curve

    # Get the market close date from the forward curve
    market_close_date = rate_dates[0]

    # Get the settle date from the accrual dates
    settle_date = accrual_dates[0]

    # Calculate balances, payment dates, and interest paid with prepayment and adjusted payment dates
    months, dates, payment_dates, scheduled_balances, actual_balances, scheduled_principal_paydowns, actual_principal_paydowns, interest_paid, net_interest_paid = calculate_balances_with_prepayment_and_dates(
        scheduled_balance_data, smms, net_annual_coupon, accrual_dates, payment_delay)

    # Create a tuple of payment dates, balances, principal paydowns, interest paid, and net interest paid
    data_tuple = (months, dates, payment_dates, scheduled_balances, actual_balances, scheduled_principal_paydowns, actual_principal_paydowns, interest_paid, net_interest_paid)

    # Convert the tuple into a pandas DataFrame for easier manipulation
    df = pd.DataFrame(list(zip(*data_tuple)), columns=['Month', 'Accrual Date', 'Payment Date', 'Scheduled Balance', 'Actual Balance', 'Scheduled Principal Paydowns', 'Actual Principal Paydowns', 'Interest Paid', 'Net Interest Paid'])

    # Calculate the present value of the cash flows using the forward curve
    present_value = calculate_present_value(df, market_close_date, rate_vals, rate_dates)

    # Calculate the weighted average life (WAL) of the MBS using the actual balance
    wal = calculate_weighted_average_life(df, settle_date, balance_name='Actual Balance')

    # Use the DataFrame to find the balance at settle
    dates = df['Accrual Date']
    settle_loc = np.searchsorted(dates, settle_date, side='right') - 1
    balance_at_settle = df.loc[settle_loc, 'Actual Balance']

    # Calculate the dirty price from the present value
    dirty_price = calculate_dirty_price(present_value, balance_at_settle)

    # Calculate the clean price from the dirty price
    last_coupon_date = df.loc[settle_loc, 'Accrual Date']
    clean_price = calculate_clean_price(dirty_price, settle_date, last_coupon_date, net_annual_coupon)

    # Return the MBS ID, clean price, present value, WAL, and the DataFrame
    return mbs_id, clean_price, present_value, wal, df

def price_mbs_cash_flows(mbs_data, smms, coarse_curve, fine_curve):
    """
    Prices MBS all cash flows.

    Parameters:
    mbs_data (list): A list containing details of the MBSs including ID, balance, number of months, gross and net annual coupons, settle date, single monthly mortality (SMM) rate, and payment delay days.
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

    # Loop through each mbs and print the ID, coarse price, fine price, and WAL
    for mbs in mbs_data:
        # Unpack the MBS details
        mbs_id, balance, num_months, gross_annual_coupon, net_annual_coupon, accrual_dates, payment_delay = mbs

        # Calculate the necessary scheduled balance data to price the cash flows
        scheduled_balance_data = calculate_scheduled_balances(balance, num_months, gross_annual_coupon, calculate_monthly_payment(balance, num_months, gross_annual_coupon))

        coarse_curve_result = evaluate_cash_flows(scheduled_balance_data, mbs, smms, coarse_curve)
        fine_curve_result = evaluate_cash_flows(scheduled_balance_data, mbs, smms, fine_curve)
        
        print(f"{mbs[0]}: Coarse Price = {coarse_curve_result[1]}, Fine Price = {fine_curve_result[1]}, WAL = {coarse_curve_result[3]} years")

        # Append the results
        coarse_curve_results.append(coarse_curve_result)
        fine_curve_results.append(fine_curve_result)

    return coarse_curve_results, fine_curve_results

def evaluate_mbs_short_rate_paths(mbs_data, short_rates, forward_curve):
    """
    Calculate expected values and prices and path variances for MBS based on short rate paths.

    Parameters:
    - mbs_data (list): List of MBS data, where each entry contains relevant information (e.g., ID, coupon, num_months).
    - short_rates (ndarray): Array of short rates for the simulation.
    - forward_curve (tuple): A tuple containing the forward curve data.

    Returns:
    - results (list): A list of dictionaries containing MBS ID, expected value, expected price, and variances for each MBS.
    """
    results = []  # To store results for each MBS
    market_close_date = forward_curve[0][0] # Extract the market clse date from the forward curve

    # Loop through each MBS in the provided data
    for mbs in mbs_data:
        vals = []  # Initialize a list to store calculated values for the current MBS
        prices = []  # Initialize a list to store calculated prices for the current MBS

        # Unpack the MBS details
        mbs_id, balance, num_months, gross_annual_coupon, net_annual_coupon, accrual_dates, payment_delay = mbs

        # Get the settle date from the accrual dates
        settle_date = accrual_dates[0]

        # Calculate the necessary scheduled balance data to value and price the cash flows
        scheduled_balance_data = calculate_scheduled_balances(balance, num_months, gross_annual_coupon, calculate_monthly_payment(balance, num_months, gross_annual_coupon))

        # Calculate the Principal Component Analysis (PCA) values based on short rates
        pccs = calculate_pccs(short_rates)

        # Calculate Single Monthly Mortality (SMM) rates
        smms = calculate_smms(pccs, gross_annual_coupon, market_close_date, settle_date, num_months)

        # Loop through each SMM to calculate cash flows
        for smm in smms:
            # Evaluate the cash flows for the current MBS using the SMM and forward curve
            eval_results = evaluate_cash_flows(scheduled_balance_data, mbs, smm, forward_curve)
            price = eval_results[1] # Extract the clean price from eval_results
            val = eval_results[2]  # Extract the present value from eval_results
            
            # Append the calculated value and price to their respective lists
            vals.append(val)
            prices.append(price)

        # Calculate the variance of the collected cash flow values and prices
        val_var = np.array(vals).var()
        price_var = np.array(prices).var()

        # Calculate the mean of the values and prices for the expected cash flow
        expected_value = np.mean(vals)
        expected_price = np.mean(prices)

        # Print the results for the current MBS
        print(f"{mbs[0]}, Expected Value: {expected_value}, Value Path Variance: {val_var}, Expected Price: {expected_price}, Price Path Variance: {price_var}")

        # Store the result in a dictionary for structured output
        results.append({
            'mbs_id': mbs[0],  # MBS identifier
            'expected_value': expected_value,  # Expected cash flow value
            'value_variance': val_var,  # Variance of the cash flow values
            'expected_price': expected_price,  # Expected price of the MBS
            'price_variance': price_var  # Variance of the cash flow prices
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

def plot_hull_white(hull_white, forward_curve):
    """
    Plots the Hull-White simulation results and forward curve.

    Parameters:
    hull_white (tuple): A tuple of rate dates and rate values for the Hull-White simulation.
    forward_curve (tuple): A tuple of rate dates and rate values for the forward curve.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.step(forward_curve[0], forward_curve[1], where='post', label='Fine Curve', color='orange')
    plt.step(hull_white[0], hull_white[2], where='post', label='Hull White', color='red')
    plt.xlabel('Date')
    plt.ylabel('Rate')
    plt.title('Hull White Average Path vs Forward Curve')
    plt.legend()
    plt.grid()
    plt.show()

def plot_hull_white_paths(hull_white, forward_curve):
    """
    Plots the Hull-White simulation results and forward curve.

    Parameters:
    hull_white (tuple): A tuple of rate dates and rate values for the Hull-White simulation.
    forward_curve (tuple): A tuple of rate dates and rate values for the forward curve.

    Returns:
        None
    """
    num_paths = len(hull_white[1])
    colors = sns.color_palette("husl", num_paths)  # Generate distinct colors
    plt.figure(figsize=(10, 6))
    
    for index, rate in enumerate(hull_white[1]):
        plt.step(hull_white[0], rate, where='post', label=f'Hull White Path {index + 1}', color=colors[index], alpha=0.6)
    
    plt.step(forward_curve[0], forward_curve[1], where='post', label='Fine Curve', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Rate')
    plt.title('Hull White Paths vs Forward Curve')
    plt.legend()
    plt.grid()
    plt.show()

def plot_hull_white_zcb_prices(hull_white):
    """
    Plots the zcb prices based on short rates from a Hull-White simulation.

    Parameters:
    hull_white (tuple): A tuple of rate dates and rate values for the hull white simulation.

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

    # Define alpha and sigma
    alpha, sigma = 1, 0.01

    print(f"Alpha: {alpha}, Sigma: {sigma}")

    # Calculate the theta function based on the fine forward curve rates
    theta = calculate_theta(fine_curve, alpha, sigma, fine_curve[0])

    # Use Hull-White to simulate short rates based on the forward curve data
    hull_white = hull_white_simulate(alpha, sigma, theta, start_rate, 10000)
    hull_white_2 = hull_white_simulate(alpha, sigma, theta, start_rate, 6, False) # Create separate simulations with low number of iterations to plot individual paths
    hull_white_3 = hull_white_simulate(alpha, sigma, theta, start_rate, 6)

    # Plot to compare the Hull-White simulation to the fine forward curve
    plot_hull_white(hull_white, fine_curve)
    plot_hull_white_paths(hull_white_2, fine_curve)
    plot_hull_white_paths(hull_white_3, fine_curve)

    # Plot the ZCB prices based on short rates from the Hull-White simulation
    plot_hull_white_zcb_prices(hull_white)

    # Print the average variance of antithetic vs regular sampling Hull-White simmulations
    print(f"Antithetic Sampling 30-yr Rate Variance: {hull_white_3[3][-1]}, No Antithetic Sampling 30-yr Rate Variance: {hull_white_2[3][-1]}")

    # Extract the short rate paths from the Hull-White simulation
    short_rates = hull_white[1]

    simulated_mbs_values = evaluate_mbs_short_rate_paths(mbs_data, short_rates, fine_curve)

if __name__ == '__main__':
    cProfile.run('main()', 'profile_output.prof')
    profile = pstats.Stats('profile_output.prof')
    profile.strip_dirs().sort_stats('time').print_stats(10)
    #main()