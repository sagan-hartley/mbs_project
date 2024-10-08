import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from financial_calculations.forward_curves import (
    bootstrap_forward_curve, 
    calibrate_finer_forward_curve
)
from financial_calculations.mbs_cash_flows import (
    calculate_balances_with_prepayment_and_dates,
    calculate_dirty_price,
    calculate_clean_price,
    calculate_present_value,
    calculate_weighted_average_life
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

def plot_curves(coarse_curve, fine_curve):
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

def price_cash_flows(mbs, smms, forward_curve):
    """
    Prices MBS cash flows considering prepayments and payment delays.

    Parameters:
    mbs (list): A list containing details of the MBS including ID, balance, number of months, gross and net annual coupons, settle date, single monthly mortality (SMM) rate, and payment delay days.
    smms (ndarray): A ndarray containing prepayment rates.
    forward_curve (tuple): A tuple containing rate dates and corresponding forward rate values.
    
    Returns:
    tuple: The MBS ID, present value of the cash flows, and weighted average life (WAL) of the security.
    """

    # Unpack the MBS details (excluding the first and last elements)
    mbs_id, balance, num_months, gross_annual_coupon, net_annual_coupon, settle_date, payment_delay = mbs

    # Unpack the forward curve into rate dates and rate values
    rate_dates, rate_vals = forward_curve

    # Get the market close date from the forward curve
    market_close_date = rate_dates[0]

    # Calculate balances, payment dates, and interest paid with prepayment and adjusted payment dates
    months, dates, payment_dates, scheduled_balances, actual_balances, principal_paydowns, interest_paid, net_interest_paid = calculate_balances_with_prepayment_and_dates(
        balance, num_months, gross_annual_coupon, net_annual_coupon, smms, market_close_date, payment_delay)

    # Create a tuple of payment dates, balances, principal paydowns, interest paid, and net interest paid
    tuple = (months, dates, payment_dates, scheduled_balances, actual_balances, principal_paydowns, interest_paid, net_interest_paid)

    # Convert the tuple into a pandas DataFrame for easier manipulation
    df = pd.DataFrame(list(zip(*tuple)), columns=['Month', 'Accruel Date', 'Payment Date', 'Scheduled Balance', 'Actual Balance', 'Principal Paydown', 'Interest Paid', 'Net Interest Paid'])

    # Calculate the present value of the cash flows using the forward curve
    present_value = calculate_present_value(df, market_close_date, rate_vals, rate_dates)

    # Calculate the balance at the settle date
    settle_loc = np.searchsorted(dates, settle_date, side='right') - 1 # Get the location for the balance at settle
    balance_at_settle = df.loc[settle_loc, 'Actual Balance']

    # Calculate the dirty price of the mbs from the present value
    dirty_price = calculate_dirty_price(present_value, balance_at_settle)

    # Calculate the clean price of the mbs from the dirty price
    last_coupon_date = df.loc[settle_loc, 'Accruel Date']
    clean_price = calculate_clean_price(dirty_price, settle_date, last_coupon_date, net_annual_coupon, balance_at_settle)

    # Calculate the weighted average life (WAL) of the MBS using the actual balance
    wal = calculate_weighted_average_life(df, settle_date, balance_name='Actual Balance')

    # Return the MBS ID, the clean price of the cash flows, and the WAL
    return mbs_id, clean_price, wal


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
        coarse_curve_result = price_cash_flows(mbs, smms, coarse_curve)
        fine_curve_result = price_cash_flows(mbs, smms, fine_curve)
        
        print(f"{mbs[0]}: Coarse Price = {coarse_curve_result[1]}, Fine Price = {fine_curve_result[1]}, WAL = {coarse_curve_result[2]} years")

        # Append the results
        coarse_curve_results.append(coarse_curve_result)
        fine_curve_results.append(fine_curve_result)

    return coarse_curve_results, fine_curve_results

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
    plot_curves(coarse_curve, fine_curve)

    # Initialize SMMs
    smms = init_smms()
    
    # Price MBS cash flows
    results = price_mbs_cash_flows(mbs_data, smms, coarse_curve, fine_curve)

if __name__ == '__main__':
    main()