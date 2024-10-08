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
    calculate_monthly_payment,
    calculate_scheduled_balances,
    calculate_scheduled_balances_with_service_fee,
    calculate_balances_with_prepayment,
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
    rates_file : str
    Path to the treasury rates data CSV file.

    Returns:
    date : datetime
    The effective date of the treasury rates
    maturity_rate_tuples : list 
    A list of tuples representing the treasury data.
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
    mbs_file : str
    Path to the MBS data CSV file.

    Returns:
    mbs_data : list 
    A list of lists representing MBS data with evaluated lists.
    """
    # Load MBS data from CSV files
    mbs_data = pd.read_csv(mbs_file, skiprows=1, header=None)

    # Apply the datetime parsing to the last column
    mbs_data[6] = mbs_data[6].apply(parse_datetime)

    # Function to safely evaluate the string representation of a list
    def safe_eval(value):
        if pd.isna(value) or value == '':
            return None  # Return None for NaN or empty strings
        try:
            return np.array(ast.literal_eval(value))  # Safely evaluate the string
        except (ValueError, SyntaxError):
            return None  # Return None for invalid strings
        
    # Convert the string representation of a list to an actual list
    mbs_data[7] = mbs_data[7].apply(safe_eval)

    mbs_data = mbs_data.values.tolist()

    return mbs_data

def plot_curves(coarse_curve, fine_curve):
    """
    Plots the coarse and fine forward curves.

    Parameters:
    coarse_curve : tuple 
        A tuple of rate dates and rate values for the coarse curve.
    fine_curve : tuple
        A tuple of rate dates and rate values for the fine curve.

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

def price_basic_cash_flows(mbs, forward_curve):
    """
    Prices basic MBS cash flows without prepayment or service fee considerations.

    Parameters:
    mbs : list
        A list containing details of the MBS including ID, balance, number of months, gross and net annual coupons, and origination date.
    forward_curve : tuple
        A tuple containing rate dates and corresponding forward rate values.
    
    Returns:
    tuple
        The MBS ID, present value of the cash flows, and weighted average life (WAL) of the security.
    """

    # Unpack the MBS details (excluding the first and last elements)
    mbs_id, balance, num_months, gross_annual_coupon, net_annual_coupon, origination_date = mbs[1:-2]

    # Calculate the monthly payment based on the balance, number of months, and gross coupon
    monthly_payment = calculate_monthly_payment(balance, num_months, gross_annual_coupon)

    # Calculate scheduled balances, principal paydowns, and interest paid
    months, balances, principal_paydowns, interest_paid = calculate_scheduled_balances(
        balance, num_months, gross_annual_coupon, monthly_payment)

    # Generate payment dates based on origination date and months
    payment_dates = [origination_date + relativedelta(months=months[i]) for i in range(len(months))]

    # Create a tuple of payment dates, balances, principal paydowns, and interest paid
    tuple = (payment_dates, balances, principal_paydowns, interest_paid)

    # Convert the tuple into a pandas DataFrame for easier manipulation
    df = pd.DataFrame(list(zip(*tuple)), columns=['Payment Date', 'Scheduled Balance', 'Principal Paydown', 'Interest Paid'])

    # Unpack the forward curve into rate dates and rate values
    rate_dates, rate_vals = forward_curve

    # Calculate the present value of the cash flows using the forward curve
    present_value = calculate_present_value(df, origination_date, rate_vals, rate_dates, date_name='Payment Date', net_interest_name='Interest Paid')

    # Calculate the weighted average life (WAL) of the MBS based on the payment dates and balances
    wal = calculate_weighted_average_life(df, origination_date)

    # Return the MBS ID, the present value of the cash flows, and the WAL
    return mbs_id, present_value, wal


def price_cash_flows_with_service_fee(mbs, forward_curve):
    """
    Prices MBS cash flows considering service fees.

    Parameters:
    mbs : list
        A list containing details of the MBS including ID, balance, number of months, gross and net annual coupons, and origination date.
    forward_curve : tuple
        A tuple containing rate dates and corresponding forward rate values.
    
    Returns:
    tuple
        The MBS ID, present value of the cash flows, and weighted average life (WAL) of the security.
    """

    # Unpack the MBS details (excluding the first and last elements)
    mbs_id, balance, num_months, gross_annual_coupon, net_annual_coupon, origination_date = mbs[1:-2]

    # Calculate the service fee rate as the difference between gross and net coupon
    service_fee_rate = gross_annual_coupon - net_annual_coupon

    # Calculate the monthly payment based on the balance, number of months, and gross coupon
    monthly_payment = calculate_monthly_payment(balance, num_months, gross_annual_coupon)

    # Calculate balances, principal paydowns, interest paid, and net interest paid with service fees
    months, balances, principal_paydowns, interest_paid, net_interest_paid = calculate_scheduled_balances_with_service_fee(
        balance, num_months, gross_annual_coupon, monthly_payment, service_fee_rate)

    # Generate payment dates based on origination date and months
    payment_dates = [origination_date + relativedelta(months=months[i]) for i in range(len(months))]

    # Create a tuple of payment dates, balances, principal paydowns, interest paid, and net interest paid
    tuple = (payment_dates, balances, principal_paydowns, interest_paid, net_interest_paid)

    # Convert the tuple into a pandas DataFrame for easier manipulation
    df = pd.DataFrame(list(zip(*tuple)), columns=['Payment Date', 'Scheduled Balance', 'Principal Paydown', 'Interest Paid', 'Net Interest Paid'])

    # Unpack the forward curve into rate dates and rate values
    rate_dates, rate_vals = forward_curve

    # Calculate the present value of the cash flows using the forward curve
    present_value = calculate_present_value(df, origination_date, rate_vals, rate_dates, date_name='Payment Date', net_interest_name='Interest Paid')

    # Calculate the weighted average life (WAL) of the MBS based on the payment dates and balances
    wal = calculate_weighted_average_life(df, origination_date)

    # Return the MBS ID, the present value of the cash flows, and the WAL
    return mbs_id, present_value, wal


def price_cash_flows_with_prepayment(mbs, forward_curve):
    """
    Prices MBS cash flows considering prepayments.

    Parameters:
    mbs : list
        A list containing details of the MBS including ID, balance, number of months, gross and net annual coupons, origination date, and single monthly mortality (SMM) rate.
    forward_curve : tuple
        A tuple containing rate dates and corresponding forward rate values.
    
    Returns:
    tuple
        The MBS ID, present value of the cash flows, and weighted average life (WAL) of the security.
    """

    # Unpack the MBS details (excluding the first and last elements)
    mbs_id, balance, num_months, gross_annual_coupon, net_annual_coupon, origination_date, smms = mbs[1:-1]

    # Calculate balances, principal paydowns, and interest paid with prepayment
    months, scheduled_balances, actual_balances, principal_paydowns, interest_paid, net_interest_paid = calculate_balances_with_prepayment(
        balance, num_months, gross_annual_coupon, net_annual_coupon, smms)

    # Generate payment dates based on origination date and months
    payment_dates = [origination_date + relativedelta(months=months[i]) for i in range(len(months))]

    # Create a tuple of payment dates, balances, principal paydowns, interest paid, and net interest paid
    tuple = (payment_dates, scheduled_balances, actual_balances, principal_paydowns, interest_paid, net_interest_paid)

    # Convert the tuple into a pandas DataFrame for easier manipulation
    df = pd.DataFrame(list(zip(*tuple)), columns=['Payment Date', 'Scheduled Balance', 'Actual Balance', 'Principal Paydown', 'Interest Paid', 'Net Interest Paid'])

    # Unpack the forward curve into rate dates and rate values
    rate_dates, rate_vals = forward_curve

    # Calculate the present value of the cash flows using the forward curve
    present_value = calculate_present_value(df, origination_date, rate_vals, rate_dates, date_name='Payment Date')

    # Calculate the weighted average life (WAL) of the MBS using the actual balance
    wal = calculate_weighted_average_life(df, origination_date, balance_name='Actual Balance')

    # Return the MBS ID, the present value of the cash flows, and the WAL
    return mbs_id, present_value, wal


def price_cash_flows_with_prepayment_and_dates(mbs, forward_curve):
    """
    Prices MBS cash flows considering prepayments and payment delays.

    Parameters:
    mbs : list
        A list containing details of the MBS including ID, balance, number of months, gross and net annual coupons, origination date, single monthly mortality (SMM) rate, and payment delay days.
    forward_curve : tuple
        A tuple containing rate dates and corresponding forward rate values.
    
    Returns:
    tuple
        The MBS ID, present value of the cash flows, and weighted average life (WAL) of the security.
    """

    # Unpack the MBS details (excluding the first and last elements)
    mbs_id, balance, num_months, gross_annual_coupon, net_annual_coupon, origination_date, smms, payment_delay = mbs[1:]

    # Calculate balances, payment dates, and interest paid with prepayment and adjusted payment dates
    months, dates, payment_dates, scheduled_balances, actual_balances, principal_paydowns, interest_paid, net_interest_paid = calculate_balances_with_prepayment_and_dates(
        balance, num_months, gross_annual_coupon, net_annual_coupon, smms, origination_date, payment_delay)

    # Create a tuple of payment dates, balances, principal paydowns, interest paid, and net interest paid
    tuple = (months, dates, payment_dates, scheduled_balances, actual_balances, principal_paydowns, interest_paid, net_interest_paid)

    # Convert the tuple into a pandas DataFrame for easier manipulation
    df = pd.DataFrame(list(zip(*tuple)), columns=['Month', 'Accruel Date', 'Payment Date', 'Scheduled Balance', 'Actual Balance', 'Principal Paydown', 'Interest Paid', 'Net Interest Paid'])

    # Unpack the forward curve into rate dates and rate values
    rate_dates, rate_vals = forward_curve

    # Calculate the present value of the cash flows using the forward curve
    present_value = calculate_present_value(df, origination_date, rate_vals, rate_dates)

    # Calculate the weighted average life (WAL) of the MBS using the actual balance
    wal = calculate_weighted_average_life(df, origination_date, balance_name='Actual Balance')

    # Return the MBS ID, the present value of the cash flows, and the WAL
    return mbs_id, present_value, wal


def price_mbs_cash_flows(mbs_data, coarse_curve, fine_curve):
    """
    Prices MBS all cash flows.

    Parameters:
    mbs_data : list
        A list containing details of the MBSs including ID, balance, number of months, gross and net annual coupons, origination date, single monthly mortality (SMM) rate, and payment delay days.
    coarse_curve : tuple
        A tuple containing coarse rate dates and corresponding forward rate values.
    fine_curve : tuple
        A tuple containing fine rate dates and corresponding forward rate values.
    
    Returns:
    tuple
        The MBS ID, present value of the cash flows, and weighted average life (WAL) of the security for each the coarse and fine curve
    """
    coarse_curve_results = []
    fine_curve_results = []

    for mbs in mbs_data:
        if mbs[0] == 'calculate_scheduled_balances':
            coarse_curve_result = price_basic_cash_flows(mbs, coarse_curve)
            fine_curve_result = price_basic_cash_flows(mbs, fine_curve)
        elif mbs[0] == 'calculate_scheduled_balances_with_service_fee':
            coarse_curve_result = price_cash_flows_with_service_fee(mbs, coarse_curve)
            fine_curve_result = price_cash_flows_with_service_fee(mbs, fine_curve)
        elif mbs[0] == 'calculate_balances_with_prepayment':
            coarse_curve_result = price_cash_flows_with_prepayment(mbs, coarse_curve)
            fine_curve_result = price_cash_flows_with_prepayment(mbs, fine_curve)
        elif mbs[0] == ' price_cash_flows_with_prepayment_and_dates':
            coarse_curve_result = price_cash_flows_with_prepayment_and_dates(mbs, coarse_curve)
            fine_curve_result = price_cash_flows_with_prepayment_and_dates(mbs, fine_curve)
        else:
            raise ValueError('Price function string not recognized')
        
        print(f"MBS {mbs[1]}: Coarse Price = {coarse_curve_result[1]}, Fine Price = {fine_curve_result[1]}, WAL = {coarse_curve_result[2]} years")

        coarse_curve_results.append(coarse_curve_result)
        fine_curve_results.append(fine_curve_result)

    return coarse_curve_results, fine_curve_results

def main():
    # Define the file paths here
    calibration_file = 'data/daily-treasury-rates.csv'
    mbs_file = 'data/mbs_data.csv'
    
    # Load data
    effective_rate_date, calibration_data = load_treasury_rates_data(calibration_file)
    mbs_data = load_mbs_data(mbs_file)
    
    # Calculate forward curves
    coarse_curve = bootstrap_forward_curve(calibration_data, effective_rate_date, 100)
    fine_curve = calibrate_finer_forward_curve(calibration_data, effective_rate_date, 100, smoothing_error_weight=50000)
    
    # Plot the curves
    plot_curves(coarse_curve, fine_curve)
    
    # Price MBS cash flows
    results = price_mbs_cash_flows(mbs_data, coarse_curve, fine_curve)

if __name__ == '__main__':
    main()