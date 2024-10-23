import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from utils import (
    DISC_DAYS_IN_YEAR,
    convert_to_datetime,
    discount_cash_flows,
    get_ZCB_vector,
    days360
)

def calculate_monthly_payment(principal, num_months, annual_interest_rate):
    """
    Calculate the monthly payment for a loan.

    Parameters:
    principal (float): Principal amount.
    num_months (int): Number of months.
    annual_interest_rate (float): Annual interest rate (as a decimal).

    Returns:
    float: The monthly payment.

    Raises:
    ValueError: If `num_months` is less than or equal to 0.
    ValueError: If `principal` is less than or equal to 0.

    Notes:
    - If the `annual_interest_rate` is 0, the monthly payment is simply the principal divided by the number of months.
    """
    if num_months <= 0:
        raise ValueError("Number of months must be greater than zero.")
    if principal <= 0:
        raise ValueError("Principal must be greater than zero.")
    
    monthly_interest_rate = annual_interest_rate / 12
    
    # If the interest rate is zero, use simple division to find the monthly payment
    if monthly_interest_rate == 0:
        return principal / num_months

    # Calculate the monthly payment using the formula for an amortizing loan
    payment = (principal * monthly_interest_rate * (1 + monthly_interest_rate) ** num_months) / \
              ((1 + monthly_interest_rate) ** num_months - 1)
    
    return payment

def calculate_scheduled_balances(principal, origination_date, num_months, annual_interest_rate, monthly_payment=None):
    """
    Calculate scheduled balances, principal paydowns, and interest paid using an exponential decay formula.

    Parameters:
    principal (float): Principal amount.
    origination_date (datetime): The origination date of the MBS
    num_months (int): Number of months.
    annual_interest_rate (float): Annual interest rate (as a decimal).
    monthly_payment (float): Monthly payment. Default is None

    Returns:
    tuple: (months, accrual_dates, balances, principal_paydowns, interest_paid)

    Raises:
    ValueError: If 'origination date' does not occur on the first of the months
    """
    # Make sure the origination date is converted to a datetime object if it isn't already
    origination_date = convert_to_datetime(origination_date)

    # Check if the origination date is on the first of the month, raise ValueError if not
    if origination_date.day != 1:
        raise ValueError("origination date is not the first of the month")
    
    # If the monthly payment is input as None calculate it here
    monthly_payment = calculate_monthly_payment(principal, num_months, annual_interest_rate)

    # Create an array of months
    months = np.arange(num_months + 1)

    # Create an array of accrual dates based on the origination date and number of months for the MBS
    accrual_dates = [origination_date + relativedelta(months=i) for i in range(num_months + 1)]

    # Monthly interest rate
    r = annual_interest_rate / 12

    # Calculate balances using the closed-form formula for amortization
    growth_factor = (1 + r) ** months
    balances = principal * growth_factor - (monthly_payment / r) * (growth_factor - 1)

    # Ensure that the balance doesn't go below zero
    balances = np.maximum(balances, 0)

    # Interest paid in each month (previous month's balance times interest rate)
    interest_paid = balances[:-1] * r

    # Principal paydowns: monthly payment minus interest paid
    principal_paydowns = monthly_payment - interest_paid

    # Adjust the last month's paydown to pay off the remaining balance
    principal_paydowns[-1] = balances[-2]

    # Add a zero at the first index for both interest_paid and principal_paydowns to correspond with no payments being made at the start of the first month
    interest_paid = np.insert(interest_paid, 0, 0)
    principal_paydowns = np.insert(principal_paydowns, 0, 0)

    return months, accrual_dates, balances, principal_paydowns, interest_paid

def calculate_actual_balances(schedule_balance_data, smms, net_annual_interest_rate, payment_delay_days=24):
    """
    Calculate actual balances with prepayment using schedule balance data.

    Parameters:
    schedule_balance_data (tuple): Data returned from `calculate_scheduled_balances`, which includes:
        - months (array-like)
        - accrual_dates (array-like)
        - scheduled_balances (array-like)
        - scheduled_principal_paydowns (array-like)
        - gross_interest_paid (array-like)
    smms (array-like): Single Monthly Mortality rates.
    net_annual_interest_rate (float): Net annual interest rate (as a decimal).
    payment_delay_days (int): Delay in days before the first payment (default is 24).

    Returns:
    tuple: (months, accrual_dates, payment_dates, scheduled_balances, actual_balances, scheduled_principal_paydowns, actual_principal_paydowns, gross_interest_paid, net_interest_paid)
    """
    
    # Unpack the schedule_balance_data
    months, accrual_dates, scheduled_balances, scheduled_principal_paydowns, gross_interest_paid = schedule_balance_data

    # Calculate the payment dates array from the accrual dates and the payment_delay_days parameter
    payment_dates = accrual_dates + np.ones_like(months)*timedelta(days=payment_delay_days)
    
    # Derive the number of months from the length of months
    num_months = len(months)

    # Initialize arrays for actual balances and net interest paid
    actual_balances = np.zeros(num_months)
    net_interest_paid = np.zeros(num_months)

    # Vectorized calculation of pool factors based on SMMS
    pool_factors = np.ones(num_months)
    pool_factors[1:] = np.cumprod(1 - smms)
    pool_factors[-1] = 0

    # Calculate actual balances, actual principal paydowns, and net interest paid
    actual_balances = scheduled_balances * pool_factors
    actual_principal_paydowns = np.insert(-np.diff(actual_balances), 0, 0)
    net_interest_paid = actual_balances * net_annual_interest_rate / 12
    net_interest_paid = np.concatenate(([net_interest_paid[-1]], net_interest_paid[:-1]))

    return months, accrual_dates, payment_dates, scheduled_balances, actual_balances, scheduled_principal_paydowns, actual_principal_paydowns, gross_interest_paid, net_interest_paid

def calculate_weighted_average_life(df, reference_date, date_name='Accrual Date', payment_date_name='Payment Date', balance_name='Actual Balance', principal_name='Actual Principal Paydown'):
    """
    Calculate the Weighted Average Life (WAL) of a loan or security, excluding payments before the reference date.

    Parameters:
    df (pd.DataFrame): DataFrame containing the payment schedule, which includes at least the payment dates and scheduled balances.
                       The column names for these can be specified using 'payment_date_name' and 'balance_name'.
    reference_date (datetime): The date from which the years to each payment date are calculated (usually the settlement or issue date).
    date_name (str): Column name for the accrual dates in the DataFrame. Defaults to 'Accrual Date'.
    paymentdate_name (str): Column name for the payment dates in the DataFrame. Defaults to 'Payment Date'.
    balance_name (str): Column name for the outstanding balances in the DataFrame. Defaults to 'Actual Balance'.
    principal_name (str): Column name for the principal paydowns in the DataFrame. Defaults to 'Actual Principal Paydowns'.

    Returns:
    float: The Weighted Average Life, which represents the average time until principal is repaid, weighted by the amount of principal.
    """
    # Ensure the accrual date column is in datetime format
    df[date_name] = pd.to_datetime(df[date_name])

    # Ensure the reference date is also a datetime object
    reference_date = pd.to_datetime(reference_date)

    # Calculate the number of years between each accrual date and the reference date
    df.loc[:, 'Years'] = (df[payment_date_name] - reference_date).dt.days / DISC_DAYS_IN_YEAR

    # Calculate the principal paydown for each period
    df.loc[:, principal_name] = df[balance_name].shift(1, fill_value=0) - df[balance_name]

     # Set the first period's paydown to 0
    df.loc[0, principal_name] = 0

    # Filter out any records where the accrual date is before the reference date
    filtered_df = df[df[date_name] > reference_date]

    if filtered_df.empty:
        return 0  # No payments after the reference date

    # Compute the numerator and denominator for WAL calculation
    wal_numerator = (filtered_df['Years'] * filtered_df[principal_name]).sum()
    wal_denominator = filtered_df[principal_name].sum()

    # Calculate WAL
    wal = wal_numerator / wal_denominator if wal_denominator != 0 else 0

    return wal

def calculate_present_value(schedule, settle_date, rate_vals, rate_dates, principal_name='Actual Principal Paydown', net_interest_name='Net Interest Paid', date_name = 'Accrual Date', payment_date_name='Payment Date'):
    """
    Calculate the present value of cash flows by discounting them with corresponding zero-coupon bond rates.

    Parameters:
    schedule (pd.DataFrame): DataFrame containing the payment schedule, including columns for payment dates, principal paydowns, and net interest paid.
    settle_date (datetime): The settlement date used to filter cash flows occurring after this date.
    rate_vals (list of float): List of discount rates (in decimal) that correspond to the rate_dates.
                               These rates are applied piecewise between the rate_dates.
    rate_dates (list of datetime or datetime64[D]): List of dates representing the start of periods where the rates change.
                                                    Rates apply between consecutive dates.
    principal_name (str): Column name for principal payments in the DataFrame. Defaults to 'Actual Principal Paydowns'.
    net_interest_name (str): Column name for net interest payments in the DataFrame. Defaults to 'Net Interest Paid'.
    date_name (str): Column name for the accrual dates in the DataFrame. Defaults to 'Accrual Date'.
    payment_date_name (str): Column name for the payment dates in the DataFrame. Defaults to 'Payment Date'.

    Returns:
    float: The present value of the cash flows, calculated by discounting them back to the settlement date using the provided rates.
    """
    # Filter out cash flows that occur before the settlement date
    filtered_schedule = schedule[schedule[date_name] > settle_date]

    # Ensure there's something to process after filtering
    if filtered_schedule.empty:
        return 0.0

    # Extract the payment dates and cash flows
    payment_dates = filtered_schedule[payment_date_name].to_numpy()
    cash_flows = (filtered_schedule[principal_name] + filtered_schedule[net_interest_name]).to_numpy()

    # Calculate the initial discount factor from the first rate date to the settle date
    initial_discount = get_ZCB_vector([settle_date], rate_vals, rate_dates)[0]

    # Calculate the present value using the discount_cash_flows function and dividing by the initial discount factor
    present_value = discount_cash_flows(payment_dates, cash_flows, rate_vals, rate_dates) / initial_discount

    return present_value

def calculate_dirty_price(present_value, balance_at_settle, par_balance=100):
    """
    Calculate the dirty price of a bond.

    Parameters:
    present_value (float): The present value of the bond.
    balance_at_settle (float): The balance at settlement.
    par_balance (float, optional): The par value or principal balance of the bond. Defaults to 100.

    Returns:
    float: The dirty price of the bond.
    
    Raises:
    ValueError: If the present value is non-zero and the balance at settle is zero.
    """
    # Check for the edge case where both the present value and balance at settle are zero
    # In this case return 0 as to avoid division by zero
    if present_value == 0 and balance_at_settle == 0:
        return 0
    
    if present_value != 0 and balance_at_settle == 0:
        raise ValueError("The present value should be zero if the balance at settle is zero")
    else:
        # Calculate the dirty price by normalizing the present value by the balance at settlement
        dirty_price = present_value * par_balance / balance_at_settle

    return dirty_price

def calculate_clean_price(dirty_price, settle_date, last_coupon_date, annual_interest_rate, cash_days_in_year=360, par_balance=100):
    """
    Calculate the clean price of a bond.

    Parameters:
    dirty_price (float): The dirty price of the bond.
    settle_date (datetime): Settlement date of the bond.
    last_coupon_date (datetime): The date of the last coupon payment.
    annual_interest_rate (float): Annual interest rate (as a decimal, e.g., 0.05 for 5%).
    cash_days_in_year (int): Number of days in a year used for interest calculation. Default is 360.
    par_balance (float, optional): Par value or principal balance of the bond. Defaults to 100.

    Returns:
    float: The clean price of the bond.
    """
    # Calculate the number of days between the last coupon and settlement date
    days_between = days360(last_coupon_date, settle_date)
    
    # Calculate the accrued interest
    accrued_interest = (annual_interest_rate / cash_days_in_year) * days_between * par_balance
    
    # Calculate the clean price by subtracting accrued interest from the dirty price
    clean_price = dirty_price - accrued_interest

    return clean_price

def evaluate_cash_flows(actual_balance_data, settle_date, net_annual_interest_rate, rate_vals, rate_dates):
    """
    Evaluates cash flows for an MBS, calculating the weighted average life (WAL), present value, and clean price.

    Parameters:
    actual_balance_data (list): A list of actual balance data, including months, accrual dates, payment dates, scheduled balances, actual balances, scheduled principal paydowns, actual principal paydowns, gross interest paid, and net interest paid
    settle_date (datetime): The settle date of the bond.
    net_annual_interest_rate (float): The net annual interest rate.
    rate_vals (ndarray): Forward rate values for discounting.
    rate_dates (ndarray): Corresponding rate dates for forward rates.

    Returns:
    tuple: Weighted average life (WAL), present value of cash flows, and clean price.
    """
    # Convert actual_balance_data into a pandas DataFrame for easier manipulation
    actual_balance_df = pd.DataFrame(list(zip(*actual_balance_data)), columns=[
        'Month', 
        'Accrual Date', 
        'Payment Date', 
        'Scheduled Balance', 
        'Actual Balance', 
        'Scheduled Principal Paydowns', 
        'Actual Principal Paydowns', 
        'Interest Paid', 
        'Net Interest Paid'
    ])

    # Calculate the weighted average life (WAL) using the 'Actual Balance' column
    wal = calculate_weighted_average_life(actual_balance_df, settle_date)

    # Calculate the present value of cash flows using the forward curve
    present_value = calculate_present_value(actual_balance_df, settle_date, rate_vals, rate_dates)

    # Use the DataFrame to find the balance at settle date
    dates = actual_balance_df['Accrual Date']
    settle_loc = np.searchsorted(dates, settle_date, side='right') - 1
    balance_at_settle = actual_balance_df.loc[settle_loc, 'Actual Balance']

    # Calculate the dirty price from the present value
    dirty_price = calculate_dirty_price(present_value, balance_at_settle)

    # Find the last coupon date before the settle date
    last_coupon_date = actual_balance_df.loc[settle_loc, 'Accrual Date']

    # Calculate the clean price from the dirty price and net annual interest rate
    clean_price = calculate_clean_price(dirty_price, settle_date, last_coupon_date, net_annual_interest_rate)

    return wal, present_value, clean_price
