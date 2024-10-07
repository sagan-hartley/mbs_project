import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from utils import discount_cash_flows

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

def calculate_scheduled_balances(principal, num_months, annual_interest_rate, monthly_payment):
    """
    Calculate scheduled balances, principal paydowns, and interest paid.

    Parameters:
    principal (float): Principal amount.
    num_months (int): Number of months.
    annual_interest_rate (float): Annual interest rate (as a decimal).
    monthly_payment (float): Monthly payment.

    Returns:
    tuple: (months, balances, principal_paydowns, interest_paid)
    """
    months = np.arange(num_months + 1)
    balances = np.zeros(num_months + 1)
    principal_paydowns = np.zeros(num_months + 1)
    interest_paid = np.zeros(num_months + 1)

    balances[0] = principal

    # Iterate over each month to calculate balances, principal paydowns, and interest paid
    for month in range(1, num_months + 1):
        interest_paid[month-1] = balances[month - 1] * annual_interest_rate / 12
        principal_paydowns[month] = monthly_payment - interest_paid[month-1]
        balances[month] = balances[month - 1] - principal_paydowns[month]

    return months, balances, principal_paydowns, interest_paid

def calculate_scheduled_balances_with_service_fee(principal, num_months, annual_interest_rate, monthly_payment, service_fee_rate):
    """
    Calculate scheduled balances, principal paydowns, interest paid, and net interest paid with servicing fee
    using the result from the calculate_scheduled_balances function.

    Parameters:
    principal (float): Principal amount.
    num_months (int): Number of months.
    annual_interest_rate (float): Annual interest rate (as a decimal).
    monthly_payment (float): Monthly payment.
    service_fee_rate (float): Service fee rate (as a decimal).

    Returns:
    tuple: (months, balances, principal_paydowns, interest_paid, net_interest_paid)
    """
    # Get results from calculate_scheduled_balances
    months, balances, principal_paydowns, interest_paid = calculate_scheduled_balances(
        principal, num_months, annual_interest_rate - service_fee_rate, monthly_payment
    )

    # Calculate net interest paid (interest paid minus servicing fee)
    net_interest_paid = interest_paid - service_fee_rate
    net_interest_paid[-1] = 0

    return months, balances, principal_paydowns, interest_paid, net_interest_paid

def calculate_balances_with_prepayment(principal, num_months, gross_annual_interest_rate, net_annual_interest_rate, smms):
    """
    Calculate scheduled and actual balances with prepayment.

    Parameters:
    principal (float): Principal amount.
    num_months (int): Number of months.
    gross_annual_interest_rate (float): Gross annual interest rate (as a decimal).
    net_annual_interest_rate (float): Net annual interest rate (as a decimal).
    smms (array-like): Single Monthly Mortality rates.

    Returns:
    tuple: (months, scheduled_balances, actual_balances, principal_paydowns, interest_paid, net_interest_paid)
    """
    
    # Calculate months, scheduled balances, scheduled principal paydowns, and gross interest paid
    monthly_payment = calculate_monthly_payment(principal, num_months, gross_annual_interest_rate)
    months, scheduled_balances, scheduled_principal_paydowns, gross_interest_paid = calculate_scheduled_balances(
        principal, num_months, gross_annual_interest_rate, monthly_payment
    )

    # Initialize arrays
    pool_factors = np.ones(num_months + 1)
    actual_balances = np.zeros(num_months + 1)
    net_interest_paid = np.zeros(num_months + 1)

    # Vectorized calculation of pool factors based on SMMS
    pool_factors = np.ones(num_months + 1)
    pool_factors[1:] = np.cumprod(1 - smms)
    pool_factors[-1] = 0

    # Calculate actual balances, actual principal paydowns, and net interest paid
    actual_balances = scheduled_balances * pool_factors
    actual_principal_paydowns = np.insert(-np.diff(actual_balances), 0, 0)
    net_interest_paid = actual_balances * net_annual_interest_rate / 12
    net_interest_paid = net_interest_paid[-1] + net_interest_paid[1:]

    return months, scheduled_balances, actual_balances, actual_principal_paydowns, gross_interest_paid, net_interest_paid

def calculate_balances_with_prepayment_and_dates(principal, num_months, gross_annual_interest_rate, net_annual_interest_rate, smms, origination_date, payment_delay_days=24):
    """
    Calculate balances with prepayment and payment dates.

    Parameters:
    principal (float): Principal amount.
    num_months (int): Number of months.
    gross_annual_interest_rate (float): Gross annual interest rate (as a decimal).
    net_annual_interest_rate (float): Net annual interest rate (as a decimal).
    smms (array-like): Single Monthly Mortality rates.
    origination_date (datetime): Date of loan origination.
    payment_delay_days (int): Delay in days before the first payment (default is 24).

    Returns:
    tuple: (months, dates, payment_dates, scheduled_balances, actual_balances, principal_paydowns, interest_paid, net_interest_paid)
    """
    # Calculate the first payment date considering the payment delay
    if payment_delay_days != 0:
        first_payment_date = origination_date + timedelta(days=payment_delay_days)
    else:
        first_payment_date = origination_date

    # Generate dates for each month and payment dates
    dates = [origination_date + relativedelta(months=i) for i in range(num_months + 1)]
    payment_dates = [first_payment_date + relativedelta(months=i) for i in range(num_months + 1)]

    # Calculate balances with prepayment
    months, scheduled_balances, actual_balances, principal_paydowns, interest_paid, net_interest_paid = calculate_balances_with_prepayment(
        principal, num_months, gross_annual_interest_rate, net_annual_interest_rate, smms
    )

    return months, dates, payment_dates, scheduled_balances, actual_balances, principal_paydowns, interest_paid, net_interest_paid

def calculate_weighted_average_life(df, reference_date, payment_date_name='Payment Date', balance_name='Scheduled Balance'):
    """
    Calculate the Weighted Average Life (WAL) of a loan or security.

    Parameters:
    df (pd.DataFrame): DataFrame containing the payment schedule, which includes at least the payment dates and scheduled balances.
                       The column names for these can be specified using 'payment_date_name' and 'balance_name'.
    reference_date (datetime): The date from which the years to each payment date are calculated (usually the settlement or issue date).
    payment_date_name (str): Column name for the payment dates in the DataFrame. Defaults to 'Payment Date'.
    balance_name (str): Column name for the outstanding balances in the DataFrame. Defaults to 'Scheduled Balance'.

    Returns:
    float: The Weighted Average Life, which represents the average time until principal is repaid, weighted by the amount of principal.
    """
    # Calculate the number of years between each payment date and the reference date
    df['Years'] = (df[payment_date_name] - reference_date).dt.days / 365.25
    # Calculate the principal paydown for each period
    df['Principal Paydown'] = df[balance_name].shift(1, fill_value=0) - df[balance_name]
    df.loc[0, 'Principal Paydown'] = 0  # Set the first period's paydown to 0

    # Compute the numerator and denominator for WAL calculation
    wal_numerator = (df['Years'] * df['Principal Paydown']).sum()
    wal_denominator = df['Principal Paydown'].sum()

    # Calculate WAL
    wal = wal_numerator / wal_denominator if wal_denominator != 0 else 0
    return wal


def calculate_present_value(schedule, settle_date, rate_vals, rate_dates, principal_name='Principal Paydown', net_interest_name='Net Interest Paid', payment_date_name='Payment Date'):
    """
    Calculate the present value of cash flows by discounting them with corresponding zero-coupon bond rates.

    Parameters:
    schedule (pd.DataFrame): DataFrame containing the payment schedule, including columns for payment dates, principal paydowns, and net interest paid.
    settle_date (datetime): The settlement date used to filter cash flows occurring after this date.
    rate_vals (list of float): List of discount rates (in decimal) that correspond to the rate_dates.
                               These rates are applied piecewise between the rate_dates.
    rate_dates (list of datetime or datetime64[D]): List of dates representing the start of periods where the rates change.
                                                    Rates apply between consecutive dates.
    principal_name (str): Column name for principal payments in the DataFrame. Defaults to 'Principal Paydown'.
    net_interest_name (str): Column name for net interest payments in the DataFrame. Defaults to 'Net Interest Paid'.
    payment_date_name (str): Column name for the payment dates in the DataFrame. Defaults to 'Payment Date'.

    Returns:
    float: The present value of the cash flows, calculated by discounting them back to the settlement date using the provided rates.
    """
    # Filter out cash flows that occur before the settlement date
    filtered_schedule = schedule[schedule[payment_date_name] > settle_date]
    payment_dates = filtered_schedule[payment_date_name].to_numpy()
    cash_flows = (filtered_schedule[principal_name] + filtered_schedule[net_interest_name]).to_numpy()

    # Calculate the present value using the discount_cash_flows function
    present_value = discount_cash_flows(payment_dates, cash_flows, rate_vals, rate_dates)

    return present_value

def calculate_dirty_price(present_value, balance_at_settle):
    """
    Calculate the dirty price of a bond.

    Parameters:
    present_value (float): The present value of the bond.
    balance_at_settle (float): The balance at settlement.

    Returns:
    float: The dirty price of the bond.
    """
    dirty_price = present_value * 100 / balance_at_settle
    return dirty_price

def calculate_clean_price(dirty_price, settle_date, last_coupon_date, annual_interest_rate, balance_at_settle):
    """
    Calculate the clean price of a bond.

    Parameters:
    dirty_price (float): The dirty price of the bond.
    settle_date (datetime): Settlement date.
    last_coupon_date (datetime): The date of the last coupon payment.
    annual_interest_rate (float): Annual interest rate (as a decimal).
    balance_at_settle (float): The balance at settlement.

    Returns:
    float: The clean price of the bond.
    """
    # Calculate the days between the settle and last coupon date
    days_between = (settle_date - last_coupon_date).days
    
    # Calculate the accrued interest
    accrued_interest = (annual_interest_rate / 365.25) * days_between * balance_at_settle
    
    # Calculate the clean price
    clean_price = dirty_price - accrued_interest

    return clean_price
