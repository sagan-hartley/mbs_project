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
        interest_paid[month] = balances[month - 1] * annual_interest_rate / 12
        principal_paydowns[month] = monthly_payment - interest_paid[month]
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
    net_interest_paid = interest_paid - np.concatenate(([0], np.full(len(interest_paid) - 1, service_fee_rate)))

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
    months = np.arange(num_months + 1)
    
    # Calculate scheduled balances, principal paydowns, and interest paid
    monthly_gross_interest_rate = gross_annual_interest_rate / 12
    monthly_payment = calculate_monthly_payment(principal, num_months, gross_annual_interest_rate)
    _, scheduled_balances, principal_paydowns, interest_paid = calculate_scheduled_balances(
        principal, num_months, gross_annual_interest_rate, monthly_payment
    )

    # Initialize arrays
    pool_factors = np.ones(num_months + 1)
    actual_balances = np.zeros(num_months + 1)
    net_interest_paid = np.zeros(num_months + 1)

    # Calculate pool factors based on SMMS
    for month in range(1, num_months + 1):
        if month < num_months:
            pool_factors[month] = pool_factors[month - 1] * (1 - smms[month - 1])
        else:
            pool_factors[month] = 0

    # Calculate actual balances and net interest paid
    actual_balances = scheduled_balances * pool_factors
    servicing_fees = actual_balances * (gross_annual_interest_rate - net_annual_interest_rate) / 12
    net_interest_paid = interest_paid - servicing_fees

    return months, scheduled_balances, actual_balances, principal_paydowns, interest_paid, net_interest_paid

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

def calculate_weighted_average_life(df, reference_date):
    """
    Calculate the Weighted Average Life (WAL) of a loan or security.

    Parameters:
    df (pd.DataFrame): DataFrame containing payment schedule with 'Payment Date' and 'Scheduled Balance'.
    reference_date (datetime): Reference date for the WAL calculation.

    Returns:
    float: The Weighted Average Life.
    """
    # Calculate the number of years between each payment date and the reference date
    df['Years'] = (df['Payment Date'] - reference_date).dt.days / 365.0
    # Calculate the principal paydown for each period
    df['Principal Paydown'] = df['Scheduled Balance'].shift(1, fill_value=0) - df['Scheduled Balance']
    df.loc[0, 'Principal Paydown'] = 0  # Set the first period's paydown to 0

    # Compute the numerator and denominator for WAL calculation
    wal_numerator = (df['Years'] * df['Principal Paydown']).sum()
    wal_denominator = df['Principal Paydown'].sum()

    # Calculate WAL
    wal = wal_numerator / wal_denominator if wal_denominator != 0 else 0
    return wal

def calculate_present_value(schedule, settle_date, rate_vals, rate_dates):
    """
    Calculate the present value of cash flows based on zero-coupon bond values.

    Parameters:
    schedule (pd.DataFrame): DataFrame containing payment schedule with 'Payment Date' and 'Principal Paydown'.
    settle_date (datetime): Settlement date.
    rate_vals : list of float
        A list of discount rates (in decimal form) corresponding to the rate_dates.
        The rates are applied in a piecewise manner between the rate_dates.
    rate_dates : list of datetime or datetime64[D]
        A list of dates where the rates change. The first entry represents the market close date (i.e., the 
        starting point for the discounting process). Rates apply between consecutive dates.

    Returns:
    float: The present value of the cash flows.
    """
    # Filter out cash flows that are before the settlement date
    filtered_schedule = schedule[schedule['Payment Date'] > settle_date]
    payment_dates = filtered_schedule['Payment Date'].to_numpy()
    cash_flows = filtered_schedule['Principal Paydown'].to_numpy()

    # Calculate the present value using discount cash flows function
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
