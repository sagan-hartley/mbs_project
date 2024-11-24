import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from utils import (
    create_regular_dates_grid
)
from financial_models.hull_white import (
    hull_white_simulate_from_curve
)
from financial_models.prepayment import (
    calculate_pccs,
    calculate_smms
)
from financial_calculations.bonds import (
    SemiBondContract,
    create_semi_bond_cash_flows,
    calculate_coupon_rate,
    pathwise_zcb_eval
)
from financial_calculations.forward_curves import (
    bootstrap_forward_curve,
    calibrate_fine_curve
)
from financial_calculations.mbs import (
    calculate_monthly_payment,
    calculate_scheduled_balances,
    calculate_actual_balances,
    MbsContract,
    pathwise_evaluate_mbs
)
from financial_calculations.cash_flows import (
    CashFlowData,
    StepDiscounter,
    filter_cash_flows,
    value_cash_flows,
    price_cash_flows,
    get_balance_at_settle,
    get_last_accrual_date,
    calculate_weighted_average_life,
    oas_search,
    calculate_dv01,
    calculate_convexity
)
MBS_ID_COL = 'MBS ID'
BAL_COL = 'Balance'
NUM_MONTHS_COL = 'Number of Months'
GROSS_CPN_COL = 'Gross Annual Coupon'
NET_CPN_COL = 'Net Annual Coupon'
SETTLE_DATE_COL = 'Settle Date'
ORIG_DATE_COL = 'Origination Date'
PYMNT_DEL_COL = 'Payment Delay'

def load_treasury_rates_data(rates_file):
    """
    Loads treasury data from a CSV file and evaluate maturity year strings.
    This function is designed to manipulate data taken directly from the treasury
    website into a format compatible with the forward curve generation functions.

    Parameters:
    rates_file (str): Path to the treasury rates data CSV file.

    Returns:
    date (datetime): The effective date of the treasury rates
    maturity_rate_tuples (list): A list of tuples representing the treasury data.
    """
    # Read the original CSV into a DataFrame
    df = pd.read_csv(rates_file)

    # Check that the data frame only has one row corresponding to the market close date and coupon rates
    if df.shape[0] != 1:
        raise ValueError("The daily treasury rates csv cannot contain more than one row of data.")

    # Extract the date
    date_str = df['Date'][0]
    date = pd.to_datetime(date_str)

    # Remove the 'Date' column to focus on maturity years and rates
    maturity_year_columns = df.columns[1:]  # All columns except 'Date'
    
    # Extract the rates from the first row (since it's only one row of data)
    rates = df.iloc[0, 1:].tolist()  # All values in the first row except the date
    
    # Create a list of tuples with maturity year (as int) and corresponding rate (as decimal)
    maturity_rate_tuples = [(int(col.split()[0]), rate/100) for col, rate in zip(maturity_year_columns, rates)]
    
    # Return the date and the list of tuples
    return date, maturity_rate_tuples

def load_mbs_data(mbs_file):
    """
    Loads MBS data from a CSV file and returns a list of MBS dataclass instances with pandas Timestamps.

    Parameters:
    mbs_file (str): Path to the MBS data CSV file.

    Returns:
    mbs_list (list): A list of MBS instances with the required attributes.
    """
    # Load MBS data from CSV
    mbs_data = pd.read_csv(mbs_file)

    # Convert the Settle Date and Origination Date columns to Pandas Timestamps
    mbs_data[SETTLE_DATE_COL] = pd.to_datetime(mbs_data[SETTLE_DATE_COL])
    mbs_data[ORIG_DATE_COL] = pd.to_datetime(mbs_data[ORIG_DATE_COL])

    # Convert each row to an MbsContract instance and append to the list
    mbs_contracts = [
        MbsContract(
            mbs_id=row[MBS_ID_COL],
            balance=row[BAL_COL],
            origination_date=row[ORIG_DATE_COL],
            num_months=row[NUM_MONTHS_COL],
            gross_annual_coupon=row[GROSS_CPN_COL],
            net_annual_coupon=row[NET_CPN_COL],
            payment_delay=row[PYMNT_DEL_COL],
            settle_date=row[SETTLE_DATE_COL]
        )
        for _, row in mbs_data.iterrows()
    ]

    return mbs_contracts

def plot_forward_curves(coarse_curve, fine_curve):
    """
    Plots the coarse and fine forward curves.

    Parameters:
    coarse_curve (StepDiscounter): An instance of StepDiscounter with dates and rates for a coarse curve.
    fine_curve (StepDiscounter): An instance of StepDiscounter with dates and rates for a fine curve.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.step(coarse_curve.dates, coarse_curve.rates, where='post', label='Coarse Curve', color='blue')
    plt.step(fine_curve.dates, fine_curve.rates, where='post', label='Fine Curve', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Rate')
    plt.title('Forward Curves')
    plt.legend()
    plt.grid()
    plt.show()

def plot_forward_curve_zcb_prices(forward_curves, curve_names, title='Forward Curve ZCB Prices'):
    """
    Plots the zcb values based on a list of forward curves.

    Parameters:
    forward_curves (list): A list of StepDiscounters representing forward curves.
    curve_names (list): A list of strings representing the names associated with the forward_curves list
    title (str): A string representing the title of the grpah. Default is 'Hull-White Paths vs Forward Curve ZCB Values'.

    Returns:
        None
    """
    # Generate distinct colors for each currve
    num_paths = len(forward_curves)
    colors = sns.color_palette("husl", num_paths)

    plt.figure(figsize=(10, 6))

    for index, forward_curve in enumerate(forward_curves):
        # Calculate the ZCB prices for the current curve
        zcb_prices = pathwise_zcb_eval(forward_curve.dates, forward_curve.rates, forward_curve.dates)

        plt.step(forward_curve.dates, zcb_prices, where='post', label=curve_names[index], color=colors[index], alpha=0.6)

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def plot_hull_white(hull_white, forward_curve, title='Hull-White Average Path vs Forward Curve'):
    """
    Plots the Hull-White simulation results and forward curve.

    Parameters:
    hull_white (tuple): A tuple of rate dates and rate values for the Hull-White simulation.
    forward_curve (StepDiscounter): An instance of StepDiscounter representing a forward curve.
    title (str): A string representing the title of the grpah. Default is 'Hull-White Average Path vs Forward Curve'.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.step(forward_curve.dates, forward_curve.rates, where='post', label='Fine Curve', color='orange')
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
    forward_curve (StepDiscounter): An instance of StepDiscounter representing a forward curve.
    title (str): A string representing the title of the grpah. Default is 'Hull-White Paths vs Forward Curve'.

    Returns:
        None
    """
    num_paths = len(hull_white[1])
    colors = sns.color_palette("husl", num_paths)  # Generate distinct colors
    plt.figure(figsize=(10, 6))
    
    for index, rate in enumerate(hull_white[1]):
        plt.step(hull_white[0], rate, where='post', label=f'Hull-White Path {index + 1}', color=colors[index], alpha=0.6)
    
    plt.step(forward_curve.dates, forward_curve.rates, where='post', label='Forward Curve', color='orange', linewidth=3.0)
    plt.xlabel('Date')
    plt.ylabel('Rate')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def plot_hull_white_path_zcb_prices(hull_white, forward_curve, title='Hull-White Paths vs Forward Curve ZCB Values'):
    """
    Plots the zcb values based on short rate paths from a Hull-White simulation.
    Also plots the zcb values based on forward curve rates for comparison.

    Parameters:
    hull_white (tuple): A tuple of rate dates and rate values for the Hull-White simulation.
    forward_curve (StepDiscounter): An instance of StepDiscounter representing a forward curve.
    title (str): A string representing the title of the grpah. Default is 'Hull-White Paths vs Forward Curve ZCB Values'.

    Returns:
        None
    """
    dates, rates , _, _ = hull_white # Extract the dates and rates from the Hull-White simulation

    # Calculate the ZCB prices based on the Hull-White simulation results
    hw_zcb_prices = pathwise_zcb_eval(dates, rates, dates)

    # Calculate the ZCB prices based on the forward curve data
    curve_zcb_prices = pathwise_zcb_eval(dates, forward_curve.rates, forward_curve.dates)

    # Generate distinct colors for each path
    num_paths = len(rates)
    colors = sns.color_palette("husl", num_paths)

    plt.figure(figsize=(10, 6))
    for index, zcb_prices in enumerate(hw_zcb_prices):
        plt.step(dates, zcb_prices, where='post', label=f'Hull-White Path {index + 1}', color=colors[index], alpha=0.6)

    plt.step(forward_curve.dates, curve_zcb_prices, where='post', label='Forward Curve', color='orange', linewidth=3.0)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def plot_hull_white_avg_zcb_prices(hull_whites, forward_curve, title='Hull-White Average Paths vs Forward Curve ZCB Values'):
    """
    Plots the zcb values based on average short rate paths from multiple Hull-White simulations.
    Also plots the zcb values based on forward curve rates for comparison.

    Parameters:
    hull_whites (list or tuple): A list of results from multiple Hull-White simulations, or just one set of results.
    forward_curve (StepDiscounter): An instance of StepDiscounter representing a forward curve.
    title (str): A string representing the title of the grpah. Default is 'Hull-White Paths vs Forward Curve ZCB Values'.

    Returns:
        None
    """
    # Calculate the ZCB prices based on the forward curve data
    curve_zcb_prices = pathwise_zcb_eval(forward_curve.dates, forward_curve.rates, forward_curve.dates)

    # Generate distinct colors for each simulation
    num_paths = len(hull_whites)
    colors = sns.color_palette("husl", num_paths)

    plt.figure(figsize=(10, 6))
    plt.step(forward_curve.dates, curve_zcb_prices, where='post', label='Forward Curve', color='orange')

    for index, hull_white in enumerate(hull_whites):
        dates, _ , avg_rates, _ = hull_white # Extract the dates and average rate from the Hull-White simulation

        # Calculate the ZCB prices based on the Hull-White simulation results
        hw_zcb_prices = pathwise_zcb_eval(dates, avg_rates, dates)

        plt.step(dates, hw_zcb_prices, where='post', label=f'Hull-White Sim {index + 1}', color=colors[index], alpha=0.6)

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def run_exercises(coarse_curve, fine_curve):
    """
    Calculates the results for all the exercises outlined in the following collab file:
    https://colab.research.google.com/drive/1kBUtBgGQ7uytfb6BrAUgF-zJbG_5mC1F?usp=sharing

    Parameters:
    coarse_curve (StepDiscounter): An instance of StepDiscounter representing the bootstrapped
        forward curve from the exercises.
    fine_curve (StepDiscounter): An instance of StepDiscounter representing the improved
        forward curve from the exercises.

    Returns:
    None

    Notes:
    The curves are not calculated inside this function as it is computationally expensive and
    they might need to be used elsewhere.
    """
    print("Exercise MBS Info...")

    mp_1 = calculate_monthly_payment(100, 360, 0.05)
    mp_2 = calculate_monthly_payment(200, 360, 0.05)
    mp_3 = calculate_monthly_payment(200, 180, 0.05)
    mp_4 = calculate_monthly_payment(200, 180, 0.06)

    print(f"\nExercise Monthly Payments: {mp_1, mp_2, mp_3, mp_4}")

    sched_balances = calculate_scheduled_balances(250, "2024-10-01", 12, 0.05, payment_delay=0)

    print(f"\nExercise Schedule Balance Grid:\n{sched_balances, 12}")

    #Do p3 plots

    actual_balances = calculate_actual_balances(sched_balances, np.zeros(len(sched_balances.accrual_dates)-1), 0.0475)

    print(f"\nExercise Actual Balance Grid:\n{actual_balances, 12}")

    # Define the simple SMMs (Single Month Mortality rates) from the exercises
    # We want the length of the SMMs to be one less than the term as no prepayment is allowed from M-1 to M
    smms = np.zeros(180)
    for t in range(180):
        smms[t] = t/60 * .01 if t < 60 else .015 - t/120 * .01

    # Let's calculate some new actual balances for the simple exercise prepayment model
    sched_balances = calculate_scheduled_balances(100, "2024-10-01", 180, 0.07)
    actual_balances = calculate_actual_balances(sched_balances, smms, 0.0675)

    print(f"\nNew Exercise Actual Balance Grid:\n{actual_balances}")

    exercise_wal = calculate_weighted_average_life(actual_balances, "2024-10-01")

    print(f"\nExercise WAL: {exercise_wal}")

    # Define the discount rates to be used for the rest of the problem
    # These rates actually correspond to the first 180 entries from the fine curve calibration done later
    rate_grid = np.array([
                0.037152689, 0.037128935, 0.037023894, 0.036950150, 0.036817723, 0.036694537,
                0.036541153, 0.036379749, 0.036206621, 0.035993993, 0.035821335, 0.035561345,
                0.035327796, 0.035046872, 0.034800912, 0.034519662, 0.034301415, 0.034039094,
                0.033837231, 0.033616164, 0.033441544, 0.033261279, 0.033157687, 0.033033966,
                0.032966727, 0.032867582, 0.032810329, 0.032709723, 0.032712051, 0.032678288,
                0.032727890, 0.032802810, 0.032882302, 0.033002311, 0.033121135, 0.033248283,
                0.033349087, 0.033481500, 0.033548198, 0.033644680, 0.033781438, 0.033828332,
                0.033988769, 0.034028321, 0.034113045, 0.034196439, 0.034279111, 0.034418190,
                0.034547958, 0.034691128, 0.034806511, 0.034901733, 0.035025973, 0.035121987,
                0.035277551, 0.035448268, 0.035594763, 0.035795894, 0.035951161, 0.036123720,
                0.036305551, 0.036484735, 0.036674024, 0.036889970, 0.037103384, 0.037297479,
                0.037495734, 0.037618304, 0.037758110, 0.037871465, 0.037921970, 0.038184057,
                0.038356549, 0.038503437, 0.038620151, 0.038680809, 0.038777976, 0.038810834,
                0.038922275, 0.038990273, 0.039054130, 0.039116377, 0.039133121, 0.039170768,
                0.039198293, 0.039257014, 0.039328614, 0.039418949, 0.039505111, 0.039616051,
                0.039672769, 0.039791109, 0.039855200, 0.039957880, 0.040105254, 0.040204305,
                0.040368062, 0.040507569, 0.040613730, 0.040767241, 0.040916601, 0.041048484,
                0.041258544, 0.041402153, 0.041559566, 0.041747338, 0.041897894, 0.042101405,
                0.042346425, 0.042540885, 0.042794073, 0.042999333, 0.043173543, 0.043377961,
                0.043518503, 0.043687666, 0.043832287, 0.043967978, 0.044100426, 0.044234340,
                0.044355315, 0.044483477, 0.044612551, 0.044731461, 0.044877540, 0.045009377,
                0.045139615, 0.045267296, 0.045386141, 0.045491997, 0.045642418, 0.045756685,
                0.045902366, 0.046034770, 0.046123281, 0.046218149, 0.046302105, 0.046370548,
                0.046476574, 0.046569591, 0.046645881, 0.046733122, 0.046782861, 0.046820931,
                0.046881562, 0.046912064, 0.046960170, 0.047014943, 0.047021509, 0.047065301,
                0.047046585, 0.047051823, 0.047028825, 0.047009286, 0.046986697, 0.046960333,
                0.046939068, 0.046912937, 0.046891320, 0.046868599, 0.046843076, 0.046822097,
                0.046794752, 0.046772979, 0.046748643, 0.046727087, 0.046706961, 0.046683387,
                0.046663736, 0.046636769, 0.046612991, 0.046588339, 0.046561760, 0.046542331,
                0.046518816, 0.046500795, 0.046480874, 0.046460978, 0.046441521, 0.046417292,
                0.046417292
            ])
    
    # Now define the discount date grid associated with the discount rates
    date_grid = create_regular_dates_grid("10/1/2024", "10/1/2039", 'm')

    # Initialize a StepDiscounter and ZCB dates grid for ZCB calculations
    discounter = StepDiscounter(date_grid, rate_grid)
    zcb_dates = pd.to_datetime(["2024-10-01", "2024-11-01", "2027-02-15", "2039-10-15"])

    print(f"\nExercise ZCBs: {discounter.zcbs_from_dates(zcb_dates)}")

    # Define the dummy exercise cash flows used for present value calculation
    exercise_flows = CashFlowData(
        balances=np.array([21.0, 16.0, 9.0, 0.0]),
        accrual_dates=create_regular_dates_grid("10/1/2024", "1/1/2025", 'm'),
        payment_dates=create_regular_dates_grid("10/25/2024", "1/25/2025", 'm'),
        principal_payments=np.array([0.0, 5.0, 7.0, 9.0]),
        interest_payments=np.array([0.0, 0.0, 0.0, 0.0])
    )

    # Define a list of settle dates to calculate the value of the dummy cash flows for
    settle_dates = pd.to_datetime(["10/01/2024", "10/15/2024", "11/10/2024", "11/15/2024"])

    print("\nSettle Value of Exercise Dummy Cash Flows")
    for date in settle_dates:
        print(f"Settle Date: {date}, Value: {value_cash_flows(discounter, exercise_flows, date)}")

    # Let's now calculate the value, price, WAL, balance at settle, and OAS for the exercise MBS based on the same 4 settle setles
    # Store it as a function for later use
    def print_mbs_attributes(mbs_flows, net_cpn, step_discounter, settle_date, find_oas = True):
        wal = calculate_weighted_average_life(mbs_flows, settle_date)
        val = value_cash_flows(step_discounter, mbs_flows, settle_date)
        settle_bal = get_balance_at_settle(mbs_flows, filter_cash_flows(mbs_flows, settle_date))
        price = price_cash_flows(val, settle_bal, settle_date, get_last_accrual_date(mbs_flows, settle_date), net_cpn)
        if find_oas:
            oas = oas_search(mbs_flows, step_discounter, settle_date)
            print(f"Settle Date: {date}, Settle Balance: {settle_bal}, "
              f"WAL: {wal}, Value: {val}, Price: {price}, OAS: {oas}")
        else:
            print(f"Settle Date: {date}, Settle Balance: {settle_bal}, "
              f"WAL: {wal}, Value: {val}, Price: {price}")
        
    print("\nAttributes of Exercise MBS Cash Flows")
    for date in settle_dates:
        print_mbs_attributes(actual_balances, 0.0675, discounter, date)

    # Define a SemiBondContract based on the exercise data and print its flows
    semi_bond = SemiBondContract("2024-10-01", 24, 0.5)
    semi_bond_flows = create_semi_bond_cash_flows(semi_bond)
    print("\nSemi Bond Cash Flows")
    print(semi_bond_flows)

    # The coarse and fine forward curves described in these exercise are calculated
    # outside this function so just print snapshots of each
    print(f"\nCoarse Curve: {[coarse_curve.dates[:5], coarse_curve.rates[:5]]}")
    print(f"\nFine Curve: {[fine_curve.dates[:5], fine_curve.rates[:5]]}")

    # Use the coarse curve to calculate the exercise forward bond rates
    effective_dates = pd.to_datetime(["2024-10-01", "2026-05-15", "2030-09-14"])
    terms = [2, 5, 10]

    print("\nExercise Coupon Rates")
    for date in effective_dates:
        for term in terms:
            cpn = calculate_coupon_rate(date, term, coarse_curve)
            print(f"Date: {date}, Term: {term}, Forward Rate: {cpn}")

    # Define some rate shocks to use for the improved prepayment model
    shocks = [-0.005, 0, 0.005]

    print("\nAttributes of Exercise MBS Cash Flows With OAS and Shocks")
    for shock in shocks:
        print(f"\nCurrent Shock: {shock}")
        smm = calculate_smms(calculate_pccs(rate_grid + shock, date_grid, sched_balances.accrual_dates, spread = 0.033), 0.07, sched_balances.accrual_dates)[:-1]
        shocked_discounter = StepDiscounter(discounter.dates, discounter.rates + 0.03 + shock)
        for date in settle_dates:
            print_mbs_attributes(calculate_actual_balances(sched_balances, smm, 0.0675), 0.0675, shocked_discounter, date, find_oas=False)
    
    # Define a more precise grid off shocks to plot the value of the MBS through time with those shocks
    shocks = (np.arange(31) * 0.001) - 0.015

    # Also define a function that takes in a shock, a StepDiscounter,
    # a CashFlowData instance, a gross annual coupon, a net annual coupon, a spread, an oas, and a settle date
    def value_shock(shock, discounter, scheduled_flows, gross_cpn, net_cpn, spread, oas, settle_date):
        """
        Calculate the values of cash flows under an interest rate shock.

        This function evaluates the impact of an interest rate shock on the value of cash flows. 
        It adjusts the discount rates, calculates primary customer coupons (PCCs), computes single monthly 
        mortality (SMM) rates, updates actual balances, and then values the cash flows for the shock.

        Parameters:
        ----------
        shock : float
            An interest rate shock to apply.
        discounter : StepDiscounter
            An instance of `StepDiscounter` that provides discount rates and dates.
        scheduled_flows : CashFlowData
            An instance containing scheduled balance data, including payment and accrual dates.
        gross_cpn : float
            The gross annual coupon rate (as a decimal, e.g., 0.05 for 5%).
        net_cpn : float
            The net annual coupon rate (after servicing fees, as a decimal).
        spread : float
            The spread to apply when calculating primary customer coupons (PCCs).
        oas : float
            The option-adjusted spread to add to the discount rates for valuation.
        settle_date : datetime
            The settle date for valuation.

        Returns:
        -------
        value : float
            The computed value of the cash flows for the input shock.
        """
        # Calculate the primary customer coupons (PCCs) based on the shocked discount rates and spread.
        pccs = calculate_pccs(
            discounter.rates + shock,  # Apply the shock to the discount rates.
            discounter.dates,
            scheduled_flows.accrual_dates,
            spread=spread
        )
        
        # Derive the single monthly mortality (SMM) rates from the PCCs and gross coupon rate.
        smm = calculate_smms(pccs, gross_cpn, scheduled_flows.accrual_dates)[:-1]
        
        # Create a shocked discounter by adding the OAS and shock to the discount rates.
        shocked_discounter = StepDiscounter(
            discounter.dates, 
            discounter.rates + oas + shock
        )
        
        # Compute the actual balances based on the shocked SMMs and net coupon rate.
        actual_balances = calculate_actual_balances(
            scheduled_flows, 
            smm, 
            net_cpn
        )
        
        # Value the cash flows using the shocked discounter and the actual balances.
        value = value_cash_flows(shocked_discounter, actual_balances, settle_date)

        return value

    vals = [value_shock(shock, discounter, sched_balances, 0.07, 0.0675, 0.033, 0.03, "2024-10-01") for shock in shocks]

    plt.figure(figsize=(10, 6))
    plt.scatter(shocks, vals)
    plt.xlabel("Shocks")
    plt.ylabel("Values")
    plt.show()

    target_value = value_shock(0.0015, discounter, sched_balances, 0.07, 0.0675, 0.033, 0.03, "2024-10-01")

    shock_vals = [0.0001, 0.001, 0.002, 0.003, 0.004, 0.005]

    for shock_val in shock_vals:
        vals = []
        for shock_direction in [-1, 0, 1]:
            shock = shock_val * shock_direction
            shocked_mbs_val = value_shock(shock, discounter, sched_balances, 0.07, 0.0675, 0.033, 0.03, "2024-10-01")
            vals.append(shocked_mbs_val)
        shock_val_bp = shock_val * 100
        target_bump_val_bp = 0.0015 * 100
        dv01 = calculate_dv01(vals[2], vals[0], shock_val_bp)
        convexity = calculate_convexity(vals[1], vals[2], vals[0], shock_val_bp)
        taylor_val_est = vals[1] + dv01 * target_bump_val_bp + (convexity * target_bump_val_bp ** 2) / 2
        error = taylor_val_est - target_value

        print(f"Shock: {shock_val}, DV01: {dv01}, Convexity: {convexity}", 
              f"Base Val: {vals[1]}, Target Val: {target_value}, "
              f"2nd Order Approximation: {taylor_val_est}, Error: {error}")

def main():
    # Define the file paths here
    calibration_file = 'data/daily-treasury-rates.csv'
    mbs_file = 'data/mbs_data.csv'
    
    # Load data
    market_close_date, calibration_data = load_treasury_rates_data(calibration_file)
    mbs_contracts = load_mbs_data(mbs_file)

    # Get the calibration data as a tuple including the date for each element
    calibration_data_with_dates = np.array([(market_close_date,) + calibration_bond for calibration_bond in calibration_data])

    # Calculate forward curves
    coarse_curve = bootstrap_forward_curve(market_close_date, calibration_data_with_dates)
    fine_curve = calibrate_fine_curve(market_close_date, calibration_data_with_dates, smoothing_error_weight=50000)

    # Run the exercises outlined in:
    # https://colab.research.google.com/drive/1kBUtBgGQ7uytfb6BrAUgF-zJbG_5mC1F?usp=sharing
    run_exercises(coarse_curve, fine_curve)
    
    # Plot the curves and their ZCB prices
    plot_forward_curves(coarse_curve, fine_curve)
    plot_forward_curve_zcb_prices([coarse_curve, fine_curve], ['coarse curve', 'fine_curve'])

    # Define the model paramters for the following Hull-White simulations
    alpha = 0.03
    sigma = 0.01
    num_iterations = 1000

    fine_curve.set_rates(fine_curve.rates*100)

    print(f"Alpha: {alpha}, Sigma: {sigma}, Number of Iterations: {num_iterations}")

    # Define the short rate dates to be used for the Hull-White simulation
    # In this case we will use the already defined monthly grid from the fine curve
    short_rate_dates = fine_curve.dates

    # Use Hull-White to simulate short rates based on the fine forward curve data
    hull_white = hull_white_simulate_from_curve(alpha, sigma, fine_curve, short_rate_dates, num_iterations)

    # Create a second simulation with no antithetic sampling to compare to the original Hull-White simulation
    hw_no_antithetic = hull_white_simulate_from_curve(alpha, sigma, fine_curve, short_rate_dates, num_iterations, False)

    # Create separate simulations with low number of iterations to plot and compare antithetic vs general sampling paths
    low_path_iterations = 6 # Define the number of paths to be simulated for the low number of iterations simulation
    small_alpha = 0.015 # Define an small alpha to limit the mean reversion effect. This will allow the difference in sampling paths to be more pronounced.
    hw_low_paths_1 = hull_white_simulate_from_curve(small_alpha, sigma, fine_curve, short_rate_dates, low_path_iterations, False)
    hw_low_paths_2 = hull_white_simulate_from_curve(small_alpha, sigma, fine_curve, short_rate_dates, low_path_iterations)

    # Plot to compare the Hull-White simulations to the fine forward curve
    plot_hull_white(hull_white, fine_curve)
    plot_hull_white(hw_no_antithetic, fine_curve, title='No Antithetic Hull-White Average Path vs Forward Curve')
    plot_hull_white_paths(hw_low_paths_1, fine_curve, title='No Antithetic Hull-White Paths vs Forward Curve')
    plot_hull_white_paths(hw_low_paths_2, fine_curve, title='Antithetic Hull-White Paths vs Forward Curve')

    # Plot the ZCB prices based on short rates from the antithetic low path number Hull-White simulation
    plot_hull_white_path_zcb_prices(hw_low_paths_2, fine_curve)

    # Plot the ZCB prices based on the average short rate paths for each of the Hull-White simulations done
    plot_hull_white_avg_zcb_prices([hull_white, hw_no_antithetic, hw_low_paths_1, hw_low_paths_2], fine_curve)

    # Extract the short rate paths from the Hull-White simulations
    short_rates = hull_white[1]/100
    no_antithetic_short_rates = hw_no_antithetic[1]/100

    # Simulate expected WALs, values, prices, and their standard deviations for each set of short rates
    simulated_mbs_values = pathwise_evaluate_mbs(mbs_contracts, short_rates, short_rate_dates, antithetic=True)
    no_antithetic_mbs_values = pathwise_evaluate_mbs(mbs_contracts, no_antithetic_short_rates, short_rate_dates)

    for index, mbs in enumerate(simulated_mbs_values):
        # Print the evalution results for each MBS in simulated_mbs_values
        print(f"MBS_ID: {mbs['mbs_id']}, Expected WAL: {mbs['expected_wal']}, Expected Value: {mbs['expected_value']}, "
              f"Expected Price: {mbs['expected_price']}, \nWAL Path STDev: {mbs['wal_stdev']}, "
              f"Value Path STDev: {mbs['value_stdev']}, Price Path STDev: {mbs['price_stdev']}")
        
        # Print the price variance of antithetic vs regular sampling Hull-White simmulations
        print(f"Antithetic Sampling Price Variance: {mbs['price_stdev'] ** 2}, No Antithetic Sampling Price Variance: {no_antithetic_mbs_values[index]['price_stdev'] ** 2}")

if __name__ == '__main__':
    main()
    