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
    years_from_reference,
    filter_cash_flows,
    value_cash_flows,
    price_cash_flows,
    get_balance_at_settle,
    get_settle_accrual_date,
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
    # These rates actually correspond to the entries from the fine curve calibration done later
    rate_grid = np.array(fine_curve.rates)
    
    # Now define the discount date grid associated with the discount rates
    date_grid = np.array(fine_curve.dates)

    # Initialize a StepDiscounter and ZCB dates grid for ZCB calculations
    discounter = StepDiscounter(date_grid, rate_grid)
    zcb_dates = pd.to_datetime(["2024-10-01", "2024-11-01", "2027-02-15", "2039-10-15", "2042-4-01", "2044-10-01", "2047-04-01", "2054-10-01"])

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
        settle_bal = get_balance_at_settle(mbs_flows, settle_date)
        price = price_cash_flows(val, settle_bal, settle_date, get_settle_accrual_date(mbs_flows, settle_date), net_cpn)
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
        smm = calculate_smms(calculate_pccs(rate_grid + shock, spread = 0.033), date_grid, sched_balances.accrual_dates[:-1], 0.07)
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
            discounter.rates + shock,
            spread=spread
        )
        
        # Derive the single monthly mortality (SMM) rates from the PCCs and gross coupon rate.
        smm = calculate_smms(pccs, discounter.dates, scheduled_flows.accrual_dates[:-1], gross_cpn)
        
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
    #fine_curve = calibrate_fine_curve(market_close_date, calibration_data_with_dates, smoothing_error_weights=[2500, 500])
    fine_curve = StepDiscounter(create_regular_dates_grid(coarse_curve.dates[0], coarse_curve.dates[-1]),
        rates= [0.040861075, 0.040759356, 0.040585068, 0.040325820, 0.039980431, 0.039554176,
            0.039042074, 0.038446557, 0.037766254, 0.037002718, 0.036154661, 0.035225321,
            0.034238589, 0.033344373, 0.032562165, 0.031897518, 0.031350018, 0.030912406,
            0.030591328, 0.030384055, 0.030292503, 0.030314527, 0.030452173, 0.030703158,
            0.031049646, 0.031387648, 0.031702604, 0.031991605, 0.032254552, 0.032492949,
            0.032705315, 0.032892379, 0.033053637, 0.033189508, 0.033299633, 0.033384448,
            0.033448149, 0.033514772, 0.033587738, 0.033667705, 0.033754823, 0.033848716,
            0.033949760, 0.034057623, 0.034172510, 0.034294188, 0.034422880, 0.034558505,
            0.034700961, 0.034850405, 0.035006596, 0.035169720, 0.035339634, 0.035515994,
            0.035699162, 0.035888961, 0.036085517, 0.036288764, 0.036498727, 0.036715182,
            0.036936252, 0.037150238, 0.037355594, 0.037551959, 0.037739397, 0.037918447,
            0.038088658, 0.038250113, 0.038402710, 0.038546580, 0.038681569, 0.038807718,
            0.038925211, 0.039034003, 0.039134340, 0.039226066, 0.039309124, 0.039384080,
            0.039450448, 0.039508350, 0.039557685, 0.039598683, 0.039631106, 0.039655318,
            0.039672646, 0.039692661, 0.039716521, 0.039744597, 0.039776962, 0.039813387,
            0.039854029, 0.039898738, 0.039947530, 0.040000344, 0.040057394, 0.040118561,
            0.040183865, 0.040253245, 0.040326682, 0.040404148, 0.040485649, 0.040570885,
            0.040660097, 0.040753236, 0.040850358, 0.040951346, 0.041056331, 0.041165298,
            0.041278205, 0.041395062, 0.041515914, 0.041640709, 0.041769373, 0.041901653,
            0.042037732, 0.042177653, 0.042321462, 0.042469129, 0.042620584, 0.042775822,
            0.042934012, 0.043090162, 0.043243690, 0.043394281, 0.043542046, 0.043687055,
            0.043829153, 0.043968508, 0.044104948, 0.044238640, 0.044369391, 0.044497419,
            0.044622560, 0.044744873, 0.044864389, 0.044981117, 0.045095102, 0.045206448,
            0.045315066, 0.045420992, 0.045524115, 0.045624523, 0.045722242, 0.045817146,
            0.045909333, 0.045998780, 0.046085532, 0.046169579, 0.046250930, 0.046329840,
            0.046406097, 0.046479670, 0.046550639, 0.046619009, 0.046684744, 0.046747930,
            0.046808589, 0.046866693, 0.046922229, 0.046975235, 0.047025703, 0.047073849,
            0.047119462, 0.047162633, 0.047203328, 0.047241661, 0.047277484, 0.047310789,
            0.047341569, 0.047369859, 0.047395866, 0.047419346, 0.047440414, 0.047459229,
            0.047475608, 0.047489635, 0.047501287, 0.047510627, 0.047517491, 0.047521999,
            0.047524138, 0.047523899, 0.047521289, 0.047516311, 0.047508938, 0.047499283,
            0.047487231, 0.047472937, 0.047456291, 0.047437449, 0.047416267, 0.047392752,
            0.047366903, 0.047338857, 0.047308643, 0.047276129, 0.047241392, 0.047204511,
            0.047165371, 0.047124100, 0.047080648, 0.047035003, 0.046987160, 0.046936952,
            0.046884630, 0.046830073, 0.046773425, 0.046714507, 0.046653476, 0.046590360,
            0.046525136, 0.046457847, 0.046388437, 0.046316949, 0.046243335, 0.046167575,
            0.046089832, 0.046009939, 0.045928039, 0.045844024, 0.045757898, 0.045669788,
            0.045579494, 0.045487142, 0.045392687, 0.045296339, 0.045197808, 0.045097405,
            0.044994965, 0.044890565, 0.044784178, 0.044675943, 0.044565715, 0.044453609,
            0.044339642, 0.044223818, 0.044105981, 0.043986127, 0.043864319, 0.043740566,
            0.043615419, 0.043491278, 0.043368278, 0.043246461, 0.043125890, 0.043006436,
            0.042888477, 0.042771824, 0.042656419, 0.042542150, 0.042429217, 0.042317276,
            0.042206554, 0.042097052, 0.041988696, 0.041881652, 0.041775938, 0.041671412,
            0.041568170, 0.041465872, 0.041364753, 0.041264728, 0.041165871, 0.041068136,
            0.040971467, 0.040875960, 0.040781567, 0.040688475, 0.040596543, 0.040505910,
            0.040416271, 0.040327748, 0.040240394, 0.040154010, 0.040068820, 0.039984721,
            0.039901724, 0.039819807, 0.039739015, 0.039659441, 0.039580983, 0.039503659,
            0.039427417, 0.039352386, 0.039278531, 0.039205688, 0.039133879, 0.039063096,
            0.038993429, 0.038924780, 0.038857304, 0.038790914, 0.038725614, 0.038661388,
            0.038598206, 0.038536133, 0.038475055, 0.038415067, 0.038356180, 0.038298392,
            0.038241570, 0.038185895, 0.038131262, 0.038077682, 0.038025136, 0.037973578,
            0.037923063, 0.037873528, 0.037825092, 0.037777681, 0.037731354, 0.037686059,
            0.037641703, 0.037598351, 0.037556014, 0.037514713, 0.037474428, 0.037435056,
            0.037396681, 0.037359266, 0.037322918, 0.037287542, 0.037253133, 0.037219654,
            0.037187194, 0.037155646, 0.037124983, 0.037095311, 0.037066608, 0.037038905,
            0.037012064, 0.036986076, 0.036961020, 0.036936911, 0.036913751, 0.036891611,
            0.036870354, 0.036849933, 0.036830323, 0.036811646, 0.036793944, 0.036777060,
            0.036761108, 0.036746041, 0.036731934, 0.036718739, 0.036706458, 0.036695175,
            0.036684588, 0.036674924, 0.036666076, 0.036658237, 0.036651161, 0.036644896,
            0.036639625, 0.036635220, 0.036631736, 0.036629110, 0.036627346, 0.036626355,
            0.036626118])

    # Run the exercises outlined in:
    # https://colab.research.google.com/drive/1kBUtBgGQ7uytfb6BrAUgF-zJbG_5mC1F?usp=sharing
    run_exercises(coarse_curve, fine_curve)
    
    # Plot the curves and their ZCB prices
    plot_forward_curves(coarse_curve, fine_curve)
    plot_forward_curve_zcb_prices([coarse_curve, fine_curve], ['coarse curve', 'fine_curve'])

    # Define the model paramters for the following Hull-White simulations
    alpha = 0.03
    sigma = 0.01
    num_iterations = 32000

    print(f"Alpha: {alpha}, Sigma: {sigma}, Number of Iterations: {num_iterations}")

    # Define the short rate dates to be used for the Hull-White simulation
    # In this case we will use the already defined monthly grid from the fine curve
    short_rate_dates = fine_curve.dates

    # Use Hull-White to simulate short rates based on the fine forward curve data
    hull_white = hull_white_simulate_from_curve(alpha, sigma, fine_curve, short_rate_dates, num_iterations)

    def hull_white_zcb_log_var(sigma, alpha, zcb_times):
        zcb_times = np.array(zcb_times)
        a_Ts = alpha*zcb_times
        exp_neg_a_Ts = np.exp(-a_Ts)
        B = 1/alpha * (1-exp_neg_a_Ts)
        V = (sigma/alpha)**2 * (zcb_times -B) - sigma**2/(2*alpha) * B**2
        return V

    def hull_white_zcb_var(sigma, alpha, zcb_times, zcb_prices):
        V = hull_white_zcb_log_var(sigma, alpha, zcb_times)
        return zcb_prices**2 * (np.exp(V) -1)

    # Create a second simulation with no antithetic sampling to compare to the original Hull-White simulation
    hw_no_antithetic = hull_white_simulate_from_curve(alpha, sigma, fine_curve, short_rate_dates, num_iterations, False)

    zcb_dates = pd.to_datetime(["2054-10-01"])
    dummy_disc = StepDiscounter(fine_curve.dates, hw_no_antithetic[1][0])
    count = []
    for rates in hw_no_antithetic[1]:
        dummy_disc.set_rates(rates)
        count.append(dummy_disc.zcbs_from_dates(zcb_dates))


    print(f"\nZCBs: {np.mean(count)}")
    
    dummy_disc = StepDiscounter(fine_curve.dates, fine_curve.rates)
    sdfs = []
    for calibration_point in calibration_data_with_dates:
        sdf_vals_list = []
        zcb_time = calibration_point[0] + pd.DateOffset(years=calibration_point[1])
        zcb_price = 100*fine_curve.zcbs_from_dates(zcb_time)
        if calibration_point[1] == 1:
            benchmark_zcb_price = zcb_price
        for rate_path in hw_no_antithetic[1]:
            dummy_disc.set_rates(rate_path)
            sdf_val = dummy_disc.zcbs_from_dates(zcb_time)
            sdf_vals_list.append(sdf_val)
        sdfs.append(np.asarray(sdf_vals_list))
        print(f"Current SDF Time Point: {calibration_point[1]}")
        print(f"MC var: {np.asarray(sdf_vals_list).var()}")
        print(f"FC theoretical var: {hull_white_zcb_var(sigma, alpha, calibration_point[1], zcb_price)}")
        print(f"MC covar: {np.cov([sdfs[0], sdfs[-1]])[0, 1]}")
        print(np.cov([sdfs[0], sdfs[-1]])[0, 1] / (np.sqrt(np.asarray(sdf_vals_list).var()) * np.sqrt(np.asarray(sdfs[0]).var())))

    # Extract the short rate paths from the Hull-White simulations
    short_rates = hull_white[1]
    no_antithetic_short_rates = hw_no_antithetic[1]

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
    