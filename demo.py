import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from financial_models.hull_white import (
    hull_white_simulate_from_curve
)
from financial_calculations.bonds import (
    pathwise_zcb_eval
)
from financial_calculations.forward_curves import (
    bootstrap_forward_curve,
    calibrate_fine_curve
)
from financial_calculations.mbs import (
    MbsContract,
    pathwise_evaluate_mbs
)
from financial_calculations.cash_flows import (
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

    # Apply the datetime parsing to the Settle Date and Origination Date columns
    mbs_data[SETTLE_DATE_COL] = pd.to_datetime(mbs_data[SETTLE_DATE_COL])
    mbs_data[ORIG_DATE_COL] = pd.to_datetime(mbs_data[ORIG_DATE_COL])

    # Convert each row to an MBS instance and append to the list
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

def plot_hull_white_zcb_prices(hull_white, forward_curve, title='Hull-White Paths vs Forward Curve ZCB Values'):
    """
    Plots the zcb values based on short rates from a Hull-White simulation.
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
    
    # Plot the curves
    plot_forward_curves(coarse_curve, fine_curve)

    # Define the start rate based on information from the fine forward curve
    start_rate = fine_curve.rates[0]

    # Define alpha, sigma, and num_iterations
    alpha = 1
    sigma = 0.01
    num_iterations = 1000

    print(f"Alpha: {alpha}, Sigma: {sigma}, Number of Iterations: {num_iterations}")

    # Define the short rate dates to be used for the Hull-White simulation
    # In this case we will use the already defined monthly grid from the fine curve
    short_rate_dates = fine_curve.dates

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

    # Plot the ZCB prices based on short rates from the antithetic low path number Hull-White simulation
    plot_hull_white_zcb_prices(hw_low_paths_2, fine_curve)

    # Define a bump amount to create Hull-White simulations where the forward curve has been shocked up and down by this amount
    # These simulation results will be used to calculate the DV01 and convexity for the value of each MBS
    bump_amount = 0.0025
    bumped_up_curve = fine_curve
    bumped_up_curve.rates = bumped_up_curve.rates + bump_amount
    bumped_down_curve = fine_curve
    bumped_down_curve.rates = bumped_down_curve.rates - bump_amount
    bumped_up_hw = hull_white_simulate_from_curve(alpha, sigma, bumped_up_curve, short_rate_dates, start_rate, num_iterations)
    bumped_down_hw = hull_white_simulate_from_curve(alpha, sigma, bumped_down_curve, short_rate_dates, start_rate, num_iterations)

    # Extract the short rate paths from the Hull-White simulations
    short_rates = hull_white[1]
    no_antithetic_short_rates = hw_no_antithetic[1]
    bumped_up_short_rates = bumped_up_hw[1]
    bumped_down_short_rates = bumped_down_hw[1]

    # Simulate expected WALs, values, prices, and their standard deviations for each set of short rates
    simulated_mbs_values = pathwise_evaluate_mbs(mbs_contracts, short_rates, short_rate_dates)
    no_antithetic_mbs_values = pathwise_evaluate_mbs(mbs_contracts, no_antithetic_short_rates, short_rate_dates)
    bumped_up_values = pathwise_evaluate_mbs(mbs_contracts, bumped_up_short_rates, short_rate_dates)
    bumped_down_values = pathwise_evaluate_mbs(mbs_contracts, bumped_down_short_rates, short_rate_dates)

    for index, mbs in enumerate(simulated_mbs_values):
        # Print the evalution results for each MBS in the simulated_mbs_values
        print(f"MBS_ID: {mbs['mbs_id']}, Expected WAL: {mbs['expected_wal']}, Expected Value: {mbs['expected_value']}, "
              f"Expected Price: {mbs['expected_price']}, \nWAL Path STDev: {mbs['wal_stdev']}, "
              f"Value Path STDev: {mbs['value_stdev']}, Price Path STDev: {mbs['price_stdev']}")
        
        # Print the price variance of antithetic vs regular sampling Hull-White simmulations
        print(f"Antithetic Sampling Price Variance: {mbs['price_stdev'] ** 2}, No Antithetic Sampling Price Variance: {no_antithetic_mbs_values[index]['price_stdev'] ** 2}")

        # Extract the value arrays for the MBS based from the normal and bumped simulations
        vals = mbs['vals']
        bumped_up_vals = bumped_up_values[index]['vals']
        bumped_down_vals = bumped_down_values[index]['vals']

        # Calculate and print the expected DV01 by averaging the DV01s from bumping up and down 25 basis points
        # Note that we multiply bump_amount by 100 to convert from decimal to basis points
        dv01 = (calculate_dv01(bumped_up_vals, vals, bump_amount*100) +
                calculate_dv01(vals, bumped_down_vals, bump_amount*100)) / 2
        print(f"Expected DV01: {dv01}")

        # Calculate and print the convexity measure from the normal and bumped simulation values
        # Note that like in the DV01 calculation, we multiply bump_amount by 100 to convert from decimal to basis points
        convexity = calculate_convexity(vals, bumped_up_vals, bumped_down_vals, bump_amount*100)
        print(f"Convexity: {convexity}")

if __name__ == '__main__':
    main()
    