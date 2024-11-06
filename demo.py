from dateutil.relativedelta import relativedelta
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from utils import integer_months_from_reference
from financial_models.hull_white import (
    hull_white_simulate_from_curve
)
from financial_models.prepayment import (
    calculate_pccs,
    calculate_smms
)
from financial_calculations.bonds import (
    SemiBondContract,
    create_semi_bond_cash_flows
)
from financial_calculations.forward_curves import (
    bootstrap_forward_curve,
    calibrate_fine_curve
)
from financial_calculations.mbs import (
    calculate_scheduled_balances,
    calculate_actual_balances
)
from financial_calculations.cash_flows import (
    StepDiscounter,
    value_cash_flows,
    evaluate_cash_flows,
    calculate_dv01,
    calculate_convexity
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
    date = pd.to_datetime(date_str)

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
    mbs_data[SETTLE_DATE_IDX] = mbs_data[SETTLE_DATE_IDX].apply(pd.to_datetime)
    mbs_data[ORIGINATION_DATE_IDX] = mbs_data[ORIGINATION_DATE_IDX].apply(pd.to_datetime)

    # Convert values to a list
    mbs_data = mbs_data.values.tolist()

    return mbs_data

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
    plt.step(coarse_curve.dates, coarse_curve.rates, where='post', label='Coarse Curve', color='blue')
    plt.step(fine_curve.dates, fine_curve.rates, where='post', label='Fine Curve', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Rate')
    plt.title('Forward Curves')
    plt.legend()
    plt.grid()
    plt.show()

def evaluate_mbs_short_rate_paths(mbs_data, short_rates, short_rate_dates):
    """
    Calculate the expected weighted average life, value, price, as well as path standard deviations for the MBS based on short rate paths.

    Parameters:
    - mbs_data (list): A list containing details of the MBSs including ID, balance, number of months, gross and net annual coupons, accrual dates, single monthly mortality (SMM) rate, and payment delay days.
    - short_rates (ndarray): Array of short rates from the simulation.
    - short_rate_dates (ndarray): Array of dates corresponding to the short rates.

    Returns:
    - results (list): A list of dictionaries containing MBS ID, WALs, expected WAL, values, expected value, prices, expected price, and standard deviations for each MBS.
    """
    results = []  # To store results for each MBS
    market_close_date = short_rate_dates[0]  # Extract the market close date from the short rate dates

    # Loop through each MBS in the provided data
    for mbs in mbs_data:
        wals, vals, prices = [], [], []  # Lists to store results for the current MBS

        # Unpack the MBS details
        mbs_id, balance, num_months, gross_annual_coupon, net_annual_coupon, settle_date, origination_date, payment_delay = mbs

        # Calculate the necessary scheduled balance data to value and price the cash flows
        scheduled_balances = calculate_scheduled_balances(balance, origination_date, num_months, gross_annual_coupon, payment_delay=payment_delay)

        # Calculate the Primary Current Coupons (PCCs) and SMMs based on the original short rates
        pccs = calculate_pccs(short_rates)
        smms = calculate_smms(pccs, gross_annual_coupon, market_close_date, origination_date, num_months)

        # Loop through each SMM path to calculate cash flows
        for index, smm_path in enumerate(smms):
            # Calculate the actual scheduled balances based on the current SMM
            actual_balances = calculate_actual_balances(scheduled_balances, smm_path, net_annual_coupon)

            # Define an instance of StepDiscounter based on the current short rate path
            discounter = StepDiscounter(short_rate_dates, short_rates[index, :])

            # Evaluate the cash flows for the current MBS using the actual balances, SMM, and current discounter
            wal, val, price = evaluate_cash_flows(actual_balances, discounter, settle_date, net_annual_coupon)
            
            # Store the calculated values
            wals.append(wal)
            vals.append(val)
            prices.append(price)

        # Calculate means for the WAL, value, and price, DV01 of the MBS
        expected_wal = np.mean(wals)
        expected_value = np.mean(vals)
        expected_price = np.mean(prices)

        # Calculate standard deviations
        wal_stdev = np.std(wals)
        val_stdev = np.std(vals)
        price_stdev = np.std(prices)

        # Store the result in a dictionary for structured output
        results.append({
            'mbs_id': mbs_id,
            'wals': wals,
            'expected_wal': expected_wal,
            'wal_stdev': wal_stdev,
            'vals': vals,
            'expected_value': expected_value,
            'value_stdev': val_stdev,
            'prices': prices,
            'expected_price': expected_price,
            'price_stdev': price_stdev
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
    # Get the number of short rate paths and initialize an array to store ZCB present values
    num_paths = short_rate_paths.shape[0]
    present_values = np.zeros(num_paths)

    # Define the market close date and the term in months of the bond from input data
    market_close_date = discount_rate_dates[0]
    term_in_months = integer_months_from_reference(market_close_date, maturity_date)

    # Set up a single cash flow of 100 at the maturity date (for a ZCB) by using a SemiBondContract with zero coupon
    cash_flows = create_semi_bond_cash_flows(SemiBondContract(market_close_date, term_in_months, 0))

    # Loop over each path to calculate present value using the discount_cash_flows function
    for i in range(num_paths):
        # Use the short rates of the current path as the discount rates
        discounter = StepDiscounter(discount_rate_dates, short_rate_paths[i, :])
        
        # Discount the cash flows for this path
        present_values[i] = value_cash_flows(discounter, cash_flows, market_close_date)
    
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
    forward_curve (ForwardCurve): A ForwardCurve object with rate dates and rate values as attributes.
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
    forward_curve (ForwardCurve): A ForwardCurve object with rate dates and rate values as attributes.
    title (str): A string representing the title of the grpah. Default is 'Hull-White Paths vs Forward Curve'.

    Returns:
        None
    """
    num_paths = len(hull_white[1])
    colors = sns.color_palette("husl", num_paths)  # Generate distinct colors
    plt.figure(figsize=(10, 6))
    
    for index, rate in enumerate(hull_white[1]):
        plt.step(hull_white[0], rate, where='post', label=f'Hull-White Path {index + 1}', color=colors[index], alpha=0.6)
    
    plt.step(forward_curve.dates, forward_curve.rates, where='post', label='Fine Curve', color='orange')
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
    num_iterations = 100

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

    # Plot the ZCB prices based on short rates from the Hull-White simulation
    plot_hull_white_zcb_prices(hull_white)

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
    simulated_mbs_values = evaluate_mbs_short_rate_paths(mbs_data, short_rates, short_rate_dates)
    no_antithetic_mbs_values = evaluate_mbs_short_rate_paths(mbs_data, no_antithetic_short_rates, short_rate_dates)
    bumped_up_values = evaluate_mbs_short_rate_paths(mbs_data, bumped_up_short_rates, short_rate_dates)
    bumped_down_values = evaluate_mbs_short_rate_paths(mbs_data, bumped_down_short_rates, short_rate_dates)

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
    