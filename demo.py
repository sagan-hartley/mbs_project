import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from financial_calculations.forward_curves import bootstrap_forward_curve, bootstrap_finer_forward_curve
from financial_calculations.mbs_cash_flows import calculate_scheduled_balances


def load_treasury_rates_data(file_path):
    # Load daily treasury rates data from CSV files
    # Read the original CSV into a DataFrame
    df = pd.read_csv(file_path)

    # Extract the date
    date_str = df['Date'].iloc[0]
    date = datetime.strptime(date_str, "%m/%d/%Y")

    # Remove the 'Date' column to focus on maturity years and rates
    maturity_year_columns = df.columns[1:]  # All columns except 'Date'
    
    # Extract the rates from the first row (since it's only one row of data)
    rates = df.iloc[0, 1:].tolist()  # All values in the first row except the date
    
    # Create a list of tuples with maturity year (as int) and corresponding rate (as percentage)
    maturity_rate_tuples = [(int(col.split()[0]), rate/100) for col, rate in zip(maturity_year_columns, rates)]
    
    # Return the date and the list of tuples
    return date, maturity_rate_tuples

def load_mbs_data(mbs_file):
    # Load MBS data from CSV files
    mbs_data = pd.read_csv(mbs_file)

    return mbs_data

def plot_curves(coarse_curve, fine_curve):
    # Plot the two curves
    plt.figure(figsize=(10, 6))
    plt.plot(coarse_curve[0], np.append(coarse_curve[1], coarse_curve[1][-1]), label='Coarse Curve', color='blue')
    plt.plot(fine_curve[0], fine_curve[1], label='Fine Curve', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Rate')
    plt.title('Forward Curves')
    plt.legend()
    plt.grid()
    plt.show()

def price_mbs_cash_flows(mbs_df, coarse_curve, fine_curve):
    # Helper method to price MBS cash flows
    results = []
    for idx, mbs in mbs_df.iterrows():
        coarse_price = price_cash_flows(mbs, coarse_curve)
        fine_price = price_cash_flows(mbs, fine_curve)
        results.append({
            'MBS': mbs['MBS_ID'],
            'Coarse Price': coarse_price,
            'Fine Price': fine_price
        })
        print(f"MBS {mbs['MBS_ID']}: Coarse Price = {coarse_price}, Fine Price = {fine_price}")
    return pd.DataFrame(results)

def main():
    # Define your file paths here
    calibration_file = 'data/daily-treasury-rates.csv'
    mbs_file = 'data/mbs_data.csv'
    
    # Load data
    effective_rate_date, calibration_data = load_treasury_rates_data(calibration_file)
    mbs_data = load_mbs_data(mbs_file)
    
    # Calculate forward curves
    coarse_curve = bootstrap_forward_curve(calibration_data, effective_rate_date, 100)
    fine_curve = bootstrap_finer_forward_curve(calibration_data, effective_rate_date, 100)
    
    # Plot the curves
    plot_curves(coarse_curve, fine_curve)
    
    # Price MBS cash flows
    #results = price_mbs_cash_flows(mbs_data, coarse_curve, fine_curve)
    #print(results)

if __name__ == '__main__':
    main()