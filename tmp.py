import numpy as np
import pandas as pd
from utils import create_regular_dates_grid
from financial_calculations.bonds import (
    StepDiscounter,
    SemiBondContract,
    create_semi_bond_cash_flows,
    calculate_coupon_rate
)

from financial_calculations.cash_flows import (
    CashFlowData,
    calculate_weighted_average_life,
    value_cash_flows,
    price_cash_flows,
    oas_search,
    evaluate_cash_flows,
    calculate_convexity
)
from financial_calculations.mbs import (
    MbsContract,
    calculate_actual_balances,
    calculate_scheduled_balances,
    pathwise_evaluate_mbs
)
from financial_calculations.forward_curves import (
    bootstrap_forward_curve
)
from financial_models.prepayment import (
    calculate_pccs,
    demo,
    refi_strength,
    calculate_smms
)

def exercise_ppm(accrual_dates):
    term = accrual_dates.size-1
    smm_vec = np.zeros(term)
    for t in range(term):
        smm_vec[t] = t/60 * .01 if t < 60 else .015 - t/120 * .01
    return smm_vec

def exercise_rates():
    discount_rates = np.array([
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
    return discount_rates

def print_flows(cash_flows, max_index=None):
    if max_index:
        max_index = min(max_index, cash_flows.get_size()-1)
    else:
        max_index = cash_flows.get_size()-1

    prin_pmnts = -np.diff(cash_flows.balances, prepend=cash_flows.balances[0])
    cpn_pmnts = cash_flows.get_total_payments() - prin_pmnts
    df = pd.DataFrame({
        "AccrualDate": cash_flows.accrual_dates,
        "PaymentDate": cash_flows.payment_dates,
        "Balance": cash_flows.balances,
        "TotPayment": cash_flows.get_total_payments(),
        "PrincipalPayment": prin_pmnts,
        "CouponPayment": cpn_pmnts
        })
    pd.options.display.float_format = "{:.4f}".format
    print(df.head(max_index+1))
    pd.reset_option("display.float_format")

def exercise_ppm(accrual_dates, scheduled_bals, gross_in_decimal, pcc_func):
    term = accrual_dates.size-1
    smm_vec = np.zeros(term)
    for t in range(term):
        smm_vec[t] = t/60 * .01 if t < 60 else .015 - t/120 * .01
    return smm_vec

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

calibration_file = 'data/daily-treasury-rates.csv'
    
# Load data
market_close_date, calibration_data = load_treasury_rates_data(calibration_file)

# Get the calibration data as a tuple including the date for each element
calibration_data_with_dates = np.array([(market_close_date,) + calibration_bond for calibration_bond in calibration_data])

# Calculate forward curves
coarse_curve = bootstrap_forward_curve(market_close_date, calibration_data_with_dates)

start_date = pd.to_datetime("2026-05-15")
#print(100*calculate_coupon_rate(start_date, 10, coarse_curve))

if (True):
    print("\n\nExercise MBS info...")
    date_grid = create_regular_dates_grid("10/1/2024", "10/1/2039", 'm')
    rate_grid = exercise_rates()
    up_rates = rate_grid + 0.005
    down_rates = rate_grid - 0.005
    discounter = StepDiscounter(date_grid, rate_grid)
    zcb_dates = pd.to_datetime(["2024-10-01", "2024-11-01", "2027-02-15", "2039-10-15"])
    print(discounter.zcbs_from_dates(zcb_dates))
    fake_flows = CashFlowData(
        balances=np.array([21.0, 16.0, 9.0, 0.0]),
        accrual_dates=create_regular_dates_grid("10/1/2024", "1/1/2025", 'm'),
        payment_dates=create_regular_dates_grid("10/25/2024", "1/25/2025", 'm'),
        principal_payments=np.array([0.0, 5.0, 7.0, 9.0]),
        interest_payments=np.array([0.0, 0.0, 0.0, 0.0])
    )
    print("\n\nsettle value of exercise fake cash flows")
    print(value_cash_flows(discounter, fake_flows, "10/01/2024"))
    print(value_cash_flows(discounter, fake_flows, "10/15/2024"))
    print(value_cash_flows(discounter, fake_flows, "11/01/2024"))
    print(value_cash_flows(discounter, fake_flows, "11/15/2024"))
    term = 180
    gross = .070
    net = gross -.0025
    delay = 24
    settle = pd.to_datetime("10/15/2024")
    mbs_contract = MbsContract('id', 100, date_grid[0], term, gross, net, delay, pd.to_datetime("10/01/2024"))
    sched_mbs_flows = calculate_scheduled_balances(mbs_contract.balance, mbs_contract.origination_date, mbs_contract.num_months, mbs_contract.gross_annual_coupon)
    smms = exercise_ppm(sched_mbs_flows.accrual_dates, sched_mbs_flows.balances, mbs_contract.gross_annual_coupon, 0)
    bad_flows = calculate_actual_balances(sched_mbs_flows, smms, net)
    print(f"oas: {oas_search(bad_flows, discounter, settle)}")
    pccs = calculate_pccs(rate_grid, sched_mbs_flows.accrual_dates, sched_mbs_flows.accrual_dates, 0.033)
    pccs_up = calculate_pccs(up_rates, sched_mbs_flows.accrual_dates, sched_mbs_flows.accrual_dates, 0.033)
    pccs_down = calculate_pccs(down_rates, sched_mbs_flows.accrual_dates, sched_mbs_flows.accrual_dates, 0.033)
    smms1 = calculate_smms(pccs, 0.07, date_grid)[:-1]
    smms_up = calculate_smms(pccs_up, 0.07, date_grid)[:-1]
    smms_down = calculate_smms(pccs_down, 0.07, date_grid)[:-1]
    mbs_flows = calculate_actual_balances(sched_mbs_flows, smms1, mbs_contract.net_annual_coupon)
    mbs_flows_up = calculate_actual_balances(sched_mbs_flows, smms_up, mbs_contract.net_annual_coupon)
    mbs_flows_down = calculate_actual_balances(sched_mbs_flows, smms_down, mbs_contract.net_annual_coupon)
    print_flows(mbs_flows, 20)
    print("\n\nsettle value of exercise MBS cash flows")
    print(value_cash_flows(discounter, mbs_flows, "10/01/2024"))
    print(value_cash_flows(discounter, mbs_flows, "10/15/2024"))
    print(value_cash_flows(discounter, mbs_flows, "11/10/2024"))
    print(value_cash_flows(discounter, mbs_flows, "11/15/2024"))

    print(value_cash_flows(discounter, mbs_flows_up, "10/01/2024"))
    print(value_cash_flows(discounter, mbs_flows_up, "10/15/2024"))
    print(value_cash_flows(discounter, mbs_flows_up, "11/10/2024"))
    print(value_cash_flows(discounter, mbs_flows_up, "11/15/2024"))

    print(value_cash_flows(discounter, mbs_flows_down, "10/01/2024"))
    print(value_cash_flows(discounter, mbs_flows_down, "10/15/2024"))
    print(value_cash_flows(discounter, mbs_flows_down, "11/10/2024"))
    print(value_cash_flows(discounter, mbs_flows_down, "11/15/2024"))

    print("\n\settle price of exercise MBS cash flows")
    print(price_cash_flows(value_cash_flows(discounter, mbs_flows, "10/01/2024"), 100, "10/01/2024", "10/01/2024", mbs_contract.net_annual_coupon))
    print(price_cash_flows(value_cash_flows(discounter, mbs_flows, "11/15/2024"), 99.68450506248092, "11/15/2024", "11/01/2024", mbs_contract.net_annual_coupon))

    print("\n\nsettle WAL of exercise  MBS cash flows")
    print(calculate_weighted_average_life(mbs_flows, "10/01/2024"))
    print(calculate_weighted_average_life(mbs_flows, "10/15/2024"))

    path_evals = pathwise_evaluate_mbs([mbs_contract], [rate_grid-0.005+0.03, rate_grid+0.03, rate_grid+0.005+0.03], date_grid)
    print(path_evals[0]['vals'])