import numpy as np
import pandas as pd
from utils import create_regular_dates_grid
from financial_calculations.bonds import (
    StepDiscounter,
    SemiBondContract,
    create_semi_bond_cash_flows
)

from financial_calculations.cash_flows import (
    calculate_weighted_average_life,
    value_cash_flows,
    price_cash_flows
)
from financial_calculations.mbs import (
    MbsContract,
    calculate_actual_balances,
    calculate_scheduled_balances
)

def exercise_ppm(accrual_dates, scheduled_bals, gross_in_decimal, pcc_func):
    term = accrual_dates.size-1
    smm_vec = np.zeros(term)
    for t in range(term):
        smm_vec[t] = t/60 * .01 if t < 60 else .015 - t/120 * .01
    return smm_vec

def exercise_rates():
    discount_rates = np.array([ 
        0.006983617, 0.050476979, 0.051396376, 0.081552298,
        0.045289981, 0.029759723, 0.073969810, 0.003862879, 0.044871200,
        0.080978350, 0.097701138, 0.039590508, 0.049888840, 0.044515418,
        0.012596944, 0.008476654, 0.055574490, 0.035067305, 0.027167920,
        0.050371274, 0.032307761, 0.063749816, 0.008043366, 0.000253558,
        0.009324560, 0.093725868, 0.018888765, 0.024112974, 0.079826671,
        0.043751914, 0.021189997, 0.056026806, 0.085557557, 0.093818736,
        0.082996689, 0.054250755, 0.001240193, 0.025839389, 0.016727361,
        0.080124400, 0.009524058, 0.094808205, 0.038406391, 0.026020178,
        0.068923774, 0.032581043, 0.042404405, 0.038393590, 0.096957855,
        0.009599077, 0.082073126, 0.038449382, 0.008570729, 0.031334810,
        0.020902689, 0.007693663, 0.075357745, 0.005020825, 0.091085792,
        0.066223832, 0.004008308, 0.082499100, 0.026124495, 0.013431940,
        0.061752758, 0.015007716, 0.067035171, 0.045503376, 0.071814626,
        0.035016636, 0.047617013, 0.036258346, 0.052667676, 0.049712667,
        0.087182244, 0.046867727, 0.056173239, 0.088372271, 0.079152652,
        0.085316035, 0.050163748, 0.092035497, 0.084334787, 0.012739123,
        0.086784384, 0.082238121, 0.012813235, 0.045083599, 0.076907051,
        0.016017826, 0.062559817, 0.071020318, 0.038820162, 0.015048803,
        0.072752192, 0.026428880, 0.019477818, 0.052925817, 0.013761646,
        0.025814134, 0.003362728, 0.097627457, 0.022566484, 0.014211485,
        0.009787030, 0.092952964, 0.062951546, 0.056557990, 0.028243254,
        0.047776915, 0.094574003, 0.075722719, 0.090114523, 0.037370282,
        0.060771702, 0.099045273, 0.047339119, 0.030011234, 0.026550502,
        0.059545442, 0.029886582, 0.017509765, 0.067687091, 0.019248311,
        0.048724795, 0.087316041, 0.082405213, 0.000383088, 0.052046979,
        0.034628922, 0.051488041, 0.039743271, 0.054243464, 0.057612575,
        0.006979987, 0.023464708, 0.087048217, 0.018840603, 0.029695179,
        0.073279064, 0.057930599, 0.084524461, 0.012712518, 0.012014110,
        0.082756616, 0.037463306, 0.097436147, 0.080246738, 0.040601990,
        0.058930602, 0.086115746, 0.088906747, 0.074375454, 0.080537366,
        0.055050880, 0.051720078, 0.070774953, 0.074015762, 0.096252685,
        0.052755209, 0.013849016, 0.090894101, 0.001734406, 0.061806135,
        0.090170217, 0.054950115, 0.079689761, 0.088656840, 0.016996897,
        0.041160525, 0.061011024, 0.096765968, 0.053248733, 0.084173193,
        0.008111603, 0.048784956, 0.086477867, 0.046061337, 0.023838794,
        0.009723155, 0.009723155]) ## 0.050000000 ])
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

if (True):
    print("\n\nExercise MBS info...")
    date_grid = create_regular_dates_grid("1/1/2008", "1/1/2023", 'm')
    rate_grid = exercise_rates()
    discounter = StepDiscounter(date_grid, rate_grid)
    term = 180
    gross = .070
    net = gross -.0025
    delay = 24
    mbs_contract = MbsContract('id', 100, date_grid[0], term, gross, net, delay, pd.to_datetime("1/01/2008"))
    sched_mbs_flows = calculate_scheduled_balances(mbs_contract.balance, mbs_contract.origination_date, mbs_contract.num_months, mbs_contract.gross_annual_coupon)
    smms = exercise_ppm(sched_mbs_flows.accrual_dates, sched_mbs_flows.balances, mbs_contract.gross_annual_coupon, 0)
    mbs_flows = calculate_actual_balances(sched_mbs_flows, smms, mbs_contract.net_annual_coupon)
    print_flows(mbs_flows, 12)
    print("\n\nsettle value of exercise MBS cash flows")
    print(value_cash_flows(discounter, mbs_flows, "1/01/2008"))
    print(value_cash_flows(discounter, mbs_flows, "1/15/2008"))

    print(price_cash_flows(value_cash_flows(discounter, mbs_flows, "1/01/2008"), 100, "1/01/2008", "1/01/2008", mbs_contract.net_annual_coupon))
    print(price_cash_flows(value_cash_flows(discounter, mbs_flows, "2/15/2008"), 99.68450506248092, "2/15/2008", "2/01/2008", mbs_contract.net_annual_coupon))

    print("\n\nsettle WAL of exercise  MBS cash flows")
    print(calculate_weighted_average_life(mbs_flows, "1/01/2008"))
    print(calculate_weighted_average_life(mbs_flows, "1/15/2008"))