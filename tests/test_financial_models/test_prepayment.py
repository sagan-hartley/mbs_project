import unittest
import numpy as np
import pandas as pd
from financial_models.prepayment import (
    calculate_pccs,
    refi_strength,
    demo,
    calculate_smms
)

class TestPrepaymentModel(unittest.TestCase):
    """
    Test suite for the prepayment model functions including PCC calculations,
    refinancing strength, demographic factors, and SMM calculations.
    """

    def test_calculate_pccs_1d(self):
        """Test the calculation of Primary Current Coupons (PCCs) with 1D short rates."""
        short_rates = np.array([0.02, 0.03, 0.04])
        short_rate_dates = pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01'])
        cash_flow_dates = pd.to_datetime(['2024-01-15', '2024-02-15', '2024-03-15'])
        expected_pccs = np.array([0.06, 0.07, 0.08])  # [0.02 + 0.04, 0.03 + 0.04, 0.04 + 0.04]
        
        pccs = calculate_pccs(short_rates, short_rate_dates, cash_flow_dates)
        np.testing.assert_array_equal(pccs, expected_pccs)

    def test_calculate_pccs_2d(self):
        """Test the calculation of Primary Current Coupons (PCCs) with 2D short rates."""
        short_rates = np.array([[0.02, 0.03, 0.04], [0.015, 0.025, 0.035]])
        short_rate_dates = pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01'])
        cash_flow_dates = pd.to_datetime(['2024-01-15', '2024-02-15', '2024-03-15'])
        expected_pccs = np.array([[0.06, 0.07, 0.08], [0.055, 0.065, 0.075]])

        pccs = calculate_pccs(short_rates, short_rate_dates, cash_flow_dates)
        np.testing.assert_array_almost_equal(pccs, expected_pccs)

    def test_refi_strength(self):
        """Test the refinancing strength calculation."""
        spreads = np.array([-0.01, 0, 0.0075, 0.015, 0.02])
        expected_strength = np.array([0.0, 0.0, 0.02125, 0.0425, 0.0425])
        np.testing.assert_array_equal(refi_strength(spreads), expected_strength)

    def test_demo(self):
        """Test the demographic factor calculation."""
        smm_dates = pd.to_datetime([
            '2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-05-01', 
            '2024-06-01', '2024-07-01', '2024-08-01', '2024-09-01', '2024-10-01', 
            '2024-11-01', '2024-12-01'
        ])
        base_smm = 0.005
        expected_demo_factors = np.array([0.00375, 0.0042, 0.00465, 0.0051, 0.00555, 0.006, 0.006,
                                          0.00555, 0.0051, 0.00465, 0.0042, 0.00375])
        np.testing.assert_array_almost_equal(demo(smm_dates, base_smm), expected_demo_factors)

    def test_calculate_smms(self):
        """Test the Single Monthly Mortality (SMM) calculation."""
        pccs = np.array([[0.06, 0.07, 0.08], [0.065, 0.075, 0.085]])
        coupon = 0.08
        smm_dates = pd.to_datetime([
            '2024-01-01', '2024-02-01', '2024-03-01'
        ])
        
        expected_smms = np.array([
            [0.04625, 0.032533, 0.00465 ],
            [0.04625, 0.018367, 0.00465 ]
        ])

        smms = calculate_smms(pccs, coupon, smm_dates)
        np.testing.assert_array_almost_equal(smms, expected_smms)

    def test_calculate_smms_invalid_pccs_length(self):
        """Test the invalid case where the length of PCCs doesn't match the number of dates in smm_dates."""
        pccs = np.array([[0.06, 0.07], [0.065, 0.075]])  # Only 2 months
        coupon = 0.08
        smm_dates = pd.to_datetime([
            '2024-01-01', '2024-02-01', '2024-03-01'  # 3 months
        ])
        
        with self.assertRaises(ValueError):
            calculate_smms(pccs, coupon, smm_dates)

    def test_calculate_smms_with_lag_one_month(self):
        """Test SMM calculation with a 1-month lag on PCCs."""
        pccs = np.array([[0.06, 0.07, 0.08], [0.065, 0.075, 0.085]])
        coupon = 0.08
        smm_dates = pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01'])
        
        # Expected SMM with 1-month lag
        # Note that demographic factors are not lagged
        expected_smms = np.array([
            [0.04625, 0.0467, 0.032983],
            [0.04625, 0.0467, 0.018817]
        ])

        smms = calculate_smms(pccs, coupon, smm_dates, lag_months=1)
        np.testing.assert_array_almost_equal(smms, expected_smms)

    def test_calculate_smms_with_lag_two_months(self):
        """Test SMM calculation with a 2-month lag on PCCs."""
        pccs = np.array([[0.06, 0.07, 0.08], [0.065, 0.075, 0.085]])
        coupon = 0.08
        smm_dates = pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01'])
        
        # Expected SMM with 2-month lag
        # Note that demographic factors are not lagged
        expected_smms = np.array([
            [0.04625, 0.0467, 0.04715],
            [0.04625, 0.0467, 0.04715]
        ])

        smms = calculate_smms(pccs, coupon, smm_dates, lag_months=2)
        np.testing.assert_array_almost_equal(smms, expected_smms)

    def test_lag_out_of_bounds(self):
        """Test SMM calculation with an out of bounds lag on PCCs."""
        pccs = np.array([[0.06, 0.07, 0.08], [0.065, 0.075, 0.085]])
        coupon = 0.08
        smm_dates = pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01'])

        with self.assertRaises(IndexError):
            calculate_smms(pccs, coupon, smm_dates, lag_months=3)

    def test_refi_strength_edge_case(self):
        """Test edge cases for the refinancing strength calculation."""
        spreads = np.array([0, 0.015, 0.03])
        expected_strength = np.array([0.0, 0.0425, 0.0425])
        np.testing.assert_array_equal(refi_strength(spreads), expected_strength)

if __name__ == '__main__':
    unittest.main()
