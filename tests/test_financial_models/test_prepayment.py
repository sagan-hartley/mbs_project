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
        expected_pccs = np.array([0.06, 0.07, 0.08])  # [0.02 + 0.04, 0.03 + 0.04, 0.04 + 0.04]
        
        pccs = calculate_pccs(short_rates)
        np.testing.assert_array_equal(pccs, expected_pccs)

    def test_calculate_pccs_2d(self):
        """Test the calculation of Primary Current Coupons (PCCs) with 2D short rates."""
        short_rates = np.array([[0.02, 0.03, 0.04], [0.015, 0.025, 0.035]])
        spread = 0.03
        expected_pccs = np.array([[0.05, 0.06, 0.07], [0.045, 0.055, 0.065]])

        pccs = calculate_pccs(short_rates, spread)
        np.testing.assert_array_almost_equal(pccs, expected_pccs)

    def test_refi_strength(self):
        """Test the refinancing strength calculation."""
        spreads = np.array([-0.01, 0, 0.0075, 0.015, 0.02])
        expected_strength = np.array([0.0, 0.0, 0.02125, 0.0425, 0.0425])
        np.testing.assert_array_equal(refi_strength(spreads), expected_strength)

    def test_refi_strength_edge_case(self):
        """Test edge cases for the refinancing strength calculation."""
        spreads = np.array([0, 0.015, 0.03])
        expected_strength = np.array([0.0, 0.0425, 0.0425])
        np.testing.assert_array_equal(refi_strength(spreads), expected_strength)

    def test_demo(self):
        """Test the demographic factor calculation."""
        smm_dates = pd.to_datetime([
            '2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-05-01', 
            '2024-06-01', '2024-07-01', '2024-08-01', '2024-09-01', '2024-10-01', 
            '2024-11-01', '2024-12-01'
        ])
        base_smm = 0.005
        expected_demo_factors = np.array([0.0, 0.000233, 0.000517, 0.00085, 0.001233, 0.001667, 0.002,
                                          0.002158, 0.002267, 0.002325, 0.002333, 0.002292])
        np.testing.assert_array_almost_equal(demo(smm_dates, base_smm), expected_demo_factors)

    def test_demo_invalid_smm_dates(self):
        """Test the invalid case where smm_dates is not a regular monthly grid"""
        invalid_smm_dates = pd.to_datetime([
            '2024-01-01', '2024-02-02', '2024-03-01'
        ])
        base_smm = 0.005

        with self.assertRaises(ValueError):
            demo(invalid_smm_dates, base_smm)

    def test_calculate_smms(self):
        """Test the Single Monthly Mortality (SMM) calculation."""
        pccs = np.array([[0.06, 0.07, 0.08], [0.065, 0.075, 0.085]])
        coupon = 0.08
        smm_dates = pd.to_datetime([
            '2024-01-01', '2024-02-01', '2024-03-01'
        ])
        pcc_dates = smm_dates
        
        expected_smms = np.array([
            [0.0425, 0.028567, 0.000517],
            [0.0425, 0.0144, 0.000517]
        ])

        smms = calculate_smms(pccs, pcc_dates, smm_dates, coupon)
        np.testing.assert_array_almost_equal(smms, expected_smms)

    def test_calculate_smms_invalid_pccs_length(self):
        """Test the invalid case where the length of PCCs doesn't match the number of dates in smm_dates."""
        pccs = np.array([[0.06, 0.07], [0.065, 0.075]])  # Only 2 months
        coupon = 0.08
        smm_dates = pd.to_datetime([
            '2024-01-01', '2024-02-01', '2024-03-01'  # 3 months
        ])
        pcc_dates = smm_dates
        
        with self.assertRaises(ValueError):
            calculate_smms(pccs, pcc_dates, smm_dates, coupon)

    def test_calculate_smms_3d_pccs(self):
        """Test the invalid case where PCCs is not 1 or 2-dimensional"""
        three_d_pccs = np.array([[[0.06, 0.07, 0.08], [0.065, 0.075, 0.08]]])
        coupon = 0.08
        smm_dates = pd.to_datetime([
            '2024-01-01', '2024-02-01', '2024-03-01'
        ])
        pcc_dates = smm_dates
        
        with self.assertRaises(ValueError):
            calculate_smms(three_d_pccs, pcc_dates, smm_dates, coupon)

    def test_calculate_smms_1d_pccs(self):
        """Test the case where PCCs is a 1d array"""
        one_d_pccs = np.array([0.06, 0.07, 0.08])
        coupon = 0.08
        smm_dates = pd.to_datetime([
            '2024-01-01', '2024-02-01', '2024-03-01'
        ])
        pcc_dates = smm_dates
        
        expected_smms = np.array([0.0425, 0.028567, 0.000517])

        smms = calculate_smms(one_d_pccs, pcc_dates, smm_dates, coupon)
        np.testing.assert_array_almost_equal(smms, expected_smms)

    def test_calculate_smms_with_lag_one_month(self):
        """Test SMM calculation with a 1-month lag on PCCs."""
        pccs = np.array([[0.06, 0.07, 0.08], [0.065, 0.075, 0.085]])
        coupon = 0.08
        smm_dates = pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01'])
        pcc_dates = smm_dates
        
        # Expected SMM with 1-month lag
        # Note that demographic factors are not lagged
        expected_smms = np.array([
            [0.0425, 0.042733, 0.02885],
            [0.0425, 0.042733, 0.014683]
        ])

        smms = calculate_smms(pccs, pcc_dates, smm_dates, coupon, lag_months=1)
        np.testing.assert_array_almost_equal(smms, expected_smms)

    def test_calculate_smms_with_lag_two_months(self):
        """Test SMM calculation with a 2-month lag on PCCs."""
        pccs = np.array([[0.06, 0.07, 0.08], [0.065, 0.075, 0.085]])
        coupon = 0.08
        smm_dates = pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01'])
        pcc_dates = smm_dates
        
        # Expected SMM with 2-month lag
        # Note that demographic factors are not lagged
        expected_smms = np.array([
            [0.0425, 0.042733, 0.043017],
            [0.0425, 0.042733, 0.043017]
        ])

        smms = calculate_smms(pccs, pcc_dates, smm_dates, coupon, lag_months=2)
        np.testing.assert_array_almost_equal(smms, expected_smms)

    def test_lag_out_of_bounds(self):
        """Test SMM calculation with an out of bounds lag on PCCs."""
        pccs = np.array([[0.06, 0.07, 0.08], [0.065, 0.075, 0.085]])
        coupon = 0.08
        smm_dates = pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01'])
        pcc_dates = smm_dates

        with self.assertRaises(IndexError):
            calculate_smms(pccs, pcc_dates, smm_dates, coupon, lag_months=3)

    def test_calculate_smms_1d_with_lag(self):
        """Test SMM calculation with 1D PCCs and a non-zero lag."""
        one_d_pccs = np.array([0.06, 0.07, 0.08])
        coupon = 0.08
        smm_dates = pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01'])
        pcc_dates = smm_dates
        
        # Expected SMM with a 1-month lag
        expected_smms = np.array([0.0425, 0.042733, 0.02885])

        smms = calculate_smms(one_d_pccs, pcc_dates, smm_dates, coupon, lag_months=1)
        np.testing.assert_array_almost_equal(smms, expected_smms)

    def test_calculate_smms_with_date_sorting(self):
        """Test SMM calculation where the pcc and smm dates differ."""
        pccs = np.array([[0.06, 0.065, 0.07, 0.075, 0.08], [0.065, 0.07, 0.075, 0.08, 0.085]])
        coupon = 0.08
        smm_dates = pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01'])
        pcc_dates = pd.to_datetime(['2024-01-01', '2024-01-15', '2024-02-01', '2024-02-15', '2024-03-01'])
        
        # Expected SMM with 1-month lag
        # Note that demographic factors are not lagged
        expected_smms = np.array([
            [0.0425, 0.042733, 0.02885],
            [0.0425, 0.042733, 0.014683]
        ])

        smms = calculate_smms(pccs, pcc_dates, smm_dates, coupon, lag_months=1)
        np.testing.assert_array_almost_equal(smms, expected_smms)

if __name__ == '__main__':
    unittest.main()
