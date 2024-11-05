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
    Test suite for the prepaymetn model functions including PCC calculations,
    refinancing strength, demographic factors, and SMM calculations.
    """

    def test_calculate_pccs(self):
        """Test the calculation of Primary Current Coupons (PCCs)."""
        short_rates = np.array([0.02, 0.03, 0.04])
        expected_pccs = np.array([0.06, 0.07, 0.08])  # 0.02 + 0.04, 0.03 + 0.04, 0.04 + 0.04
        np.testing.assert_array_equal(calculate_pccs(short_rates), expected_pccs)

    def test_refi_strength(self):
        """Test the refinancing strength calculation."""
        spreads = np.array([-0.01, 0, 0.0075, 0.015, 0.02])
        expected_strength = np.array([0.0, 0.0, 0.02125, 0.0425, 0.0425])
        np.testing.assert_array_equal(refi_strength(spreads), expected_strength)

    def test_demo(self):
        """Test the demographic factor calculation."""
        origination_date = pd.Timestamp('2024-01-01')
        num_months = 12
        base_smm = 0.005
        expected_demo_factors = np.array([0.00375, 0.0042, 0.00465, 0.0051, 0.00555, 0.006, 0.006,
                                          0.00555, 0.0051, 0.00465, 0.0042 ,0.00375])
        np.testing.assert_array_almost_equal(demo(origination_date, num_months, base_smm), expected_demo_factors)

    def test_calculate_smms(self):
        """Test the Single Monthly Mortality (SMM) calculation."""
        pccs = np.array([[0.06, 0.07, 0.08], [0.065, 0.075, 0.085]])
        coupon = 0.04
        market_close_date = pd.Timestamp('2024-11-05')
        origination_date = pd.Timestamp('2024-12-01')
        num_months = 3
        
        expected_smms = np.array([
            [0.00375, 0.00375, 0.0042 ],
            [0.00375, 0.00375, 0.0042 ]
        ])

        smms = calculate_smms(pccs, coupon, market_close_date, origination_date, num_months)
        np.testing.assert_array_almost_equal(smms, expected_smms)

    def test_calculate_smms_invalid_pccs_length(self):
        """Test invalid length of PCCs raises ValueError."""
        pccs = np.array([[0.06]])
        coupon = 0.04
        market_close_date = pd.Timestamp('2024-11-05')
        origination_date = pd.Timestamp('2024-01-01')
        num_months = 3

        with self.assertRaises(ValueError):
            calculate_smms(pccs, coupon, market_close_date, origination_date, num_months)

if __name__ == "__main__":
    unittest.main()
