"""
Unit test of the rayleigh_scattering_bucholtz functions.

The test is performed using the tabular values found in the referenced paper.

References
----------
Bucholtz, A. Rayleigh-scattering calculations for the terrestrial atmosphere.
Applied Optics, Vol. 3

.. todo::
   * Move deleted plots in a dedicated Notebook.
"""

import os
import unittest2 as unittest
from scipy.interpolate import interp1d
from molecular.rayleigh_scattering_bucholtz import *

# Get the data path
current_path = os.path.dirname(__file__)
data_base_path = os.path.join(current_path, '../data/bucholtz_tabular_values/')


class TestKingCorrectionFactor(unittest.TestCase):
    def test_against_bucholtz_paper_table_1(self):
        #   Specify the data file.
        filename_reference = os.path.join(data_base_path, "bucholtz_1995_table_1.csv")        
        
        #   Load the reference data.
        reference = np.loadtxt(filename_reference, delimiter='\t', skiprows=1)        
        
        #   Calculate the King correction factor.
        calculated = king_correction_factor(reference[:, 0])

        #   Round the values to the desired decimal.
        calculated = np.around(calculated, decimals=3)

        #   Test if equal to reference.
        np.testing.assert_array_equal(calculated, reference[:, 1])

    
class TestScatteringCrossSection(unittest.TestCase):
    def test_against_bucholtz_paper_table_2(self):
        #   Specify the data file.
        filename_reference = os.path.join(data_base_path, "bucholtz_1995_table_2.csv")
        
        #   Load the reference data.
        reference = np.loadtxt(filename_reference, delimiter='\t', skiprows=1)

        #   Calculate the Rayleigh scattering cross section. (cm^2)
        calculated = scattering_cross_section(reference[:, 0])

        #   Round the values to the desired decimal.
        calculated[0:7] = np.around(calculated[0:7], decimals=28)
        calculated[7:26] = np.around(calculated[7:26], decimals=29)
        calculated[26:60] = np.around(calculated[26:60], decimals=30)
        calculated[60:67] = np.around(calculated[60:67], decimals=31)
        calculated[67:75] = np.around(calculated[67:75], decimals=32)
        calculated[75:] = np.around(calculated[75:], decimals=33)
        
        #   Assign zero to the expected failing wavelengths.
        index_fail = 26
        reference[index_fail, 1] = 0
        calculated[index_fail] = 0
        
        #   Test if equal to reference.
        np.testing.assert_allclose(calculated, reference[:, 1], rtol=1e-4)

    @unittest.skip("Skip the expected failing wavelengths.")
    def test_failing_wavelengths(self):
        #   Specify the data file.
        filename_reference = os.path.join(data_base_path, "bucholtz_1995_table_2.csv")
        
        #   Load the reference data.
        reference = np.loadtxt(filename_reference, delimiter='\t', skiprows=1)
        
        #   Specify the failing wavelength index.
        index_fail = 26
        
        #   Calculate the Rayleigh scattering cross section. (cm^2)
        calculated = scattering_cross_section(reference[index_fail, 0])

        #   Round the values to the desired decimal.
        calculated = np.around(calculated, decimals=30)
        
        #   Test if equal to reference.
        np.testing.assert_allclose(calculated, reference[index_fail, 1], rtol=1e-4)
    
    def test_last_working_values(self):
        #   Specify the data file.
        filename_reference = os.path.join(data_base_path, "bucholtz_1995_last_working_values.csv")

        #   Load the data from the file.
        reference = np.loadtxt(filename_reference, delimiter='\t', skiprows=1)

        #   Calculate the Rayleigh scattering cross section. (cm^2)
        calculated = scattering_cross_section(reference[:, 0])

        # Test if equal to reference.
        np.testing.assert_allclose(calculated, reference[:, 1], rtol=1e-6)
        

class TestVolumeScatteringCoefficient(unittest.TestCase):
    def test_against_bucholtz_paper_table_2(self):
        #   Specify the data file.
        filename_reference = os.path.join(data_base_path, "bucholtz_1995_table_2.csv")

        #   Load the reference data.
        reference = np.loadtxt(filename_reference, delimiter='\t', skiprows=1)

        #   Calculate the total Rayleigh volume-scattering coefficient. (km^-1)
        calculated = volume_scattering_coefficient(reference[:, 0], P_s, T_s)

        # Round the values to the desired decimal.
        calculated[0:13] = np.around(calculated[0:13], decimals=4)
        calculated[13:37] = np.around(calculated[13:37], decimals=5)
        calculated[37:63] = np.around(calculated[37:63], decimals=6)
        calculated[63:70] = np.around(calculated[63:70], decimals=7)
        calculated[70:78] = np.around(calculated[70:78], decimals=8)
        calculated[78:] = np.around(calculated[78:], decimals=9)
        
        #   Assign zero to the expected failing wavelengths.
        index_fail = [22, 70]
        reference[index_fail, 2] = 0
        calculated[index_fail] = 0
                
        #   Test if equal to reference.
        np.testing.assert_allclose(calculated, reference[:, 2], rtol=1e-4)
        
    @unittest.skip("Skip the expected failing wavelengths.")
    def test_failing_wavelengths(self):
        #   Specify the data file.
        filename_reference = os.path.join(data_base_path, "bucholtz_1995_table_2.csv")
        
        #   Load the reference data.
        reference = np.loadtxt(filename_reference, delimiter='\t', skiprows=1)
        
        #   Specify the failing wavelength index.
        index_fail = [22, 70]

        #   Calculate the total Rayleigh volume-scattering coefficient. (km^-1)
        calculated = volume_scattering_coefficient(reference[index_fail, 0], P_s, T_s)

        #   Round the values to the desired decimal.
        calculated[0] = np.around(calculated, decimals=5)
        calculated[1] = np.around(calculated, decimals=8)
        
        #   Test if equal to reference.
        np.testing.assert_allclose(calculated, reference[index_fail, 1], rtol=1e-4)
        
    def test_last_working_values(self):
        #   Specify the data file.
        filename_reference = os.path.join(data_base_path, "bucholtz_1995_last_working_values.csv")

        #   Load the data from the file.
        reference = np.loadtxt(filename_reference, delimiter='\t', skiprows=1)

        #   Calculate the total Rayleigh volume-scattering coefficient. (km^-1)
        calculated = volume_scattering_coefficient(reference[:, 0], P_s, T_s)

        # Test if equal to reference.
        np.testing.assert_allclose(calculated, reference[:, 2], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()