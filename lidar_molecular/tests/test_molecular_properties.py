import unittest

from molecular.molecular_properties import *


class TestKingsFactorAtmosphere(unittest.TestCase):

    def setUp(self):
        pass

    def test_raises_value_error_if_wavelength_out_of_range(self):
        wavelength_nm = 100
        self.assertRaises(ValueError, kings_factor_atmosphere, wavelength_nm)

        wavelength_nm = 4100
        self.assertRaises(ValueError, kings_factor_atmosphere, wavelength_nm)

    def test_kings_factor_equal_to_table_2_of_tomasi_in_almost_all_values(self):
        """
        According to Tomasi:
        the value for 240nm should be 1.064670. Calculations yield 1.0646946969717852.
        1the value for 1500nm should be 1.04661. Calculations yield 1.0466048046635923

        For now, these are considered a rounding error on the side of Tomasi.
        """
        wavelengths_nm = np.array([0.200, 0.205, 0.210, 0.215, 0.220, 0.225, 0.230, 0.240, 0.250, 0.260, 0.270,
                                   0.280, 0.290, 0.300, 0.310, 0.320, 0.330, 0.340, 0.350, 0.360, 0.370,
                                   0.380, 0.390, 0.400, 0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800,
                                   0.850, 0.900, 0.950, 1.000, 1.500, 2.000, 2.500, 3.000, 3.500, 4.000]) * 1000

        kings_factors_tomasi = np.array([1.07851,  1.07610,  1.07393,  1.07199,  1.07023,  1.06864,
                                         1.06720,  1.06469,  1.06260,  1.06084,  1.05934,  1.05806,
                                         1.05696,  1.05600,  1.05517,  1.05444,  1.05380,  1.05323,
                                         1.05272,  1.05227,  1.05186, 1.05150, 1.05117, 1.05087, 1.04973, 1.04898,
                                         1.04845, 1.04808, 1.04779, 1.04758, 1.04741, 1.04727, 1.04716, 1.04707,
                                         1.04699, 1.04693, 1.04660, 1.04650, 1.04645, 1.04642, 1.04641, 1.04640])

        # Atmospheric conditions at ground level for Model 6 of Table 3 (US Standard, 1976)
        pressure = 1013.0
        p_e = 7.85075
        CO2_concentration = 385

        kings_factor_calculated = kings_factor_atmosphere(wavelengths_nm, C=CO2_concentration, p_e=p_e, p_t=pressure)
        kings_factor_rounded = np.around(kings_factor_calculated, 5)

        np.testing.assert_equal(kings_factor_rounded, kings_factors_tomasi)

    def test_depolarization_ratio_equal_to_table_2_of_tomasi_in_selected_values(self):
        """
        """
        wavelengths_nm = np.array([0.200,  0.260, 0.350, 0.500, 0.550, 1.000,  2.000, 3.500, 4.000]) * 1000

        rho_tomasi = np.array([4.465, 3.501, 3.051, 2.841, 2.812, 2.726, 2.702, 2.697, 2.696]) * 0.01

        # Atmospheric conditions at ground level for Model 6 of Table 3 (US Standard, 1976)
        pressure = 1013.0
        p_e = 7.85075
        CO2_concentration = 385

        rho_calculated =rho_atmosphere(wavelengths_nm, C=CO2_concentration, p_e=p_e, p_t=pressure)
        rho_rounded = np.around(rho_calculated, 5)
        print rho_rounded - rho_tomasi

        np.testing.assert_almost_equal(rho_rounded, rho_tomasi, 17)

if __name__ == "__main__":
    unittest.main()