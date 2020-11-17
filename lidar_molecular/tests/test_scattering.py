import unittest

from molecular.rayleigh_scattering_bucholtz import *


class BucholtzTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_number_density_in_STP(self):
        temperature = 288.15  # in K
        pressure = 1013.25  # in hPa
        density_stp = number_density_at_tp(temperature, pressure)
        self.assertAlmostEqual(density_stp, 2.546917378744801e+25, 12)

    def test_depolariation(self):
        rho_200 = depolarization_factor(200)
        self.assertEqual(rho_200, 4.545 * 1e-2)

        rho_350 = depolarization_factor(350)
        self.assertEqual(rho_350, 3.010 * 1e-2)

        rho_1000 = depolarization_factor(1000)
        self.assertEqual(rho_1000, 2.730 * 1e-2)

    def test_kigs_factor(self):
        fk_200 = kings_factor(200)
        self.assertEqual(round(fk_200, 3), 1.080)

        fk_350 = kings_factor(350)
        self.assertEqual(round(fk_350, 3), 1.052)

        fk_1000 = kings_factor(1000)
        self.assertEqual(round(fk_1000, 3), 1.047)

    def test_scattering_cross_section(self):
        sigma_200 = scattering_cross_section(200)
        #self.assertEqual(round(sigma_200 * 1e25, 3), 3.612)
        # Check the refractive index for wavelengths.

        sigma_350 = scattering_cross_section(350)
        self.assertEqual(round(sigma_350 * 1e26, 3), 2.924)

        #sigma_4000 =scattering_cross_section(4000)
        #self.assertEqual(round(sigma_4000 * 1e30, 3), 1.550)


if __name__ == "__main__":
    unittest.main()