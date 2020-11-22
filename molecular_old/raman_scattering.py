''' This module calculates the (rotational) Raman scattering induced in N2 and O2 molecules in
the atmosphere. It is an application of the method found in:

A. Behrendt and T. Nakamura, "Calculation of the calibration constant of polarization lidar 
and its dependency on atmospheric temperature," Opt. Express, vol. 10, no. 16, pp. 805-817, 2002.
 
The molecular parameters gamma and epsilon are wavelength dependent, and this 
makes the results of the original paper valid only for 532nm. Some new formulas
have been implemented from:

Tomasi, C., Vitale, V., Petkov, B., Lupi, A. & Cacciari, A. Improved 
algorithm for calculations of Rayleigh-scattering optical depth in standard 
atmospheres. Applied Optics 44, 3320 (2005).

and

Chance, K. V. & Spurr, R. J. D. Ring effect studies: Rayleigh scattering, 
including molecular parameters for rotational Raman scattering, and the 
Fraunhofer spectrum. Applied Optics 36, 5224 (1997).

It is not thoroughly tested, so use with care.
'''
import numpy as np
from matplotlib import pyplot as plt

import us_std
from constants import hc_k, hc, k_b
from molecular_properties import gamma_square_N2, gamma_square_O2, epsilon_N2, epsilon_O2

N2_parameters = {'name': "N_{2}",
                 'B0': 1.989500,
                 'D0': 5.48E-6,
                 'I': 1,
                 'gamma_square': gamma_square_N2,  # Originally 0.509E-48,
                 'epsilon': epsilon_N2,  # Originally 0.161,
                 'g': [6, 3],
                 'relative_concentration': 0.79}  # for even/odd J

O2_parameters = {'name': "O_{2}",
                 'B0': 1.437682,
                 'D0': 4.85E-6,
                 'I': 0,
                 'gamma_square': gamma_square_O2,  # Originally 1.27E-48,
                 'epsilon': epsilon_O2,  # Originally 0.467,
                 'g': [0, 1],
                 'relative_concentration': 0.21}  # for even/odd J


def rotational_energy(J, molecular_parameters):
    """ Calculates the rotational energy of a homonuclear diatomic molecule for
    quantum number J. The molecule is specified by passing a dictionary with
    parameters.

    Parameters
    ----------
    J : int
       Rotational quantum number.
    molecular_parameters : dict
       A dictionary containing molecular parameters (specifically, B0 and D0).

    Returns
    -------
    E_rot : float
       Rotational energy of the molecule (J)
    """

    B0 = molecular_parameters['B0']
    D0 = molecular_parameters['D0']

    E_rot = (B0 * J * (J + 1) - D0 * J ** 2 * (J + 1) ** 2) * hc
    return E_rot


def raman_shift_stokes(J, molecular_parameters):
    """ Calculates the rotational Raman shift  (delta en) for the Stokes branch for
    quantum number J.

    Parameters
    ----------
    J : int
       Rotational quantum number
    molecular_parameters : dict
       A dictionary containing molecular parameters (specifically, B0 and D0)

    Returns
    -------
    delta_n: float
       Rotational Raman shift [cm-1]
    """

    B0 = molecular_parameters['B0']
    D0 = molecular_parameters['D0']

    delta_n = -B0 * 2 * (2 * J + 3) + D0 * (3 * (2 * J + 3) + (2 * J + 3) ** 3)
    return delta_n


def raman_shift_antistokes(J, molecular_parameters):
    """ Calculates the rotational Raman shift (delta en) for the anti-Stokes branch for
    quantum number J.

    Parameters
    ----------
    J: int
       Rotational quantum number
    molecular_parameters: dict
       A dictionary containing molecular parameters (specifically, B0 and D0)

    Returns
    -------
    delta_n: float
       Rotational Raman shift [cm-1]
    """
    B0 = molecular_parameters['B0']
    D0 = molecular_parameters['D0']

    delta_n = B0 * 2 * (2 * J - 1) - D0 * (3 * (2 * J - 1) + (2 * J - 1) ** 3)
    return delta_n


def cross_section_stokes(n_incident, J, temperature, molecular_parameters):
    """ Calculates the rotational Raman backsattering cross section for the Stokes
    branch for quantum number J at a temperature T.

    Parameters
    ----------
    n_incident : float
       Wavenumber of incident light [cm-1]
    J : int
       Rotational quantum number
    temperature : float
       The ambient temperature [K]
    molecular_parameters : dict
       A dictionary containing molecular parameters.

    Returns
    -------
    b_s : float
       Scattering cross section [cm^{2}sr^{-1}]
    """
    B0 = molecular_parameters['B0']

    # Check if callable or just a number
    gamma_square_input = molecular_parameters['gamma_square']

    if callable(gamma_square_input):
        gamma_square = gamma_square_input(n_incident)
    else:
        gamma_square = gamma_square_input  # Assume a float is provided

    g_index = np.remainder(J, 2)
    g = molecular_parameters['g'][g_index]

    J = float(J)
    b_s = 64 * np.pi ** 4 * hc_k / 15
    b_s *= g * B0 * (n_incident + raman_shift_stokes(J, molecular_parameters)) ** 4 * gamma_square
    b_s /= (2 * molecular_parameters['I'] + 1) ** 2 * temperature
    b_s *= (J + 1) * (J + 2) / (2 * J + 3)
    b_s *= np.exp(-rotational_energy(J, molecular_parameters) / (k_b * temperature))
    return b_s


def cross_section_antistokes(n_incident, J, temperature, molecular_parameters):
    """ Calculates the rotational Raman backsattering cross section for the Stokes
    branch for quantum number J at a temperature T.

    Parameters
    ----------
    n_incident : float
       Wavelnumber of incident light [cm-1]
    J : int
       Rotational quantum number
    temperature : float
       The ambient temperature [K]
    molecular_parameters : dict
       A dictionary containing molecular parameters.

    Returns
    -------
    b_s : float
       Scattering cross section [cm^{2}sr^{-1}]
    """
    B0 = molecular_parameters['B0']

    # Check if callable or just a number
    gamma_square_input = molecular_parameters['gamma_square']

    if callable(gamma_square_input):
        gamma_square = gamma_square_input(n_incident)
    else:
        gamma_square = gamma_square_input  # Assume a float is provided

    g_index = np.remainder(J, 2)
    g = molecular_parameters['g'][g_index]

    J = float(J)

    b_s = 64 * np.pi ** 4 * hc_k / 15.

    b_s *= g * B0 * (n_incident + raman_shift_antistokes(J, molecular_parameters)) ** 4 * gamma_square
    b_s /= (2 * molecular_parameters['I'] + 1) ** 2 * temperature
    b_s *= J * (J - 1) / (2 * J - 1)
    b_s *= np.exp(-rotational_energy(J, molecular_parameters) / (k_b * temperature))
    return b_s


def delta_mol(n_incident, molecular_parameters, relative_transmissions):
    """ Calculates the depolarization ratio of the molecular signal detected by a lidar.

    Parameters
    ----------
    n_incident : float
       Wavenumber of the incident wave (cm-1)
    molecular_parameters : list
       A list of dictionaries that describe the molecules.
    relative_transmissions : list
       The relative fraction of the intensity of the rotational Raman wings of the molecules detected by the lidar.

    Returns
    -------
    delta_m : float
       The apparent molecular depolarization ratio.
    """
    concentrations = [parameter['relative_concentration'] for parameter in molecular_parameters]

    # Check if callable or just a number
    gamma_squares = []
    for parameter in molecular_parameters:
        gamma_square_input = parameter['gamma_square']

        if callable(gamma_square_input):
            gamma_square = gamma_square_input(n_incident)
        else:
            gamma_square = gamma_square_input  # Assume a float is provided

        gamma_squares.append(gamma_square)

    epsilons = []
    for parameter in molecular_parameters:
        epsilon_input = parameter['epsilon']

        if callable(epsilon_input):
            epsilon = epsilon_input(n_incident)
        else:
            epsilon = epsilon_input  # Assume a float is provided

        epsilons.append(epsilon)

    numerator = np.sum([con * gamma_square * (3 * x + 1) for (con, gamma_square, x)
                        in zip(concentrations, gamma_squares, relative_transmissions)])
    denominator = np.sum([con * gamma_square * (3 * x + 1 + 45 / epsilon) for (con, gamma_square, x, epsilon)
                          in zip(concentrations, gamma_squares, relative_transmissions, epsilons)])
    delta_m = 3.0 / 4 * numerator / denominator
    return delta_m


class DepolarizationLidar:

    def __init__(self, wavelength=532.0, fwhm=0.55):
        """
        This class calculates the volume depolarization ratio of the molecular
        backscatter signal detected with a polarization lidar.


        Parameters
        ----------
        wavelength: float
           The lidar emission wavelength (nm)
        fhwm : float
           The full-widht at half-maximum of the detecting filter

        Available methods:

        delta_mol_temperature(T) - calculate the molecular depolarization ratio at temperature T (in Kelvin)

        """
        self.J_stokes = np.arange(0, 40)
        self.J_astokes = np.arange(2, 40)

        self.wavelength = float(wavelength)
        self.wavenumber = 10 ** 7 / self.wavelength

        self.optical_filter = FilterFunction(self.wavelength, fwhm)

        self.dn_stokes_N2 = np.array(
            [raman_shift_stokes(J, N2_parameters) for J in self.J_stokes])
        self.dn_astokes_N2 = np.array(
            [raman_shift_antistokes(J, N2_parameters) for J in self.J_astokes])

        self.dl_stokes_N2 = 1 / \
                            (1 / self.wavelength + np.array(self.dn_stokes_N2) * 10 ** -7)
        self.dl_astokes_N2 = 1 / \
                             (1 / self.wavelength + np.array(self.dn_astokes_N2) * 10 ** -7)

        self.dn_stokes_O2 = [
            raman_shift_stokes(J, O2_parameters) for J in self.J_stokes]
        self.dn_astokes_O2 = [
            raman_shift_antistokes(J, O2_parameters) for J in self.J_astokes]

        self.dl_stokes_O2 = 1 / \
                            (1 / self.wavelength + np.array(self.dn_stokes_O2) * 10 ** -7)
        self.dl_astokes_O2 = 1 / \
                             (1 / self.wavelength + np.array(self.dn_astokes_O2) * 10 ** -7)

    def delta_mol_temperature(self, T):
        x_N2, x_O2 = self.rotation_contribution_temperature(T)

        delta_m = delta_mol(self.wavenumber, [N2_parameters, O2_parameters], [x_N2, x_O2])

        return delta_m, x_N2, x_O2

    def rotation_contribution_temperature(self, T):
        ds_stokes = [cross_section_stokes(
            self.wavenumber, J, T, N2_parameters) for J in self.J_stokes]
        ds_astokes = [cross_section_antistokes(
            self.wavenumber, J, T, N2_parameters) for J in self.J_astokes]

        x_N2 = (np.sum(self.optical_filter(self.dl_stokes_N2) * ds_stokes) +
                np.sum(self.optical_filter(self.dl_astokes_N2) * ds_astokes)) / (np.sum(ds_stokes) + np.sum(ds_astokes))

        ds_stokes = [cross_section_stokes(
            self.wavenumber, J, T, O2_parameters) for J in self.J_stokes]
        ds_astokes = [cross_section_antistokes(
            self.wavenumber, J, T, O2_parameters) for J in self.J_astokes]

        x_O2 = (np.sum(self.optical_filter(self.dl_stokes_O2) * ds_stokes) +
                np.sum(self.optical_filter(self.dl_astokes_O2) * ds_astokes)) / (np.sum(ds_stokes) + np.sum(ds_astokes))
        return x_N2, x_O2

    def delta_mol_at_altitude(self, altitudes):
        ''' Calculates delta_mol at an altitude (in m)
        '''

        atmosphere = us_std.Atmosphere()
        Ts = np.array([atmosphere.temperature(altitude)
                       for altitude in altitudes])

        delta_mols = []

        for T in Ts:
            delta_m, _, _ = self.delta_mol_temperature(T)
            delta_mols.append(delta_m)

        return delta_mols

    def plot_spectrum(self, T, molecular_parameters, figsize=(10, 5)):
        ds_stokes = [cross_section_stokes(
            self.wavenumber, J, T, molecular_parameters) for J in self.J_stokes]
        ds_astokes = [cross_section_antistokes(
            self.wavenumber, J, T, molecular_parameters) for J in self.J_astokes]

        dn_stokes = np.array(
            [raman_shift_stokes(J, molecular_parameters) for J in self.J_stokes])
        dn_astokes = np.array(
            [raman_shift_antistokes(J, molecular_parameters) for J in self.J_astokes])

        dl_stokes = 1 / (1 / self.wavelength + np.array(dn_stokes) * 10 ** -7)
        dl_astokes = 1 / \
                     (1 / self.wavelength + np.array(dn_astokes) * 10 ** -7)

        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)

        x_step = np.diff(dl_stokes)[0]
        bar_width = x_step / 2.

        bar1 = ax1.bar(dl_stokes, ds_stokes, width=bar_width, color='blue', label='Full')
        ax1.bar(dl_astokes, ds_astokes, width=bar_width, color='blue')
        bar2 = ax1.bar(dl_stokes, self.optical_filter(dl_stokes)
                       * ds_stokes, width=bar_width, color='green', alpha=1, label='Filtered')
        ax1.bar(dl_astokes, self.optical_filter(dl_astokes)
                * ds_astokes, width=bar_width, color='green', alpha=1)
        # ax1.bar(dl_astokes, ds_astokes, width = 0.1, color = 'blue')

        ax1.set_xlabel('Wavelength [nm])')
        ax1.set_ylabel(r'$\left( \frac{d\sigma}{d\omega}\right)_{\pi}$ [$cm^{2}sr^{-1}$]')
        ax1.grid(True)

        ax2 = ax1.twinx()

        xmin, xmax = ax1.get_xlim()

        filter_wavelengths = np.linspace(xmin, xmax, 1000)
        filter_efficiency = self.optical_filter(filter_wavelengths)

        filter_label = 'Filter fwhm %s nm' % self.optical_filter.fwhm
        line_1, = ax2.plot(filter_wavelengths, filter_efficiency, '-g',
                           label=filter_label)
        ax2.set_ylabel('Filter efficiency')
        ax2.set_ylim(0, 1)
        
        ax2.yaxis.label.set_color('green')
        ax2.tick_params(axis='y', colors='green')

        fig.suptitle('Rotational raman spectrumn of $%s$' %
                     molecular_parameters['name'])

        ax2.legend([bar1[0], bar2[0], line_1], ['Full', 'Filtered', filter_label], loc=1)
        plt.draw()
        plt.show()


class FilterFunction:
    def __init__(self, wavelength, fwhm):
        '''
        This simple class represents a gausian filter function. To generate 
        a new filter use::

           my_filter = FilterFunction(wavelegnth, fwhm)

        with

        wavelegnth - The central wavelength of the filter in nm
        fwhm       - The fwhm of the filter in nm

        If the the filter is called with a wavelegnth (in nm) as an argument 
        it will return the  efficiency at this wavelength, for example::

           my_filter = FilterFunction(532, 5)
           my_filter(532) # Will return 1.0
           my_filter(535) # Will return 0.3685
        '''
        self.wavelength = wavelength
        self.fwhm = fwhm
        self.c = fwhm / 2.354820045031

    def __call__(self, wavelength):
        value = np.exp(-(wavelength - self.wavelength)
                        ** 2 / (2 * self.c ** 2))
        return value
