''' This module calculates the volume depolarization ratio of the molecular backscatter signal
dected with a polarization lidar. It is an application of the method found in:

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


For 532nm the results are similare (but not identical)
with the orignal.


Its use::

   # Create a new DepolarizationLidar instance
   my_lidar = DepolarizationLidar(532, 0.7) # Laser wavelegnth, fwhm of the filter
   # Calculate the molecular volume depolarization at temperature T (in deg Kelvin)
   delta_mol = my_lidar.delta_mol_temperature(240) # will return 0.003976

Any comments should be sent to ioannis@inoe.ro

'''
import numpy as np
from matplotlib import pyplot as plt

import us_std

from molecular_properties import gamma_square_N2, gamma_square_O2, epsilon_N2, epsilon_O2
from constants import hc_k


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


def energy_rotational_norm(J, molecular_parameters):
    ''' Calculates the roational energy of a homonuclear diatomic molecule for 
    quantum number J. The molecule is specified by passing a dictionary with 
    parameters.

    The result is divided by the Boltzmann constant for ease of calculation.
    '''

    B0 = molecular_parameters['B0']
    D0 = molecular_parameters['D0']

    E_rot = (B0 * J * (J + 1) - D0 * (J ** 2) * ((J + 1) ** 2)) * hc_k
    return E_rot


def delta_n_stokes(J, molecular_parameters):
    '''Calculates the rotational raman shift for the Stokes branch for
    quantum number J. The molecule is specified by passing a dictionary with
    parameters.
    
    Output in cm-1
    '''

    B0 = molecular_parameters['B0']
    D0 = molecular_parameters['D0']

    delta_n = -B0 * 2 * (2 * J + 3) + D0 * (3 * (2 * J + 3) + (2 * J + 3) ** 3)
    return delta_n


def delta_n_antistokes(J, molecular_parameters):
    '''Calculates the rotational raman shift for the anti-Stokes branch for
    quantum number J. The moelcule is specified by passing a dictionary with 
    parameters.
    
    Output in cm-1
    '''
    B0 = molecular_parameters['B0']
    D0 = molecular_parameters['D0']

    delta_n = B0 * 2 * (2 * J - 1) - D0 * (3 * (2 * J - 1) + (2 * J - 1) ** 3)
    return delta_n


def delta_sigma_stokes(n_incident, J, temperature, molecular_parameters):
    '''Calculates the backscatter coefficient of the Raman scattering for the Stokes 
    branch for quantum number J at a temperature T (in Kelvin ). The molecule is specified
    by passing a dictionary with parameters.
    '''

    B0 = molecular_parameters['B0']
    gamma_square_func = molecular_parameters['gamma_square']
    gamma_square = gamma_square_func(n_incident)
    
    g_index = np.remainder(J, 2)
    g = molecular_parameters['g'][g_index]

    J = float(J)

    b_s = 64 * np.pi ** 4 * hc_k / 15.

    b_s = b_s * (g * B0 * (n_incident +
                           delta_n_stokes(J, molecular_parameters)) ** 4 * gamma_square)
    b_s = b_s / ((2 * molecular_parameters['I'] + 1) ** 2 * temperature)
    b_s = b_s * ((J + 1) * (J + 2)) / (2 * J + 3.)
    b_s = b_s * \
        np.exp(-energy_rotational_norm(J, molecular_parameters) / temperature)
    return b_s


def delta_sigma_antistokes(n_incident, J, temperature, molecular_parameters):
    '''Calculates the backscatter coefficient of the Raman scattering for the Stokes 
    branch for quantum number J at a temperature T (in Kelvin ). The moelcule is specified 
    by passing a dictionary with parameters.
    
    In cm^2sr-1.
    '''

    B0 = molecular_parameters['B0']
    gamma_square_func = molecular_parameters['gamma_square']
    gamma_square = gamma_square_func(n_incident)

    g_index = np.remainder(J, 2)
    g = molecular_parameters['g'][g_index]

    J = float(J)

    b_s = 64 * np.pi ** 4 * hc_k / 15.

    b_s = b_s * (g * B0 * (n_incident +
                           delta_n_antistokes(J, molecular_parameters)) ** 4 * gamma_square)
    b_s = b_s / ((2 * molecular_parameters['I'] + 1) ** 2 * temperature)
    b_s = b_s * (J * (J - 1)) / (2 * J - 1.)
    b_s = b_s * \
        np.exp(-energy_rotational_norm(J, molecular_parameters) / temperature)
    return b_s


def delta_mol(n_incident, concentrations, molecular_parameters, relative_transmisions):
    """ Calculates the depolarization ratio of the molecular signal detected by a lidar.
    
    It's inputs are:

    concentrations         - a list of the relative concentration of the moleculres ( [0.79, 0.21] -
                             for the Nitrogen and Oxygen in the atmosphere)
    molecular_parameters   - a list of dictionaries that describe the molecules.
    relative_transimssions - the ralative part of the intensity of the rotational Raman wings of the 
                             molecules detected by the lidar.
    """

    gamma_squares = [parameter['gamma_square'](n_incident)
                     for parameter in molecular_parameters]
    epsilons = [parameter['epsilon'](n_incident) for parameter in molecular_parameters]

    numerator = np.sum([con * gamma_square * (3 * x + 1) for (con, gamma_square, x)
                        in zip(concentrations, gamma_squares, relative_transmisions)])
    denominator = np.sum([con * gamma_square * (3 * x + 1 + 45. / epsilon) for (con, gamma_square, x, epsilon)
                          in zip(concentrations, gamma_squares, relative_transmisions, epsilons)])
    delta_m = 3. / 4. * numerator / denominator
    return delta_m


class DepolarizationLidar:

    def __init__(self, wavelength=532.0, fwhm=0.55):
        '''
        This class calculates the volume depolarization ratio of the molecular 
        backscatter signal dected with a polarization lidar. 

        It's unputs are:

        wavelength  - The lidar emission wavelength in nm
        fhwm        - The fhwm of the detecting filter

        Available methods:

        delta_mol_temperature(T) - calculate the molecular depolarization ratio at temperature T (in Kelvin)

        '''
        self.J_stokes = np.arange(0, 60)
        self.J_astokes = np.arange(2, 60)

        self.wavelength = float(wavelength)
        self.wavenumber = 10 ** 7 / self.wavelength

        self.optical_filter = FilterFunction(self.wavelength, fwhm)

        self.dn_stokes_N2 = np.array([delta_n_stokes(J, N2_parameters) for J in self.J_stokes])
        self.dn_astokes_N2 = np.array([delta_n_antistokes(J, N2_parameters) for J in self.J_astokes])

        self.wl_stokes_N2 = 1e7 / (self.wavenumber - self.dn_stokes_N2)
        self.wl_astokes_N2 = 1e7 / (self.wavenumber - self.dn_astokes_N2)

        self.dn_stokes_O2 = np.array([delta_n_stokes(J, O2_parameters) for J in self.J_stokes])
        self.dn_astokes_O2 = np.array([delta_n_antistokes(J, O2_parameters) for J in self.J_astokes])

        self.wl_stokes_O2 = 1e7 / (self.wavenumber - self.dn_stokes_O2)
        self.wl_astokes_O2 = 1e7 / (self.wavenumber - self.dn_astokes_O2)

    def delta_mol_temperature(self, T):

        x_N2, x_O2 = self.rotation_contribution_temperature(T)

        delta_m = delta_mol(self.wavenumber, [0.78084, 0.209476], [N2_parameters, O2_parameters], [x_N2, x_O2])
        
        return delta_m

    def rotation_contribution_temperature(self, T):
        ds_stokes = np.array([delta_sigma_stokes(
            self.wavenumber, J, T, N2_parameters) for J in self.J_stokes])
        ds_astokes = np.array([delta_sigma_antistokes(
            self.wavenumber, J, T, N2_parameters) for J in self.J_astokes])

        x_N2 = (np.sum(self.optical_filter(self.wl_stokes_N2) * ds_stokes) +
                np.sum(self.optical_filter(self.wl_astokes_N2) * ds_astokes)) / (np.sum(ds_stokes) + np.sum(ds_astokes))

        ds_stokes = np.array([delta_sigma_stokes(
            self.wavenumber, J, T, O2_parameters) for J in self.J_stokes])
        ds_astokes = np.array([delta_sigma_antistokes(
            self.wavenumber, J, T, O2_parameters) for J in self.J_astokes])

        x_O2 = (np.sum(self.optical_filter(self.wl_stokes_O2) * ds_stokes) +
                np.sum(self.optical_filter(self.wl_astokes_O2) * ds_astokes)) / (np.sum(ds_stokes) + np.sum(ds_astokes))
        return x_N2, x_O2

    def delta_mol_at_altitude(self, altitudes):
        """ Calculates delta_mol at an altitude (in m)
        """

        atmosphere = us_std.Atmosphere()
        Ts = np.array([atmosphere.temperature(altitude)
                       for altitude in altitudes])

        delta_mols = []

        for T in Ts:
            delta_m = self.delta_mol_temperature(T)
            delta_mols.append(delta_m)

        return delta_mols

    def plot_spectrum(self, T, molecular_parameters, show_filter=True, show_filtered=True, legend=True, suptitle=None,
                      figsize=(8, 5), xlim=None):
        ds_stokes = [delta_sigma_stokes(
            self.wavenumber, J, T, molecular_parameters) for J in self.J_stokes]
        ds_astokes = [delta_sigma_antistokes(
            self.wavenumber, J, T, molecular_parameters) for J in self.J_astokes]

        dn_stokes = np.array(
            [delta_n_stokes(J, molecular_parameters) for J in self.J_stokes])
        dn_astokes = np.array(
            [delta_n_antistokes(J, molecular_parameters) for J in self.J_astokes])

        dl_stokes = 1 / (1 / self.wavelength + np.array(dn_stokes) * 10 ** -7)
        dl_astokes = 1 / \
            (1 / self.wavelength + np.array(dn_astokes) * 10 ** -7)

        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)
        
        x_step = np.diff(dl_stokes)[0]
        bar_width = x_step / 4.

        bar1 = ax1.bar(dl_stokes, ds_stokes, width=bar_width, color='blue', label='Full')
        ax1.bar(dl_astokes, ds_astokes, width=bar_width, color='blue')

        lines = [bar1[0], ]
        legends = ['Full', ]

        if show_filtered:
            bar2 = ax1.bar(dl_stokes, self.optical_filter(dl_stokes)
                    * ds_stokes, width=bar_width, color='green', alpha=1, label='Filtered')
            ax1.bar(dl_astokes, self.optical_filter(dl_astokes)
                    * ds_astokes, width=bar_width, color='green', alpha=1)
            #ax1.bar(dl_astokes, ds_astokes, width = 0.1, color = 'blue')

            lines.append(bar2[0])
            legends.append('Filtered')

        ax1.set_xlabel('Wavelength [nm])')
        ax1.set_ylabel(r'$\left( \frac{d\sigma}{d\omega}\right)_{\pi}$ [$cm^{2}sr^{-1}$]')
        #ax1.grid(True)
        
        if show_filter:
            ax2 = ax1.twinx()

            xmin, xmax = ax1.get_xlim()

            filter_wavelengths = np.linspace(xmin, xmax, 1000)
            filter_efficiency = self.optical_filter(filter_wavelengths)

            filter_label = 'Filter fwhm %s nm' % self.optical_filter.fwhm
            line_1, = ax2.plot(filter_wavelengths, filter_efficiency, '-g',
                     label=filter_label)
            ax2.set_ylabel('Filter efficiency')

            ax2.yaxis.label.set_color('green')
            ax2.tick_params(axis='y', colors='green')

            lines.append(line_1)
            legends.append(filter_label)
            plt.ylim(0, 1)

        if suptitle==None:
            fig.suptitle('Rotational Raman spectrum of $%s$' %
                        molecular_parameters['name'])
        else:
            fig.suptitle(suptitle)

        if legend:
            ax1.legend(lines, legends, loc=1)

        if xlim is not None:
            plt.xlim(xlim)

        plt.draw()
        plt.show()


class FilterFunction:

    def __init__(self, wavelength, fwhm):
        '''
        This simple class represents a Gaussian filter function. To generate 
        a new filter use::

           my_filter = FilterFunction(wavelength, fwhm)

        with

        wavelength - The central wavelength of the filter in nm
        fwhm       - The fwhm of the filter in nm

        If the the filter is called with a wavelength (in nm) as an argument 
        it will return the  efficiency at this wavelength, for example::

           my_filter = FilterFunction(532, 5)
           my_filter(532) # Will return 1.0
           my_filter(535) # Will return 0.3685
        '''
        self.wavelength = wavelength
        self.fwhm = fwhm
        self.c = fwhm / 2.354820045031

    def __call__(self, wavelength):
        #value = np.exp(-(wavelength - self.wavelength) ** 2 / (2 * self.c ** 2))
        value = np.exp((-4 * np.log(2) * (wavelength - self.wavelength) ** 2) / self.fwhm**2)
        return value
