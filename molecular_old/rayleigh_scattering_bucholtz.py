"""
Molecular scattering according to Bucholtz A., 1995
===================================================

This module contains functions that compute the Rayleigh scattering
parameters based on


Bucholtz, A. Rayleigh-scattering calculations for the terrestrial atmosphere.
Applied Optics, Vol. 34, No. 15, 2766-2773 (1995)

https://doi.org/10.1364/AO.34.002765

"""

import numpy as np
from scipy.integrate import cumtrapz

N_s = 2.54743 * 1E+19  # Molecular number density for standard air [cm^-3]
P_s = 1013.25  # Pressure for standard air [mbars]
T_s = 288.15  # The temperature for standard air [K]


def scattering_cross_section(wavelength):
    r"""
    Calculation of the total Rayleigh scattering cross section per molecule,
    for a specific wavelength.
    
    Parameters
    ----------
    wavelength : float or array of floats
       The wavelength of the radiation [nanometers].
    
    Returns
    -------
    sigma : float or array of floats
       The Rayleigh scattering cross section per molecule [:math:`cm^2`].
    
    Notes
    -----
    The total Rayleigh-scattering cross section per molecule, is calculated according [1]_:
    
    .. math::

       \sigma(\lambda) = \frac{24 \pi^3 (n_s^2 - 1)^2}{\lambda^4 N_s^2 (n_s^2 + 2)^2} \cdot F_k

    where

    - :math:`\sigma` : the scattering cross section.
    - :math:`\lambda` : the wavelength [cm].
    - :math:`n_s` : the refractive index.
    - :math:`N_s` : the molecular number density for standard air (:math:`2.54743 \cdot 10^{19}cm^{-3}`).
    - :math:`F_K` : the King correction factor.
    
    Although the Rayleigh-scattering cross section per molecule is independent
    of temperature or pressure, standard air is assumed for consistency reasons
    when selecting the values of :math:`n_s` and :math:`N_s`.
    
    "Standard air" is defined as dry air containing :math:`0.03% CO_2` by volume
    at normal pressure :math:`P_s = 1013.25mb` and air temperature :math:`T_s = 288.15K`.
    
    References
    ----------
    .. [1] Bucholtz, A. Rayleigh-scattering calculations for the terrestrial atmosphere.
       Applied Optics, Vol. 34, No. 15, 2766-2773 (1995)
    """
    #   Calculate the refractive index.
    n_s = refractive_index_standard_air(wavelength)

    #   Calculate the King correction factor.
    f_king = king_correction_factor(wavelength)

    #   Convert from nanometers to centimeters.
    wavelength_cm = wavelength * 1E-7

    #   Calculate the numerator of the formula.
    numerator = 24 * np.pi ** 3 * (n_s ** 2 - 1) ** 2

    #   Calculate the denominator of the formula.
    denominator = wavelength_cm ** 4 * N_s ** 2 * (n_s ** 2 + 2) ** 2

    #   Calculate the Rayleigh-scattering cross section per molecule.
    sigma = numerator * f_king / denominator

    return sigma


def volume_scattering_coefficient(wavelength, pressure, temperature):
    r"""
    Calculation of the total Rayleigh volume-scattering coefficient, for a 
    specific wavelength, pressure, and temperature.
        
    Parameters
    ----------
    wavelength: float or array of floats
       The wavelength of the radiation [nanometers].
    pressure: float or array of floats
       The atmospheric pressure [mbars].
    temperature: float or array of floats
       The atmospheric temperature [K].
           
    Returns
    -------
    beta: float or array of floats
       The total Rayleigh volume-scattering coefficient [:math:`km^-1`].
       
    Notes
    -----
    The amount of scattering for a volume of gas, characterized by the total
    Rayleigh volume-scattering coefficient, is given by the following formula:
    
    .. math::
       \beta(\lambda, z) = N(z) \cdot \sigma(\lambda)
    
    where,
    
    :math:`N(z)` : product of the molecular number density at a given altitude.
    :math:`\sigma(\lambda)` : the total Rayleigh cross section per molecule.
    
    For standard air [1]_ is assuming molecular number density :math:`N_s = 2.54743 \cdot 10^{19}.
    As the total Rayleigh volume-scattering coefficient scales with the
    molecular number density, the correction to any pressure and temperature is
    done using:
    
    .. math::
       \beta = \beta_s \frac{N}{N_s} = \beta_s \frac{P}{P_s} \frac{T_s}{T}

    where
    
    - :math:`\beta_s` : the reference Rayleigh volume-scattering coefficient.
    - :math:`\N_s` : the molecular number density at which :math:`\beta_s` was calculated.
    - :math:`\P_s` : the pressure at which :math:`\beta_s` was calculated.
    - :math:`\T_s` : the temperature at which :math:`\beta_s` was calculated [K].
    - :math:`\P` : any pressure [units same as :math:`\P_s`].
    - :math:`\T` : any temperature [K].
    
    References
    ----------
    .. [1] Bucholtz, A. Rayleigh-scattering calculations for the terrestrial atmosphere.
       Applied Optics, Vol. 3
    """
    #   Calculate the total Rayleigh volume-scattering coefficient.
    beta = N_s * scattering_cross_section(wavelength) * pressure * T_s / (P_s * temperature)

    #   Convert from cm^-1 to km^-1.
    beta_km = beta * 1E+5

    return beta_km


def rayleigh_phase_function(scattering_angle, wavelength):
    r"""
    Rayleigh phase function calculation.
        
    Parameters
    ----------
    scattering_angle: float
       The scattering angle [radians].
    wavelength: float
       The wavelength of the radiation [nanometers].
       
    Returns
    -------
    p_ray: float
       The Rayleigh phase function.
       
    Notes
    -----
    The Rayleigh phase function describes the angular distribution of unpolarized
    radiation by the air.
    
    The formula given by Chandrasekhar is used for the calculation, in order to
    account for the molecular anisotropy effects:
    
    .. math::
       P_{ray}(\theta) = \frac{3}{4(1 + 2\gamma)}[(1 + 3\gamma) + (1 - \gamma) cos^2 \theta]
       
    where, :math:`\gamma = \frac{\rho_n}{2 - \rho_n}`
    and :math:`\rho_n` is the depolarization factor.
    
    References
    ----------
    Bucholtz, A. Rayleigh-scattering calculations for the terrestrial atmosphere.
    Applied Optics, Vol. 34, No. 15, 2766-2773 (1995)
    """
    #   Calculate the depolarization factor.
    rho_n = depolarization_factor(wavelength)

    #   Calculate the anistropy relateed, gamma.
    gamma = rho_n / (2 - rho_n)

    #   Calculate the 1st part of the equation.
    f1 = 3 / (4 * (1 + 2 * gamma))

    #   Calculate the 2nd part of the equation
    f2 = (1 + 3 * gamma) + (1 - gamma) * (np.cos(scattering_angle)) ** 2

    #   Calculate the Rayleigh phase function.
    p_ray = f1 * f2

    return p_ray


def angular_volume_scattering_coefficient(wavelength, pressure, temperature, scattering_angle):
    r"""
    Angular volume-scattering coefficient calculation.
    
    Parameters
    ----------
    wavelength: float
       The wavelength of the radiation [nanometers].
    pressure: float
       The atmospheric pressure [mbars].
    temperature: float
       The atmospheric temperature [K].
    scattering_angle: float
       The scattering angle [radians].
    
    Returns
    -------
    beta_angular: float
       The angular volume-scattering coefficient [:math:`km^-1`].
    
    Notes
    -----
    In order to include the dispersion of the depolarization factor with wavelength,
    in the calculation of the angular volume-scattering coefficient, the following
    formula is used:
    
    .. math::
       \beta(\theta, \lambda, z) = \frac{\beta(\lambda, z)}{4\pi} P_{ray}(\theta, \lambda)
       
    References
    ----------
    Bucholtz, A. Rayleigh-scattering calculations for the terrestrial atmosphere.
    Applied Optics, Vol. 34, No. 15, 2766-2773 (1995)
    """
    #   Calculate the Rayleigh volume-scattering coefficient.
    beta = volume_scattering_coefficient(wavelength, pressure, temperature)

    #   Calculate the Rayleigh phase function.
    p_ray = rayleigh_phase_function(scattering_angle, wavelength)

    #   Calculate the angular volume-scattering coefficient.
    beta_angular = beta * p_ray / (4 * np.pi)

    return beta_angular


def atmospheric_optical_depth(wavelength, pressure, temperature, altitude):
    r"""
    Calculate the Rayleigh optical depth at certain altitude.
    
    Parameters
    ----------
    wavelength: float
       The wavelength of the radiation [nanometers].
    pressure: array
       The pressure profile [mbars].
    temperature: array
       The temperature profile [K].
    altitude: array
       The altitude corresponding to the profiles of the physical quantities. [km]
       
    Returns
    -------
    tau: float
       The atmospheric optical depth.
       
    Notes
    -----
    The Rayleigh optical depth :math:`\tau` at a certain altitude :math:`z_0`
    is given as the integral of the total volume-scattering coefficient from :math:`z_0`
    to the top of the atmosphere:
    
    .. math::
       \tau(\lambda, z_0) = \int_{z_0}^{\infty} \beta_s(\lambda) \frac{P(z)}{P_s} \frac{T_s}{T(z)} dz
       
    where,
    - :math:`P_s` : standard air pressure.
    - :math:`T_s` : standard air temperature [K].
    - :math:`P(z)` : pressure profile [units same as :math:`P_s`].
    - :math:`T(z)` : temperature profile [K].
    
    References
    ----------
    Bucholtz, A. Rayleigh-scattering calculations for the terrestrial atmosphere.
    Applied Optics, Vol. 3
    """
    #   Calculate the profile of the total Rayleigh volume-scattering coefficient.
    beta = volume_scattering_coefficient(wavelength, pressure, temperature)

    #   Calculate the trapezoid.
    tau = np.trapz(beta, altitude)

    return tau


def depolarization_factor(wavelength):
    r"""
    Calculate the depolarization factor for a specific wavelength.
    Standard air is assumed: 0.03% CO2, P=1013.25mb, T=288.15K
    
    Parameters
    ----------
    wavelength: float
       The radiation wavelength. (nanometers)
       
    Returns
    -------
    depolarization_factor: float
       The depolarization factor.
       
    Notes
    -----
    A linear interpolation is applied on the tabular values of the depolarization
    factor, as seen in Table 2 of the referenced paper.

    References
    ----------
    Bucholtz, A. Rayleigh-scattering calculations for the terrestrial atmosphere.
    Applied Optics, Vol. 34, No. 15, 2766-2773 (1995)
    """
    #   Create vector with the original wavelength values that were used by Bates. (micrometers)
    wavelength_reference = np.concatenate((np.arange(0.2, 0.231, 0.005),
                                           np.arange(0.24, 0.401, 0.01),
                                           np.arange(0.45, 1.01, 0.05)),
                                          axis=0)

    #   Convert the wavelength values from micrometers to nanometers.
    wavelength_reference_nm = wavelength_reference * 1E+3

    #   Create vector with the original depolarization factor values, as calculated by Bates.
    depolarization_factor_reference = np.array([4.545, 4.384, 4.221, 4.113, 4.004, 3.895, 3.785,
                                                3.675, 3.565, 3.455, 3.4, 3.289, 3.233, 3.178,
                                                3.178, 3.122, 3.066, 3.066, 3.01, 3.01, 3.01,
                                                2.955, 2.955, 2.955, 2.899, 2.842, 2.842, 2.786,
                                                2.786, 2.786, 2.786, 2.73, 2.73, 2.73, 2.73, 2.73]) * 1E-2

    #   Interpolate the depolarization factor value according to the desired wavelength.
    depolarization_factor = np.interp(wavelength, wavelength_reference_nm, depolarization_factor_reference)

    return depolarization_factor


def _refractive_index_standard_air(wavelength):
    r"""
    Refractive index dispersion with wavelength.

    This is a two-branch function, normally accepting only floats inputs. We fake the vector input using the
    numpy.vectorize function. This may be sub-optimal for performance, but works for now.

    Standard air is assumed: 0.03% CO2, P=1013.25mb, T=288.15K

    Parameters
    ----------
    wavelength: float
       The radiation wavelength [nanometers].
    
    Returns
    -------
    n_s: float
       Refractive index.
    
    Notes
    -----
    For wavelengths greater than :math:`0.23\mu m`, we use the four-parameter formula:

    .. math::
       (n_s - 1) \cdot 10^8 = \frac{5,791,817}{238.0185 - (1/\lambda)^2}+\frac{167,909}{57.362 - (1/\lambda)^2}
    
    where the wavelength is given in micrometers.
    
    For wavelengths less than or equal to :math:`0.23\mu m`, we use the five-parameter formula:
    
    .. math::
       (n_s - 1) \cdot 10^8 = 8060.51 + \frac{2,480,990}{132.274 - (1/\lambda)^2}+\frac{17,455.7}{39.32957 - (1/\lambda)^2}
    
    References
    ----------
    Bucholtz, A. Rayleigh-scattering calculations for the terrestrial atmosphere.
    Applied Optics, Vol. 34, No. 15, 2766-2773 (1995)
    """
    wavelength_um = wavelength * 1E-3  # Convert from nanometers to micrometers.

    # Determine the parametric equation according to the wavelength_um range.
    if wavelength_um > 0.23:
        n_s = ((5791817 / (238.0185 - (1 / wavelength_um) ** 2)) +
               (167909 / (57.362 - (1 / wavelength_um) ** 2))) * 1E-8 + 1
    else:
        n_s = ((8060.51 + 2480990 / (132.274 - (1 / wavelength_um) ** 2)) +
               (17455.7 / (39.32957 - (1 / wavelength_um) ** 2))) * 1E-8 + 1

    return n_s
refractive_index_standard_air = np.vectorize(_refractive_index_standard_air)  # Make the function accept vector input


def king_correction_factor(wavelength):
    r"""
    King correction factor calculation.
    Standard air is assumed: 0.03% CO2, P=1013.25mb, T=288.15K
    
    Parameters
    ----------    
    wavelength: float
       The radiation wavelength [nanometers].
       
    Returns
    -------
    f_k: float
       The king correction factor.
    
    Documentation
    -------------
    The following formula is used: :math:`F_k = (\frac{6+3p_n}{6-7p_n})`
    
    References
    ----------
    Bucholtz, A. Rayleigh-scattering calculations for the terrestrial atmosphere.
    Applied Optics, Vol. 34, No. 15, 2766-2773 (1995)
    """
    #   Calculate the depolarization factor.
    rho_n = depolarization_factor(wavelength)

    #   Calculate the King correction factor.    
    f_king = (6 + 3 * rho_n) / (6 - 7 * rho_n)

    return f_king


### The functions that follow are not used. In the process of deletion. ###

def attenuated_profile(h, wavelength):
    """
    Calculate the attenuated backscatter molecular profile.
    
    Input: h in meters
           wavelength in nm
    Output: attenuated backscatter profile
    
    .. todo::
       The formula is not found in Bucholtz paper. Reference required.
    """
    beta = angular_scattering(h, np.pi, wavelength) * 100
    alpha = scattering(h, wavelength) * 100  # Convert to meters
    T = np.exp(-2 * cumtrapz(alpha, h))

    # Add 1 transmission to the first bin
    T = np.insert(T, 0, 1)

    beta_attenuated = beta * T
    return beta_attenuated, T


def atmospheric_optical_depth_old(wavelength, zmin=0, zmax=30000):
    """
    Calculates the total atmosperhic optical depth due to molecules, 
    at the specified wavelegnth. 
    
    Input: wavelength in nm
          zmin bottom of the atmopshere in meters
          zmax top of the atmosphere in meters

    Output: total optical depth.
    """
    h = np.linspace(zmin, zmax, 1000)
    alpha = scattering(h, wavelength) * 100  # Convert to meters
    tau = np.trapz(alpha, h)
    return tau
