'''
This module provides some function to compute Rayleigh scattering
parameters in the atmosphere. It is based on

Anthony Bucholtz, "Rayleigh-scattering calculations for the terrestrial atmosphere", 
Applied Optics 34, no. 15 (May 20, 1995): 2765-2773.    
'''

import numpy as np

from .refractive_index import n_air
from .molecular_properties import kings_factor_atmosphere, rho_atmosphere, epsilon_atmosphere
from .utilities import number_density_at_pt, rh_to_pressure

ASSUME_AIR_IDEAL = True


def sigma_rayleigh(wavelength, pressure=1013.25, temperature=288.15, C=385., rh=0.):
    ''' Calculates the Rayleigh-scattering cross section per molecule.
    
    Parameters
    ----------
    wavelength: float
       Wavelegnth [nm]
    pressure: float
       The atmospheric pressure [hPa]
    temperature: float
       The atmospheric temperature [K]   
    C: float
       CO2 concentration [ppmv].
    rh: float
       Relative humidity from 0 to 100 [%] 
       
    Returns
    --------
    sigma: float
       Rayleigh-scattering cross section [m2]
    '''
    p_e = rh_to_pressure(rh, temperature)

    # Just to be sure that the wavelength is a float
    wavelength = np.array(wavelength, dtype=float)

    # Calculate properties of standard air
    n = n_air(wavelength, pressure, temperature, C, rh)
    N = number_density_at_pt(pressure, temperature, rh, ideal=ASSUME_AIR_IDEAL)

    # Wavelength of radiation
    wl_m = wavelength * 10 ** -9  # from nm to m

    # King's correction factor
    f_k = kings_factor_atmosphere(wavelength, C=C, p_e=p_e, p_t=pressure)  # no units

    # first part of the equation
    f1 = (24. * np.pi ** 3) / (wl_m ** 4 * N ** 2)
    # second part of the equation
    f2 = (n ** 2 - 1.) ** 2 / (n ** 2 + 2.) ** 2

    # result
    sigma = f1 * f2 * f_k
    return sigma


def alpha_rayleigh(wavelength, pressure=1013.25, temperature=288.15, C=385., rh=0.):
    """ Cacluate the extinction coefficient for Rayleigh scattering. 
    
    Parameters
    ----------
    wavelength : float or array of floats
       Wavelegnth [nm]
    pressure : float or array of floats
       Atmospheric pressure [hPa]
    temperature : float
       Atmospheric temperature [K]
    C : float
       CO2 concentration [ppmv].
    rh : float
       Relative humidity from 0 to 100 [%] 
    
    Returns
    -------
    alpha: float
       The molecular scattering coefficient [m-1]
    """
    sigma = sigma_rayleigh(wavelength, pressure, temperature, C, rh)
    N = number_density_at_pt(pressure, temperature, rh, ideal=ASSUME_AIR_IDEAL)

    alpha = N * sigma

    return alpha


def phase_function(theta, wavelength, pressure=1013.25, temperature=288.15, C=385., rh=0.):
    ''' Calculates the phase function at an angle theta for a specific wavelegth.
    
    Parameters
    ----------
    theta: float
       Scattering angle [rads]
    wavelength: float
       Wavelength [nm]
    pressure: float
       The atmospheric pressure [hPa]
    temperature: float
       The atmospheric temperature [K]   
    C: float
       CO2 concentration [ppmv].
    rh: float
       Relative humidity from 0 to 100 [%]    
    
    Returns
    -------
    p: float
       Scattering phase function
       
    Notes
    -----
    The formula is derived from Bucholtz (1995). A different formula is given in 
    Miles (2001). 

    The use of this formula insetad of the wavelenght independent 3/4(1+cos(th)**2)
    improves the results for back and forward scatterring by ~1.5%

    Anthony Bucholtz, "Rayleigh-scattering calculations for the terrestrial atmosphere", 
    Applied Optics 34, no. 15 (May 20, 1995): 2765-2773.  

    R. B Miles, W. R Lempert, and J. N Forkey, "Laser Rayleigh scattering", 
    Measurement Science and Technology 12 (2001): R33-R51
    
    '''
    # TODO: Check use formula and values.
    p_e = rh_to_pressure(rh, temperature)

    r = rho_atmosphere(wavelength, C=C, p_e=p_e, p_t=pressure)
    gamma = r / (2 - r)

    # first part of the equation
    f1 = 3 / (4 * (1 + 2 * gamma))
    # second part of the equation
    f2 = (1 + 3 * gamma) + (1 - gamma) * (np.cos(theta)) ** 2
    # results
    p = f1 * f2

    return p


def dsigma_phi_rayleigh(theta, wavelength, pressure=1013.25, temperature=288.15, C=385., rh=0.):
    ''' Calculates the angular rayleigh scattering cross section per molecule.
    
    Parameters
    ----------
    theta: float
       Scattering angle [rads]
    wavelength: float
       Wavelength [nm]
    pressure: float
       The atmospheric pressure [hPa]
    temperature: float
       The atmospheric temperature [K]   
    C: float
       CO2 concentration [ppmv].
    rh: float
       Relative humidity from 0 to 100 [%] 
    
    Returns
    -------
    dsigma: float
      Angular rayleigh-scattering cross section [m2sr-1]
    '''

    phase = phase_function(theta, wavelength, pressure, temperature, C, rh) / (4 * np.pi)
    sigma = sigma_rayleigh(wavelength, pressure, temperature, C, rh)
    dsigma = sigma * phase

    return dsigma


def beta_pi_rayleigh(wavelength, pressure=1013.25, temperature=288.15, C=385., rh=0.):
    ''' Calculates the total Rayleigh backscatter coefficient.
    
    Parameters
    ----------
    wavelength: float
       Wavelength [nm]
    pressure: float
       The atmospheric pressure [hPa]
    temperature: float
       The atmospheric temperature [K]   
    C: float
       CO2 concentration [ppmv].
    rh: float
       Relative humidity from 0 to 100 [%] 
    
    Returns
    -------
    ...
    '''

    dsigma_pi = dsigma_phi_rayleigh(np.pi, wavelength, pressure, temperature, C, rh)
    N = number_density_at_pt(pressure, temperature, rh, ideal=ASSUME_AIR_IDEAL)

    beta_pi = dsigma_pi * N

    return beta_pi


def sigma_pi_cabannes(wavelength, pressure=1013.25, temperature=288.15, C=385., rh=0.):
    """ Cacluate the backscattering cross section for the cabannes line. 
    
    Parameters
    ----------
    wavelength: float
       Light wavelegnth in nm
    pressure: float
       The atmospheric pressure [hPa]
    temperature: float
       The atmospheric temperature [K]   
    C: float
       CO2 concentration [ppmv].
    rh: float
       Relative humidity from 0 to 100 [%] 
    
    Returns
    -------
    sigma:
       The backscattering cross section of the Cabannes line [m2sr-1].  
    """
    p_e = rh_to_pressure(rh, temperature)

    epsilon = epsilon_atmosphere(wavelength, C=C, p_e=p_e, p_t=pressure)

    # Calculate properties of standard air
    n = n_air(wavelength, pressure, temperature, C, rh)
    N = number_density_at_pt(pressure, temperature, rh, ideal=ASSUME_AIR_IDEAL)

    # Convert wavelegth to meters
    lamda_m = wavelength * 10 ** -9

    # Separate in three factors for clarity
    f1 = 9 * np.pi ** 2 / (lamda_m ** 4 * N ** 2)
    f2 = (n ** 2 - 1) ** 2 / (n ** 2 + 2) ** 2
    f3 = 1 + 7 / 180. * epsilon
    sigma_pi = f1 * f2 * f3

    return sigma_pi


def beta_pi_cabannes(wavelength, pressure=1013.25, temperature=288.15, C=385., rh=0.):
    """ Cacluate the backscattering coefficient for the cabannes line. 
    
    Parameters
    ----------
    wavelength: float
       Light wavelegnth in nm
    pressure: float
       The atmospheric pressure [hPa]
    temperature: float
       The atmospheric temperature [K]
    C: float
       CO2 concentration [ppmv].
    rh: float
       Relative humidity from 0 to 100 [%] 
    
    Returns
    -------
    beta_pi:
       The backscattering coefficient of the Cabannes line [m-1sr-1].  
    """

    sigma_pi = sigma_pi_cabannes(wavelength, pressure, temperature, C, rh)
    N = number_density_at_pt(pressure, temperature, rh, ideal=ASSUME_AIR_IDEAL)  # Number density of the atmosphere

    beta_pi = N * sigma_pi

    return beta_pi
