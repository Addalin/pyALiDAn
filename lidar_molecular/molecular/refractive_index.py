""" This file includes functions to calculate the refractive index of air
according to Ciddor (1996, 2002), summarized by Tomasi et al. (2005).
"""

import numpy as np
from .constants import R


def n_air(wavelength, pressure, temperature, C, relative_humidity):
    """ Calculate the refractive index of air. 
    
    Parameters
    ----------
    wavelength : float
       Light wavelegnth [nm]
    pressure : float
       Atmospheric pressure [hPa]
    temperature : float
       Atmospehric temperature [K]
    C : float
       Concentration of CO2 [ppmv]
    relative_humidity : float
       Relative humidity from 0 to 100 [%]
    
    Returns
    -------
    n_air : float
       Refractive index of air.
    """

    Xw = molar_fraction_water_vapour(pressure, temperature, relative_humidity)

    rho_axs, _, _ = moist_air_density(1013.25, 288.15, C, 0)
    rho_ws, _, _ = moist_air_density(13.33, 293.15, 0, 1)  # C not relevant

    _, rho_a, rho_w = moist_air_density(pressure, temperature, C, Xw)

    n_axs = n_standard_air_with_CO2(wavelength, C)
    n_ws = n_water_vapor(wavelength)

    n = 1 + (rho_a / rho_axs) * (n_axs - 1) + (rho_w / rho_ws) * (n_ws - 1)

    return n


def moist_air_density(pressure, temperature, C, Xw):
    """ Calculate the moist air density using the BIPM (Bureau International des
    Poids et Mesures) 1981/91 equation. See Tomasi et al. (2005), eq. 12.
    
    Parameters
    ----------
    pressure: float
       Total pressure [hPa]
    temperature: float
       Atmospehric temperature [K]
    C: float
       CO2 concentration [ppmv]
    Xw: float
       Molar fraction of water vapor
    """
    Ma = molar_mass_dry_air(C)  # in kg/mol--  Molar mass  dry air.
    Mw = 0.018015  # in kg/mol -- Molar mass of water vapour. 

    Z = compressibility_of_moist_air(pressure, temperature, Xw)

    P = pressure * 100.  # In Pa
    T = temperature

    rho = P * Ma / (Z * R * T) * (1 - Xw * (1 - Mw / Ma))

    rho_air = (1 - Xw) * P * Ma / (Z * R * T)
    rho_wv = Xw * P * Mw / (Z * R * T)

    return rho, rho_air, rho_wv


def molar_mass_dry_air(C):
    """ Molar mass of dry air, as a function of CO2 concentration.
    
    Parameters
    ----------
    C: float
       CO2 concentration [ppmv]
    
    Returns
    -------
    Ma: float
       Molar mass of dry air [km/mol]
    """
    C1 = 400.

    Ma = 10 ** -3 * (28.9635 + 12.011e-6 * (C - C1))

    return Ma


def molar_fraction_water_vapour(pressure, temperature, relative_humidity):
    """ Molar fraction of water vapor. 
    
    Parameters
    ----------
    pressure: float
       Total pressure [hPa]
    temperature: float
       Atmospehric temperature [K] 
    relative_humidity:
       Relative humidity from 0 to 100 [%]
    """
    # Convert units
    p = pressure  # In hPa
    h = relative_humidity / 100.  # From 0 to 1

    # Calculate water vapor pressure
    f = enhancement_factor_f(pressure, temperature)
    svp = saturation_vapor_pressure(temperature)

    p_wv = h * f * svp  # Water vapor pressure

    Xw = p_wv / p

    return Xw


def enhancement_factor_f(pressure, temperature):
    """ Enhancement factor.
    
    Parameters
    ----------
    pressure: float
       Atmospheric pressure [hPa]
    temperature: float
       Atmospehric temperature [K]    
    """
    T = temperature
    p = pressure * 100.  # In Pa

    f = 1.00062 + 3.14e-8 * p + 5.6e-7 * (T - 273.15) ** 2

    return f


def saturation_vapor_pressure(temperature):
    """ Saturation vapor pressure of water of moist air.
    
    Note: In original documentation, this was specified as the saturation pressure of 
    pure water vapour. This seems wrong. 
    
    
    Parameters
    ----------
    temperature: float
       Atmospheric temperature [K] 
    
    Returns
    -------
    E: float
       Saturation vapor pressure [hPa]
             
    References
    ----------
    Ciddor, P. E.: Refractive index of air: new equations for the visible and near 
    infrared, Appl. Opt., 35(9), 1566-1573, doi:10.1364/AO.35.001566, 1996.
    
    Davis, R. S.: Equation for the Determination of the Density of 
    Moist Air (1981/91), Metrologia, 29(1), 67, doi:10.1088/0026-1394/29/1/008, 1992.
    """
    T = temperature
    E = np.exp(1.2378847e-5 * T ** 2 - 1.9121316e-2 * T +
               33.93711047 - 6343.1645 / T)
    return E / 100.  # In hPa


def compressibility_of_moist_air(pressure, temperature, molar_fraction):
    """ Compressibility of moist air.
    
    Parameters
    ----------
    pressure: float
       Atmospheric pressure [hPa]
    temperature: float
       Atmospehric temperature [K]   
    molar_fraction: float
       Molar fraction.
       
    Note
    ----
    Eg. 16 of Tomasi et al. is missing a bracket. The formula of Ciddor 1996
    was used instead.
    """
    a0 = 1.58123e-6  # K Pa-1
    a1 = -2.9331e-8  # Pa-1
    a2 = 1.1043e-10  # K Pa-1
    b0 = 5.707e-6  # K Pa-1
    b1 = -2.051e-8  # Pa-1
    c0 = 1.9898e-4  # Pa-1
    c1 = -2.376e-6  # Pa-1
    d0 = 1.83e-11  # K2 Pa-2
    d1 = -7.65e-9  # K2 Pa-2

    p = pressure * 100.  # in Pa
    T = np.array(temperature, dtype=float)
    Tc = temperature - 273.15  # in C

    Xw = molar_fraction

    Z = 1 - (p / T) * (a0 + a1 * Tc + a2 * Tc ** 2 + (b0 + b1 * Tc) * Xw + \
                       (c0 + c1 * Tc) * Xw ** 2) + (p / T) ** 2 * (d0 + d1 * Xw ** 2)
    return Z


def n_standard_air(wavelength):
    """The refractive index of air at a specific wavelength with CO2 concentration 450 ppmv. 
    
    Calculated for standard air at T = 15C, P=1013.25hPa, e = 0, C=450ppmv. 
    (see Tomasi, 2005, eg. 17).
      
    Parameters
    ----------
    wavelength : float
       Wavelength [nm]
    
    Returns
    -------
    ns : float
       Refractivity of standard air with C = 450ppmv
    """

    wl_micrometers = wavelength / 1000.0  # Convert nm to um

    s = 1 / wl_micrometers  # the reciprocal of wavelength
    c1 = 5792105.
    c2 = 238.0185
    c3 = 167917.
    c4 = 57.362
    ns = 1 + (c1 / (c2 - s ** 2) + c3 / (c4 - s ** 2)) * 1e-8
    return ns


def n_standard_air_with_CO2(wavelength, C):
    """ The refractive index of air at a specific wavelength including random CO2. 
    
    Calculated for standard air at T = 15C, P=1013.25hPa, e = 0. 
    (see Tomasi, 2005, eq. 18)
      
    Parameters
    ----------
    wavelength : float
       Wavelength [nm]
    C : float
       CO2 concentration [ppmv]
       
    Returns
    -------
    n_axs : float
       Refractive index of air for the specified CO2 concentration.
    """
    C2 = 450.  # ppmv

    n_as = n_standard_air(wavelength)

    n_axs = 1 + (n_as - 1) * (1 + 0.534e-6 * (C - C2))

    return n_axs


def n_water_vapor(wavelength):
    """ Refractive index of water vapour. 

    Calculated for T = 20C, e=1333Pa  (see Tomasi, 2005, eq. 19)
    
    Parameters
    ----------
    wavelength: float
       Wavelength [nm]
    
    Returns
    -------
    n_wv : float
       Refractive index of water vapour.
    """
    wl_micrometers = wavelength / 1000.0  # Convert nm to um

    s = 1 / wl_micrometers  # the reciprocal of wavelength

    c1 = 1.022
    c2 = 295.235
    c3 = 2.6422
    c4 = 0.032380  # Defined positive
    c5 = 0.004028

    n_ws = 1 + c1 * (c2 + c3 * s ** 2 - c4 * s ** 4 + c5 * s ** 6) * 1e-8

    return n_ws
