import numpy as np

'''
Calculates a (slightly modified version of) the US standard atmosphere. 
The temperature and pressure can be modified and the atmospheric profile
will be adjusted accordingly. 

Temperature in Kelvin
Pressure in hPa
Altitude in m


This is basically a copy of the AeroCalc package of Kevin Horton. 
The original (and much more complete) module can be found at

http://www.kilohotel.com/python/aerocalc/

From the original module:

# #############################################################################
# Copyright (c) 2008, Kevin Horton
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# *
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Kevin Horton may not be used to endorse or promote products
#       derived from this software without specific prior written permission.
# *
# THIS SOFTWARE IS PROVIDED BY KEVIN HORTON ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL KEVIN HORTON BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# #############################################################################

'''


class Atmosphere:

    def __init__(self, t_r=288.15, p_r=1013.25, alt=0.0):
        '''
        This class represents a standard atmosphere. It's values are modified
        based on ground measurements (in contrast with the US standard atmosphere)
        You need to specify the temperature and pressure at an altitude < 10km

        You can use it like::

           my_atmosphere = Atmosphere(t_r, p_r, alt)

        where 

        t_r  - The temperature (in Kelvin)
        p_r  - The pressure (in hPa)
        alt  - The altitude of the above two values (in meters)

        Alternatively, you can get THE US standard atmosphere by::

           my_atmosphere = Atmosphere()

        After this you can do for example the following:

            my_atmosphere.temperature(10000) # Gives the temperature at 10 km
            my_atmosphere.pressure(10000) # Gives the pressure at 10 km
            my_atmosphere.density(10000) # Gives the density at (in km/m^3) at 10 km
        '''
        alt = alt / 1000.0  # Make the calculations in km

        self.T_r = t_r
        self.P_r = p_r

        # Acceleration of gravity at 45.542 deg latitude, m/s**s
        self.g = 9.80665
        self.Rd = 287.056987  # R=8.3144621, m_w = 28.9645  Gas constant for dry air, J/kg K (Old value: 287.05307)

        # conditions starting at sea level, in a region with temperature
        # gradient

        self.L0 = -6.5  # Temperature lapse rate, at sea level deg K/km
        self.T0 = t_r - alt * self.L0  # Temperature at sea level, degrees K
        # Pressure at sea level, (in mbar ?)
        self.P0 = p_r / (self.T0 / t_r) ** -5.255
        # Density at sea level, kg/m**3, 100* is needed to convert hPa to Pa
        self.Rho0 = 100 * self.P0 / (self.Rd * self.T0)

        # conditions starting at 11 km, in an isothermal region

        self.T11 = self.T0 + 11 * self.L0  # Temperature at 11,000 m, degrees K
        # pressure ratio at 11,000 m
        self.PR11 = (
            self.T11 / self.T0) ** ((-1000 * self.g) / (self.Rd * self.L0))
        self.P11 = self.PR11 * self.P0
        self.Rho11 = (self.Rho0 * self.PR11) * (self.T0 / self.T11)

        # conditions starting at 20 km, in a region with temperature gradient

        self.T20 = self.T11
        self.PR20 = self.PR11 * \
            np.exp(((-1000 * self.g) * (20 - 11)) / (self.Rd * self.T11))
        self.L20 = 1  # temperature lapse rate, starting at 20,000 m, deg K/km
        self.P20 = self.PR20 * self.P0
        self.Rho20 = (self.Rho0 * self.PR20) * (self.T0 / self.T20)

        # conditions starting at 32 km, in a region with temperature gradient

        # Temperature at 32 km, degrees K (before was 228.65).
        self.T32 = self.T20 + self.L20 * (32 - 20)
        self.PR32 = self.PR20 * \
            (self.T32 / self.T20) ** ((-1000 * self.g) / (self.Rd * self.L20))

        # PR32 = PR20 * M.exp((-1000 * g) * (32 - 20)/(R * T20))

        # temperature lapse rate, starting at 32,000 m, deg K/km
        self.L32 = 2.8
        self.P32 = self.PR32 * self.P0
        self.Rho32 = (self.Rho0 * self.PR32) * (self.T0 / self.T32)

        # conditions starting at 47 km, in an isothermal region

        self.T47 = self.T32 + self.L32 * (47 - 32)  # before 270.65
        self.PR47 = self.PR32 * \
            (self.T47 / self.T32) ** ((-1000 * self.g) / (self.Rd * self.L32))
        self.P47 = self.PR47 * self.P0
        self.Rho47 = (self.Rho0 * self.PR47) * (self.T0 / self.T47)

        # conditions starting at 51 km, in a region with temperature gradient

        self.T51 = self.T47  # Temperature at 51 km, degrees K
        self.PR51 = self.PR47 * \
            np.exp(((-1000 * self.g) * (51 - 47)) / (self.Rd * self.T47))
        # temperature lapse rate, starting at 51,000 m, deg K/km
        self.L51 = -2.8
        self.P51 = self.PR51 * self.P0
        self.Rho51 = (self.Rho0 * self.PR51) * (self.T0 / self.T51)

        # conditions starting at 71 km, in a region with temperature gradient

        # Temperature at 71 km, degrees K
        self.T71 = self.T51 + self.L51 * (71 - 51)
        self.PR71 = self.PR51 * \
            (self.T71 / self.T51) ** ((-1000 * self.g) / (self.Rd * self.L51))
        self.L71 = - \
            2.  # temperature lapse rate, starting at 71,000 m, deg K/km
        self.P71 = self.PR71 * self.P0
        self.Rho71 = (self.Rho0 * self.PR71) * (self.T0 / self.T71)

    def temperature(self, H):
        """Return the standard temperature for the specified altitude. 
        H in meters
        """
        H = H / 1000.0  # Make the calculations in km

        if H <= 11:
            temp = self.T0 + H * self.L0
        elif H <= 20:
            temp = self.T11
        elif H <= 32:
            temp = self.T20 + (H - 20) * self.L20
        elif H <= 47:
            temp = self.T32 + (H - 32) * self.L32
        elif H <= 51:
            temp = self.T47
        elif H <= 71:
            temp = self.T51 + (H - 51) * self.L51
        elif H <= 84.852:
            temp = self.T71 + (H - 71) * self.L71
        else:
            raise ValueError('This function is only implemented for altitudes of 84.852 km and below.')

        return temp

    def _alt2press_ratio_gradient(self, H, Hb, Pb, Tb, L,):
        # eqn from USAF TPS PEC binder, page PS1-31
        return (Pb / self.P0) * (1 + (L / Tb) * (H - Hb)) ** ((-1000 * self.g) / (self.Rd
                                                                                  * L))

    def _alt2press_ratio_isothermal(self, H, Hb, Pb, Tb,):
        # eqn from USAF TPS PEC binder, page PS1-26
        return (Pb / self.P0) * np.exp((-1 * (H - Hb)) * ((1000 * self.g) / (self.Rd * Tb)))

    def _alt2press_ratio(self, H):
        """
        Return the pressure ratio (atmospheric pressure / standard pressure
        for sea level).   
        """

        if H <= 11:
            return self._alt2press_ratio_gradient(H, 0, self.P0,  self.T0,  self.L0)
        if H <= 20:
            return self._alt2press_ratio_isothermal(H, 11,  self.P11,  self.T11)
        if H <= 32:
            return self._alt2press_ratio_gradient(H, 20,  self.P20,  self.T20,  self.L20)
        if H <= 47:
            return self._alt2press_ratio_gradient(H, 32,  self.P32,  self.T32,  self.L32)
        if H <= 51:
            return self._alt2press_ratio_isothermal(H, 47,  self.P47,  self.T47)
        if H <= 71:
            return self._alt2press_ratio_gradient(H, 51,  self.P51,  self.T51,  self.L51)
        if H <= 84.852:
            return self._alt2press_ratio_gradient(H, 71,  self.P71,  self.T71,  self.L71)
        else:
            raise ValueError('This function is only implemented for altitudes of 84.852 km and below.')

    def pressure(self, H):
        """
        Return the atmospheric pressure for a given altitude.
        """
        H = H / 1000.0  # Make the calculations in km

        press = self.P0 * self._alt2press_ratio(H)

        return press

    def _alt2density_ratio(self, H):
        """
        Return the density ratio (atmospheric density / standard density
        for sea level).  
        """

        return self._alt2press_ratio(H) / (self.temperature(H) / self.T0)

    def density(self, H):
        """
        Return the density given the pressure altitude. 
        """
        #H = H / 1000.0 # Make the calculations in km
        # get density in kg/m**3

        #dens = self.Rho0 * self._alt2density_ratio(H)
        
        # IMPORTANT! Original calculations are probably wrong (at least 
        # are different from what is given online). So density calculation 
        # are substituted with direct calculations.
        
        p = self.pressure(H) * 100  # in Pascal
        T = self.temperature(H)  # in K
        dens = p / (self.Rd * T)
        return dens
