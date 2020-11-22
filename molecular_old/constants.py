"""
Basic physics constants used in all submodules.

Values taken from http://physics.nist.gov/cuu/Constants/index.html (2014 values)
"""
h = 6.626070040e-34  # plank constant in J s
c = 299792458.  # speed of light in m s-1
k_b = 1.38064852 * 10**-23  # Boltzmann constant in J/K
# k_b = 1.3806504 * 10**-23  # J/K  - Folker value

# Molar gas constant
#R = 8.3144598  # in J/mol/K --
R = 8.314510  # Value in Ciddor 1996.


# plank constant * speed of light in cm * J
hc = h * c * 100  # cm * J

# plank constant * speed of light / Boltzmann constant in cm * K
hc_k = h * c / k_b * 100  # cm * K


