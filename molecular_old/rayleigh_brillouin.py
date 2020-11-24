""" Functions to calculate the spectral broadening due of incident laser light
due to thermal motion of molecules. For high pressure / low temperature
gases, the shape of the spectrum is not Gaussian.
"""
import math
import numpy as np


def analytic_model(x, y):
    """ This is an analytic approximation to the Tenti S6 model, proposed
    by B. Witschas. 
    
    The fit was made assuming atmospheric air (mixture of N2 and O2). The 
    approximation is valid for y = 0 -- 1.027. Within that regions the
    deviations are less than 0.85%.  
    
    IMPORTANT: The model was constructed using a constant temperature T=250K.
    It is not valid for very different temperatures
    
    Witschas, B. Analytical model for Rayleigh-Brillouin line shapes in air. 
    Appl. Opt. 50, 267-270 (2011).

    Input
    -----
    x: 
       Normalized optical frequency shift.
    y:
       Normalized collision frequency.
       
    Output
    ------
    S:
       Normalized integrated intensity
       
    """
    
    # Calculate the empirical parameters for given y
    A = 0.18526 * np.exp(-1.31255 * y)  \
        + 0.07103 * np.exp(-18.26117 * y) + 0.74421
   
    # The width of the Rayleigh pick
    sigma_r = 0.70813 - 0.16366 * y**2 + 0.19132 * y**3 - 0.07217 * y**4   # Corrected from errata
    # The width of the side Brillouin picks
    sigma_b = 0.07845 * np.exp(-4.88663 * y) + 0.804 * np.exp(-0.15003 * y) - 0.45142
    
    # The location of the side Brillouin picks
    x_b = 0.80893 - 0.30208 * 0.10898 ** y
    
    # Calculate the intensity of the three picks separately, for clarity.
    S_r =  A / (math.sqrt(2 * math.pi) * sigma_r) * np.exp(-0.5 * (x / sigma_r)**2)
    
    S_b1 = (1 - A) / (2 * math.sqrt(2 * math.pi) * sigma_b) * np.exp(-0.5 * ((x + x_b) / sigma_b)**2)
    
    S_b2 = (1 - A) / (2 * math.sqrt(2 * math.pi) * sigma_b) * np.exp(-0.5 * ((x - x_b) / sigma_b)**2)
    
    S = S_r + S_b1 + S_b2
    
    return S

    
