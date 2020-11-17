#!/usr/bin/env python
# MIT License
# Copyright (c) 2020  Adi Vainiger
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Modules for physical and pollyXT constants.
"""

import numpy as np

# %% Basic physics constants

eps = np.finfo(np.float).eps
c = 299792.458  # Speed of light [Km/sec]
h_plank = 6.62606e-34  # plank constant [J sec]

# %% pollyXT Lidar info
n_chan = 13
min_height = 0.229  # The height of the Lidar in Haifa [km] 


class CHANNELS():
    def __init__(self):
        pass

    UV = 0  # UV channel - 355[nm]
    UVs = 1  # UV polarized chnanel - 355 [nm]
    V_1 = 2  # V_1 Ramman channel of Nitrogen N2 - 387[nm]
    V_2 = 3  # V_2 Ramman channel of water H2O - 407[nm]
    G = 4  # Green channel - 532[nm]
    Gs = 5  # Green channel - 532[nm]
    R = 6  # Red Raman channel - 607[nm]
    IR = 7  # IR channel - 1064[nm]
    GNF = 8  # Near Field green channel - 532[nm]
    RNF = 9  # Near Field red channel - 607[nm]
    UVNF = 10  # Near Field UV channel - 355[nm]
    V1NF = 11  # Near field Ramman channel - 387[nm]


class LAMBDA():
    def __init__(self):
        pass

    UV = 355e-9  # UV channel - 355[nm]
    V_1 = 387e-9  # V_1 Ramman channel of Nitrogen N2 - 386[nm]
    V_2 = 407e-9  # V_2 Ramman channel of water H2O - 407[nm]
    G = 532e-9  # Green channel - 532[nm]
    R = 607e-9  # Red Raman channel - 607[nm]
    IR = 1064e-9  # IR channel - 1064[nm]

# %%DEBUG -----------------------------
if __name__ == '__main__':
	print('This files contains some useful constants')