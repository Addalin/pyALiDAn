Overview
========

A collection of scripts to calculate scattering parameters of molecular atmosphere, useful (at least) for lidar data analysis.

The modules implements:

* Molecular scattering properties according to Bucholtz A. 1995
* Molecular scattering properties as summarized by Tomasi, C. et al. 2005.
* An analytical model of Reyleight-Brillouin scattering from Witschas, B. (2011)
* (Apparent) molecular depolarization at 532nm, based on A. Behrendt and T. Nakamura (2002).

Installation
============
1. Get a copy of the code.  You can do this either by cloning this repository using mercurial, or by copying the repository at
   the download section (https://bitbucket.org/iannis_b/lidar_molecular/downloads).
2. Install the module using `pip`. Assuming you have a copy of the code in the directory "lidar_molecular" you can run the command.

   .. code:: python

      pip install -e lidar_molecular

   The -e option will install the module in "editable" mode, which will allow you to edit the code and directly use the changes.

Usage
=====
You can use the module by importing the appropriate function.

.. code:: python

  from molecular import rayleigh_scattering_bucholtz
  print(rayleigh_scattering_bucholtz.scattering_cross_section(532.)) 
  # This will return 5.164829874033986e-27



References
==========
Bucholtz, A.: Rayleigh-scattering calculations for the terrestrial atmosphere, 
Appl. Opt. 34, 2765-2773 (1995) 

Tomasi, C., Vitale, V., Petkov, B., Lupi, A. and Cacciari, A.: Improved 
algorithm for calculations of Rayleigh-scattering optical depth in standard 
atmospheres, Applied Optics, 44(16), 3320, doi:10.1364/AO.44.003320, (2005)

Witschas, B. Analytical model for Rayleigh-Brillouin line shapes in air. 
Appl. Opt. 50, 267-270 (2011).

A. Behrendt and T. Nakamura, "Calculation of the calibration constant of polarization lidar 
and its dependency on atmospheric temperature," Opt. Express, vol. 10, no. 16, pp. 805-817, 2002.