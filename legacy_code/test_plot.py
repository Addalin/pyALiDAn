# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:40:49 2019

@author: addalin
"""

import matplotlib.pyplot as plt
import numpy as np

import plotly.plotly as py
import plotly.tools as tls
# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api

mpl_fig = plt.figure()
ax = mpl_fig.add_subplot(111)

ax.plot(range(10))
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')

ax.xaxis.label.set_color('red')
ax.yaxis.label.set_color('green')

plotly_fig = tls.mpl_to_plotly( mpl_fig )
plot_url = py.plot(plotly_fig, filename='mpl-axes-color')