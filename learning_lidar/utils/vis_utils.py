import seaborn as sns
from matplotlib import dates as mdates, pyplot as plt

FIGURE_DPI = 300
SAVEFIG_DPI = 300
SMALL_FONT_SIZE = 14
MEDIUM_FONT_SIZE = 16
BIG_FONT_SIZE = 18
TITLE_FONT_SIZE = 18
SUPTITLE_FONT_SIZE = 20
TIMEFORMAT = mdates.DateFormatter('%H:%M')
MONTHFORMAT = mdates.DateFormatter('%Y-%m')
COLORS = ["darkblue", "darkgreen", "darkred"]


def set_visualization_settings():
    # TODO make sure this actually propagates to other functions
    plt.rcParams['figure.dpi'] = FIGURE_DPI
    plt.rcParams['savefig.dpi'] = SAVEFIG_DPI

    # Create an array with the colors to use

    # Set a custom color palette
    sns.set_palette(sns.color_palette(COLORS))

    plt.rc('font', size=SMALL_FONT_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=TITLE_FONT_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIG_FONT_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_FONT_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_FONT_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_FONT_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=SUPTITLE_FONT_SIZE)  # fontsize of the figure title
