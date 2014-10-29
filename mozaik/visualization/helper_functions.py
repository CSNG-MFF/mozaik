"""
This module contains several low level plotting function used mainly in simple_plot module.
"""
import pylab
from matplotlib.ticker import FuncFormatter


def disable_top_right_axis(ax):
    for loc, spine in ax.spines.iteritems():
        if loc in ['right', 'top']:
            spine.set_color('none')  # don't draw spine
    for tick in ax.yaxis.get_major_ticks():
        tick.tick2On = False
    for tick in ax.xaxis.get_major_ticks():
        tick.tick2On = False
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    

def disable_bottom_axis(ax):
    for loc, spine in ax.spines.iteritems():
        if loc in ['bottom']:
            spine.set_color('none')  # don't draw spine
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1On = False


def disable_left_axis(ax):
    for loc, spine in ax.spines.iteritems():
        if loc in ['left']:
            spine.set_color('none')  # don't draw spine
    for tick in ax.yaxis.get_major_ticks():
        tick.tick1On = False

def three_tick_axis(axis,log=False):
    import matplotlib.ticker as mticker
    if log:
        axis.set_major_locator(mticker.LogLocator(numticks=3))
    else:
        axis.set_major_locator(mticker.LinearLocator(3))
    def millions(x, pos):
        s_g = '%.4g' % (x)
        s_f = '%.4f' % (x)
        if len(s_f) < len(s_g):
            return s_f
        return s_g
        
    a = FuncFormatter(millions)
    axis.set_major_formatter(a)

def short_tick_labels_axis(axis):
    def millions(x, pos):
        s_g = '%.4g' % (x)
        s_f = '%.4f' % (x)
        if len(s_f) < len(s_g):
            return s_f
        return s_g
        
    a = FuncFormatter(millions)
    axis.set_major_formatter(a)

    

def disable_xticks(ax):
    for t in ax.xaxis.get_ticklines():
        t.set_visible(False)


def disable_yticks(ax):
    for t in ax.yaxis.get_ticklines():
        t.set_visible(False)


def remove_x_tick_labels():
    pylab.xticks([], [])


def remove_y_tick_labels():
    pylab.yticks([], [])
