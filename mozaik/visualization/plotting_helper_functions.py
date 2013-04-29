"""
docstring goes here
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


def three_tick_axis1(axis):
    import matplotlib.ticker as mticker
    axis.set_major_locator(mticker.LinearLocator(3))
    axis.set_major_formatter(mticker.FormatStrFormatter('%.2g'))

def three_tick_axis(axis):
    import matplotlib.ticker as mticker
    axis.set_major_locator(mticker.LinearLocator(3))
    def millions(x, pos):
        s_g = '%2g' % (x)
        s_f = '%2f' % (x)
        if len(s_f) < len(s_g):
            return s_f
        return s_g
    axis.set_major_formatter(FuncFormatter(millions))

    

def disable_xticks(ax):
    for t in ax.xaxis.get_ticklines():
        t.set_visible(False)


def disable_yticks(ax):
    for t in ax.yaxis.get_ticklines():
        t.set_visible(False)


def remove_x_tick_labels():
    print 'YO'
    pylab.xticks([], [])


def remove_y_tick_labels():
    pylab.yticks([], [])
