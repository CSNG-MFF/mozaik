import pylab

def disable_top_right_axis(ax):
    for loc, spine in ax.spines.iteritems():
            if loc in ['right','top']:
               spine.set_color('none') # don't draw spine
    for tick in ax.yaxis.get_major_ticks():
        tick.tick2On = False
    for tick in ax.xaxis.get_major_ticks():
        tick.tick2On = False

def disable_bottom_axis(ax):
    for loc, spine in ax.spines.iteritems():
            if loc in ['bottom']:
               spine.set_color('none') # don't draw spine
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1On = False

def disable_left_axis(ax):
    for loc, spine in ax.spines.iteritems():
            if loc in ['left']:
               spine.set_color('none') # don't draw spine
    for tick in ax.yaxis.get_major_ticks():
        tick.tick1On = False


def three_tick_axis(axis):
    nticks = len([t for t in axis.get_major_ticks()])
    if (nticks % 2) != 0 and nticks > 2:
       middle_tick = int(nticks) / 2  
    else:
       middle_tick = 0
       
    for i,tick in enumerate(axis.get_major_ticks()):
        if i != 0 and i != (nticks-1) and i != middle_tick:
           tick.tick2On = False     
           tick.tick1On = False     
           tick.label2On = False     
           tick.label1On = False     
    
    
def disable_xticks(ax):
    for t in ax.xaxis.get_ticklines(): t.set_visible(False)
    
def disable_yticks(ax):
    for t in ax.yaxis.get_ticklines(): t.set_visible(False)

def remove_x_tick_labels():
    pylab.xticks([],[])  
 
def remove_y_tick_labels(): 
    pylab.yticks([],[])  
