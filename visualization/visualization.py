# the visualization for mozaik objects
# it is based on the matplotlib 
# one important concept the visualization code should follow is that it should not
# itself call figure() or subplot() commands, instead assume that they were already
# called before it. This allows for a simple encapsulation of the individual figures 
# into more complex matplotlib figures.

import pylab

def plot_layer_activity(sheet,value_to_plot,cortical_coordinates=False,labels=True):
    """
    This function creates a scatter plot, where each point corresponds to a neuron
    (in cortical or visual space coordinates) and color of each point corresponds to
    the values_to_plot.
    
    sheet - an instance of the Sheet class
    value_to_plot - an list of numbers whose length corresponds to the number of neurons in sheet
    cortical_coordinates - if true plotted in cortical coordinates, otherwise in degrees of visual field
    labels - whether to include labels
    """
    
    if cortical_coordinates:
       # first we need to check whether sheet is instance of SheetWithMagnificationFactor or rather whether it has
       # the property magnification_factor
       if hasattr(sheet, 'magnification_factor'):
	    pylab.scatter(sheet.pop.positions[0]*sheet.magnification_factor,sheet.pop.positions[1]*sheet.magnification_factor,c=value_to_plot, faceted = False,edgecolors='none') 
	    if labels:
        	    pylab.xlabel('x (μm)')
		    pylab.ylabel('y (μm)')
       
    else:
       pylab.scatter(sheet.pop.positions[0],sheet.pop.positions[1],c=value_to_plot, faceted = False,edgecolors='none') 
       if labels:
           pylab.xlabel('x (° of visual field)')
	   pylab.ylabel('y (° of visual field)')







def plot_connection_field(neuron):
    """ 
    will plot the connection of pyNN neuron
    """

    
