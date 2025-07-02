# encoding: utf-8
"""
This module contains visualization code not conforming the Plotting framework
defined in the mozaik.visualization.plotting module.

Most of this code is likely being used as a debugging visualization tools or is
generic visualization tools that can in turn be used by plotting algorithms.
"""

import pylab


def plot_layer_activity(sheet, value_to_plot, cortical_coordinates=False,
                        labels=True):
    """
    This function creates a scatter plot, where each point corresponds to a
    neuron (in cortical or visual space coordinates) and color of each point
    corresponds to the values_to_plot.
    
    Parameters
    ----------
    
    sheet : :class:`mozaik.sheets.Sheet`
        An instance of the Sheet class
          
    value_to_plot : list
        An list of numbers whose length corresponds to the number of neurons in sheet
                  
    cortical_coordinates : bool
        If true plotted in cortical coordinates, otherwise in degrees of visual field
                         
    labels : bool
        Whether to include labels.
           
    """

    if cortical_coordinates:
        # first we need to check whether sheet is instance of
        # SheetWithMagnificationFactor or rather whether it has the property
        # magnification_factor
        if hasattr(sheet, 'magnification_factor'):
            pylab.scatter(sheet.pop.positions[0] * sheet.magnification_factor,
                          sheet.pop.positions[1] * sheet.magnification_factor,
                          c=value_to_plot, faceted=False, edgecolors='none')
            if labels:
                pylab.xlabel(u'x (μm)')
                pylab.ylabel(u'y (μm)')
    else:
        pylab.scatter(sheet.pop.positions[0], sheet.pop.positions[1],
                      c=value_to_plot, faceted=False, edgecolors='none')
        if labels:
            pylab.xlabel(u'x (° of visual field)')
            pylab.ylabel(u'y (° of visual field)')
