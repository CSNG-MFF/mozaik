visualization Package
=====================

This package contains the definition of the visualization API, together with implementation 
of numerous plotting methods that take advantage of the *mozaik* data structures to automatize a 
lot of the visualization functionality.

The philoshophy of the visualization package is similar to the analysis package :mod:`mozaik.analysis.analysis`.
The input of each plotting is a datastore, it is the role of the plotting to extract as 
much as possible data from the provided datastore and display as much as possible information
(of course with respect to the role of the given plotting class). It is then up to the 
user to manipulate the datastore before passing it to the plotting function to 
restrict the information that is being plotted. Together with the querying system
this provides a very flexible and unified framework for defining what users want 
to plot.

The implementation is based on matplotlib and its GridSpec
objects. The plotting framwork is divided into two main concepts, represented by the two
high-level classes Plotting :class:`mozaik.visualization.plotting.Plotting` and :class:`mozaik.visualization.simple_plot.SimplePlot`.

The `SimplePlot` represent low-level plotting. It is assumed that this plot has
only a single axis that is drawn into the region defined by the GridSpec
instance that is passed into it. The role of the set of classes derived from
SimplePlot is to standardize the low level looks of all figures (mainly related
to axis, lables, titles etc.), and should assume data given to them in a format
that is easy to use by the given plot. In order to unify the look of figures it
defines four functions pre_axis_plot, pre_plot, plot, and post_plot. The actual
plotting that user defines is typically defined in the plot function while the
pre_axis_plot, pre_plot and post_plot functions handle the pre and post
plotting adjustments to the plot (i.e. typical post_plot function for
example adjusts the ticks of the axis to a common format other such axis
related properties). When defining a new `SimplePlot` function user is encoureged
to push as much of it's 'decorating' funcitonality into the post and pre plot
function and define only the absolute minimum in the plot function. At the same
time, there is already a set of classes implementing a general common look
provided, and so users are encouraged to use these as much as possible. If
their formatting features are not sufficient or incompatible with a given plot,
users are encoureged to define new virtual class that defines the formatting in
the pre and post plot functions (and thus sepparating it from the plot itself),
and incorporating these as low as possible within the hierarchy of the
`SimplePlot` classes to re-use as much of the previous work as possible.


The `Plotting` class (and its children) define the high level plotting
mechanisms. They are mainly responsible for hierarchical organization of
figures with multiple plots, any mechanisms that require consideration of
several plots at the same time, and the translation of the data form the general
format provided by `Datastore`, to specific format that the `SimplePlot` plots
require. In general the Plotting instances should not do any plotting of axes
themselves (but instead calling the `SimplePlot` instances to do the actual
plotting), with the exception of multi-axis figures whith complicated inter-axis
dependencies, for which it is not practical to break them down into single
`SimplePlot` instances.


:mod:`plotting` Module
----------------------

.. automodule:: mozaik.visualization.plotting
    :members:
    :show-inheritance:


:mod:`simple_plot` Module
-------------------------

.. automodule:: mozaik.visualization.simple_plot
    :members:
    :show-inheritance:

:mod:`plot_constructors` Module
-------------------------------

.. automodule:: mozaik.visualization.plot_constructors
    :members:
    :show-inheritance:


:mod:`helper_functions` Module
------------------------------

.. automodule:: mozaik.visualization.helper_functions
    :members:
    :show-inheritance:


:mod:`misc` Module
------------------

.. automodule:: mozaik.visualization.misc
    :members:
    :show-inheritance: