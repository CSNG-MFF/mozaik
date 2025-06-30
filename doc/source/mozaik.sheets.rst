sheets Package
==============

This package contains the API defining the basic building block
of *mozaik* models - sheets - and related concepts.

Module
^^^^^^
    * :py:mod:`mozaik.sheets.sheet` - This module provides spatially-organized neural sheets as core Mozaik model components, integrating PyNN populations with spatial structure, stimulation, and recording support.
    * :py:mod:`mozaik.sheets.vision` - This module provides vision-specific neural sheets modeling retinal and cortical areas, including support for visual field coordinates, magnification factors, and 3D cortical structure.
    * :py:mod:`mozaik.sheets.population_selector` - This module defines the PopulationSelector API and implementations for selecting neuron subpopulations within Sheets based on spatial, random, or annotation-based criteria.
    * :py:mod:`mozaik.sheets.direct_stimulator` - This module provides APIs and classes for direct artificial stimulation of neuron populations in cortical sheets, enabling injection of spikes, currents, or patterned stimulations at the population level during simulations.

:mod:`sheets` Module
--------------------

.. automodule:: mozaik.sheets.__init__
    :members:
    :undoc-members:
    :show-inheritance:


:mod:`sheets.vision` Module
---------------------------

.. automodule:: mozaik.sheets.vision
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`sheets.population_selector` Module
----------------------------------------

.. automodule:: mozaik.sheets.population_selector
    :members:
    :undoc-members:
    :show-inheritance:



:mod:`sheets.direct_stimulator` Module
--------------------------------------

.. automodule:: mozaik.sheets.direct_stimulator
    :members:
    :undoc-members:
    :show-inheritance:


