connector Package
=================

Modules
^^^^^^^

    * :py:mod:`mozaik.connectors.connector` - This module provides the Interface and implementations for flexible synaptic connection strategies between neural populations 
    * :py:mod:`mozaik.connectors.fast` - This module provides high-performance mozaik connectors using backend-optimized PyNN methods but trading off flexibility for speed in establishing neural connections.
    * :py:mod:`mozaik.connectors.modular` - This module provides a flexible, extensible connector framework that supports modular composition of weight and delay functions, probabilistic sampling, and local connectivity constraints.
    * :py:mod:`mozaik.connectors.modular_connector_functions` - This module is a collection of modular connector function classes that define customizable weight, delay, and connection sampling logic based on distance or distributions for use with modular connectors.
    * :py:mod:`mozaik.connectors.vision` - This module defines a set of biologically-inspired connector functions that define visual cortex-specific connectivity patterns based on retinotopic maps, Gabor filters, push-pull organization, and receptive field correlations.
    * :py:mod:`mozaik.connectors.meta_connectors` - This module contains high-level meta-connectors for constructing complex biologically-inspired projections, such as Gabor-based LGN-to-V1 connectivity, using parameterized distributions and cortical feature maps.





:mod:`connectors` Module
------------------------

.. automodule:: mozaik.connectors.__init__
    :members:
    :undoc-members:
    :show-inheritance:


:mod:`connectors.fast` Module
-----------------------------

.. automodule:: mozaik.connectors.fast
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`connectors.modular` Module
--------------------------------

.. automodule:: mozaik.connectors.modular
    :members:
    :undoc-members:
    :show-inheritance:


:mod:`connectors.modular_connector_functions` Module
----------------------------------------------------

.. automodule:: mozaik.connectors.modular_connector_functions
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`connectors.vision` Module
-------------------------------

.. automodule:: mozaik.connectors.vision
    :members:
    :undoc-members:
    :show-inheritance:


:mod:`mozaik.connectors.meta_connectors` Package
------------------------------------------------

.. automodule:: mozaik.connectors.meta_connectors
    :members:
    :undoc-members:
    :show-inheritance:
