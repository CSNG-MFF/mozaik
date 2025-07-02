models Package
==============

This package defines the API for model definition in *mozaik*, and the `mozaik.models.retinal` subpackage
contains implementations for models of retina.

Modules
^^^^^^^

    * :py:mod:`mozaik.models.model` - This module defines the base Model class that defines the structure, parameters, and control logic for Mozaik neural network simulations.
    * :py:mod:`mozaik.models.vision` - This contains two sub-module which provides spatiotemporal receptive field models for retinal/LGN input, including a Cai et al. (1997)-based filter (cai97) and a parameterized, gain-controlled, cacheable filter (spatiotemporalfilter).
  


:mod:`models` Module
--------------------

.. automodule:: mozaik.models.__init__
    :members:
    :undoc-members:
    :show-inheritance:

Subpackages
-----------

.. toctree::
    mozaik.models.vision

