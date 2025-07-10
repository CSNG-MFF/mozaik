stimuli Package
===============

This sub-package contains modules that are related to definition and manipulation of stimuli.

Modules
^^^^^^^    
    * :py:mod:`mozaik.stimuli.__init__` - contains definition of the stimulus API 
    * :py:mod:`mozaik.stimuli.vision.visual_stimulus` - contains definitions of the visual stimulus API, and implements some of the functions specified in the stimulus interface
    * :py:mod:`mozaik.stimuli.vision.topographica_based` - contains implementation of number of visual stimuli, using the `Topographica <http://topographica.org/>`_ pattern generation package.
    * :py:mod:`mozaik.stimuli.vision.texture_based` - Generates visual stimuli by synthesizing textures and controlling pixel-level statistics using image-based and algorithmic methods for vision experiments.

See the :py:mod:`mozaik.stimuli.stimulus` for more general information on stimulus handling in *mozaik*.

:mod:`stimuli` Module
---------------------

.. automodule:: mozaik.stimuli.__init__
    :members:
    :show-inheritance:


Subpackages
-----------

.. toctree::
    mozaik.stimuli.vision



