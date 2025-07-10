mozaik API
==========

Mozaik is a modular workflow system for building, simulating, and analyzing large-scale spiking neural network models. It provides a unified interface that coordinates various tools like NEST, PyNN, and imaging libraries, enabling reproducible and scalable simulation workflows.

The top-level ``mozaik`` package is intentionally minimal. Most functionality is organized into subpackages, each responsible for a different stage of the modeling pipeline:

**Subpackages**

- ``mozaik.analysis`` — Analysis of simulation results
- ``mozaik.connector`` — Connection generation between sheets
- ``mozaik.experiments`` — Definition and control of experimental protocols
- ``mozaik.sheets`` — Sheet-based architecture and components
- ``mozaik.models`` — Model construction logic
- ``mozaik.stimuli`` — Stimulus generation and management
- ``mozaik.storage`` — Data storage and retrieval
- ``mozaik.tools`` — Utility functions
- ``mozaik.visualization`` — Visualization of results
- ``mozaik.meta-workflow`` — Meta-workflow orchestration

**mozaik Package Overview**

The root ``mozaik`` package provides essential infrastructure:

- Global random number generators (`rng`, `pynn_rng`)
- MPI communication object (`mpi_comm`)
- Workflow utilities for setting up and running simulations
- Component loading and logging

It also includes the `setup_mpi`, `get_seeds`, `getMozaikLogger`, and `load_component` functions that help manage reproducibility and inter-process coordination.

As Mozaik evolves, subpackages may be added or deprecated as coordination with external tools improves.

For details on each module, browse the documentation of the respective subpackages listed above.


Subpackages
-----------

.. toctree::
    :maxdepth: 1
   
    mozaik.analysis
    mozaik.connectors
    mozaik.experiments
    mozaik.sheets
    mozaik.models
    mozaik.stimuli
    mozaik.storage
    mozaik.tools
    mozaik.visualization
    mozaik.meta_workflow


:mod:`mozaik` Package
---------------------

.. automodule:: mozaik.__init__
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`mozaik.core` Package
--------------------------

.. automodule:: mozaik.core
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`space` Module
-------------------

.. automodule:: mozaik.space
    :members:
    :show-inheritance:

:mod:`controller` Module
------------------------

.. automodule:: mozaik.controller
    :members:
    :show-inheritance:

:mod:`cli` Module
-----------------

.. automodule:: mozaik.cli
    :members:
    :show-inheritance:
