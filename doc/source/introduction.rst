Introduction
============

*mozaik* is an integrated workflow framework for large scale neural simulations, intended
to relieve users from writing boilerplate code for projects involving complex heterogenous 
neural network models, complex stimulation and experimental protocols and subsequent analysis and plotting.

It is built on top of the following tools:

    * `PyNN <http://neuralensemble.org/PyNN/>`_ (for simulator independent neural network model definition)
    * `neo  <http://pythonhosted.org/neo/Neo>`_  (for exchange and internal representation of data)
    * `matplotib <http://matplotlib.org/>`_ (for plotting)


*Mozaik* currently covers the following main areas of the neural simulation workflow:
    
    * High-level components for definition of topologically organized spiking networks (built on top of `PyNN <http://neuralensemble.org/PyNN/>`_)
    * Experiment control (description and execution of experiments)
    * Stimulus definition framework
    * Data storage (storage of recordings and analysis results)
    * Data manipulation (a query based system for performing high-level filtering operations over the datastore)
    * Analysis module
    * Plotting module

Tasks that we would like to cover in *mozaik* in future are:
    
    * Support of FACETS-like benchmarks within the experiment control framework
    * Versioning system built on top of Sumatra
    * GUI for recording and analysis browsing and plotting

*mozaik* is currently subdivided into 7 sub-packages:
    
    * :doc:`mozaik.framework` - contains the core of the *mozaik* package:
	
        * sheets - Code defining 2D sheets of neurons, one of the basic building blocks of *Mozaik* networks
        * connectors - Defines various connections between sheets
        * experiment - Defines the experiment interface
        * experiment_controller - the control center of *mozaik* workflows
        * interfaces - collection of various basic *mozaik* interfaces
        * space - visual space handling

    * :doc:`mozaik.models` - code encapsulating a *mozaik* network consisting of sheets and connectors and providing
                                 an 'I/O' interface to it. Currently it also contains retinal models (these might be reorganized in future).
    * :doc:`mozaik.stimuli` - definition of stimulus interfaces
    * :doc:`mozaik.storage` - data storage and data querying code
    * :doc:`mozaik.analysis` - analysis code
    * :doc:`mozaik.visualization` - plotting code
    

This might change as *mozaik* grows, and code in the framework 
is likely to get separated into new sub-packages as it matures.

Scetch of how the control flows between the *mozaik* elements:

.. image:: mozaik_control_flow.png
   :width: 800px


