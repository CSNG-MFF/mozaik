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

*mozaik* is currently subdivided into the core package and 10 subpackages:
    
    * :doc:`mozaik` - contains the core of the *mozaik*:
	  * core - The core API objects
      * controller - the control center of *mozaik* workflows
      * space - the input space handling
      
    * :doc:`mozaik.sheets` - Code defining 2D sheets of neurons, one of the basic building blocks of *mozaik* networks
    * :doc:`mozaik.connectors` - Defines various connections between sheets
    * :doc:`mozaik.experiments` - Defines the experiment interface
    * :doc:`mozaik.models` - code encapsulating a *mozaik* network consisting of sheets and connectors and providing
                             an 'I/O' interface to it. Currently it also contains retinal models (these might be reorganized in future).
    * :doc:`mozaik.stimuli` - definition of stimulus interfaces
    * :doc:`mozaik.storage` - data storage and data querying code
    * :doc:`mozaik.analysis` - analysis code
    * :doc:`mozaik.visualization` - plotting code
    * :doc:`mozaik.meta_workflow` - code supporting meta-workflows, such as parameter searches
    * :doc:`mozaik.tools` - utility code
    
    

This might change as *mozaik* grows, and code in the framework 
is likely to get separated into new sub-packages as it matures.

Scetch of how the control flows between the *mozaik* elements:

.. image:: mozaik_control_flow.png
   :width: 800px


