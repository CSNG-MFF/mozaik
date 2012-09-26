Introduction
============

*Mozaik* is an integrated workflow framework for large scale neural simulations.
Note that currently it is under intense development, and anything can change
on a daily basis.

It is built on top of the following tools:
    

    * PyNN (for definition and running of simulations)
    * Neo  (for exchange and internal representation of data)
    * matplotib (for plotting)


*Mozaik* currently covers the following main areas of the neural simulation workflow:
    
    * High-level components for definition of topologically organized spiking networks (built on top of PyNN_)
    * Experiment control (description and execution of experiments, very basic so far)
    * Stimulus definition framework
    * Data storage (storage of recordings and analysis results)
    * Data manipulation (a query based system for performing high-level filtering operations over the datastore)
    * Analysis module
    * Plotting module

Tasks that we would like to cover in *Mozaik* in future (any contributors welcome :-)!!!):
    
    * Support of FACETS-like benchmarks within the experiment control framework
    * Versioning system built on top of Sumatra_
    * GUI for recording and analysis browsing and plotting
    * Caching

*Mozaik* is currently subdivided into 7 sub-packages:
    
    * :doc:`MozaikLite.framework` - contains the core of the *mozaik* package:
	
        * sheets - Code defining 2D sheets of neurons, one of the basic building blocks of *Mozaik* networks
        * connectors - Defines various connections between sheets
        * experiment - Defines the experiment interface
        * experiment_controller - the control center of *Mozaik* workflows
        * interfaces - collection of various basic *Mozaik* interfaces
        * space - visual space handling

    * :doc:`MozaikLite.models` - code encapsulating a *Mozaik* network consisting of sheets and connectors and providing
                                 an 'I/O' interface to it. Currently it also contains retinal models (these might be reorganized in future).
    * :doc:`MozaikLite.stimuli` - definition of stimulus interfaces
    * :doc:`MozaikLite.storage` - data storage and data querying code
    * :doc:`MozaikLite.analysis` - analysis code
    * :doc:`MozaikLite.visualization` - plotting code
    

This might change as *Mozaik* grows, and especially code in the framework 
is likely to get separated into new sub-packages as it matures.

The following diagram offers an intuition of how the control flows between the *Mozaik* elements:


.. image:: mozaik_control_flow.png
   :width: 800px


.. _PyNN: http://neuralensemble.org/PyNN/
.. _Sumatra: http://neuralensemble.org/sumatra/