Getting started
===============


ParameterSet
------------

In many of its modules *mozaik* uses `ParameterSet <https://github.com/apdavison/parameters?source=cr>`_  package to parametrize the classes. 
The advantage of this system is that it allows for easy import of parameters for configuration files, automatic type checking, and offers advanced parameter
manipulation methods. For example the model specification modules are entirely written to use this system, but also the :mod:`mozaik.analysis`, :mod:`mozaik.plotting` and :mod:`mozaik.storage.queries` package 
use it extensively. 

Each class in *mozaik* parameterized via the `ParameterSet` module will be derived from the :class:`mozaik.framework.interfaces.MozaikParametrizeObject`
and will specify a required_parameters dictionary atributed that will hold the names of the required parameters as keys and their required types as
values. Each such class will accept a `parameters` argument in its constructor that has to be set to a dictionary containing the parameter names as values
and the parameter values as the values. The presence of all the required parameters and the matching of their types with the supplied 
values will be automatically checked, and a exception will be raised if the supplied parameters and required_parameters dictionary do not match
in parameter names or types. The required_parameters attributes specified along the inheritance hierarchy are concatentated, so user does not 
have to specify parameters that have already been specified by the base class again, but also it means that derived classes cannot ignore or delete
parameters specified in their parents required_parameters attribute.

The ParameterSet instances can be nested.  It is also possible to use  `ParameterSet` as an parameter type in the reuiqred_parameters dictionary. In this one exceptional 
case the supplied parameter type does not have to match the required_parameters values, as it can be set to None. 

The *mozaik* documentation quidelines stipulate that the parameters in the required_parameters attribute are documented via the standard 
numpy parameter syntax in the numpydoc 'Other parameters' section and this section should not be used otherwise (this is a workaround as numpydoc does
not support and other parameter sections, in future we would like to extend numpy doc to recongnice special 'Required parameters' section).

Common abreviations
-------------------

Throughout the documentation we use several common abbreviations

* DSV - Data Store View (see :class:`mozaik.storage.datastore.DataStoreView`)
* ADS - Analysis Data Structure (see :mod:`mozaik.analysis.analysis_data_structures`)
* PNV - Per Neuron Value analysis data structure (see :class:`mozaik.analysis.analysis_data_structures.PerNeuronValue`)
