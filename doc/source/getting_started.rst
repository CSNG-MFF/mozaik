Getting started
===============

Here we will discuss several general topics that the user should be familiar before starting to use *mozaik*.


Parametrization
~~~~~~~~~~~~~~~

There are two systems used to parametrize objects throughout mozaik, each with its own role.



* ParemterSet - This system is used to parametrize objects for which the parameters are typically loaded from configuration files, and for which
                we want to enforce typing. For more details. See :ref:`parametersetsection` section for more details.

* MozaikParametrizedObject - For objects that are stored in large numbers and for which we need to have a flexible way to sort between them based on their parameters
                             we use the :class:`.mozaik.tools.mozaik_parametrized.MozaikParametrizedObject` interface. See :ref:`mposection` section for more details.


.. _mposection:

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
not support and other parameter sections, in future we would like to extend numpydoc to recongnice special 'Required parameters' section).

Throughout the documentation we will refer to this parameterization scheme as the 'required parameters` or RP.


MozaikParametrizedObject
------------------------

There are several situations in mozaik where we deal with large numbers of objects, each uniquely identified by a (potentiall different) set 
of parameters. Often we want to refer to subsets of such set of objects based on combinations of their parameter values. In mozaik this happens
when we deal with stimuli, with recordings and with analysis data structures. To facilitate a common handling of these cases we define a class 
:class:`.mozaik.tools.mozaik_parametrized.MozaikParametrizedObject`, which represents a extension of this parameter package: `param <http://ioam.github.io/param/>`_.

We restrict the types of parameters that can be used with :class:`.mozaik.tools.mozaik_parametrized.MozaikParametrizedObject` 
class to those defined in :mod:`.mozaik.tools.mozaik_parametrized` module: :class:`.mozaik.tools.mozaik_parametrized.SNumber`, :class:`.mozaik.tools.mozaik_parametrized.SInteger` and :class:`.mozaik.tools.mozaik_parametrized.SString`,
representing a floating number, integer or a string respectively. On top of the properties inherited from identical parameters of the `param <http://ioam.github.io/param/>`_ package we 
allow for definition of period for the SNumber and SInteger parameter types, and units for the SNumber parameter. 
If period is set, it declares the parameter to be periodic with the given period. The units declare the units in which the parameter value is given.

The :mod:`.mozaik.tools.mozaik_parametrized` module containes number of methods that allow for powerfull filtering of sets of *MozaikParametrizedObject*
objects. These methods are primarily used the the :mod:`.mozail.storage.queries` package. 

The user will encounter the :class:`.mozaik.tools.mozaik_parametrized.MozaikParametrizedObject` class if he wants to define a new Stimulus or a
new AnalysisDataStructure class. In this case it has to derive the new class from :class:`.mozaik.tools.mozaik_parametrized.MozaikParametrizedObject`
and declare all parameters that will identify the object using the three parameter types declared in :mod:`.mozaik.tools.mozaik_parametrized` module.

.. _parametersetsection:


Common abreviations
-------------------

Throughout the documentation we use several common abbreviations

* DSV - Data Store View (see :class:`mozaik.storage.datastore.DataStoreView`)
* ADS - Analysis Data Structure (see :mod:`mozaik.analysis.data_structures`)
* PNV - Per Neuron Value analysis data structure (see :class:`mozaik.analysis.data_structures.PerNeuronValue`)
* RP (or required parameters) - The required parameters parametrization scheme (see ParameterSet section above)


