# coding: utf-8
"""
Definition of the component interfaces. These interfaces are not currently directly checked or enforced.
"""

from mozaik import __version__
from parameters import ParameterSet, ParameterDist
import parameters.random
from parameters.random import UniformDist
import mozaik
from mozaik.tools.distribution_parametrization import PyNNDistribution
from string import Template


logger = mozaik.getMozaikLogger()


class ParametrizedObject(object):
    """
    Base class for for all Mozaik objects using the dynamic parameterization framework. See `getting_started`_ for more details.
    
    Parameters
    ----------
    parameters : dict
               Dictionary of the parameter names and their values that has to match the required_parameters variable. 
    """
    required_parameters = ParameterSet({})
    version = __version__

    def check_parameters(self, parameters):
        """
        This is a function that checks whether all required (and no other) parameters have been specified and all their values have matching types.
        This function gets automatically executed during initialization of each :class:.ParametrizedObject object. 

        Parameters
        ----------
        parameters : dict
                   Dictionary of the parameter names and their values that has to match the required_parameters variable. 
        """
        
        def walk(tP, P, section=None):
            if set(tP.keys()) != set(P.keys()):
                raise KeyError("Invalid parameters for %s.%s Required: %s. Supplied: %s. Difference: %s" % (self.__class__.__name__, section or '', tP.keys(), P.keys(), set(tP.keys()) ^ set(P.keys())))
            for k, v in tP.items():
                if isinstance(v,ParameterSet):
                    if P[k] != None:
                        assert isinstance(P[k], ParameterSet), "Type mismatch for parameter %s: %s !=  ParameterSet, for %s " % (k, type(P[k]), P[k])
                        walk(v, P[k], section=k)
                elif v == PyNNDistribution:
                     # We will allow for parameters requiring PyNN Distirbution to also fall back to single value - this is compatible with PyNN
                     if not (isinstance(P[k],int) or isinstance(P[k],float)):
                        assert isinstance(P[k], PyNNDistribution), "Type mismatch for parameter %s: %s != %s " % (k, PyNNDistribution, P[k])
                elif v == ParameterDist:
                     # We will allow for parameters requiring ParameterDist to also give a scalar value, in which case we will change it to UniformDist
                     # with minimum and maximum equal to the scalar value.
                     if not (isinstance(P[k],int) or isinstance(P[k],float)):
                        assert isinstance(P[k], ParameterDist), "Type mismatch for parameter %s: %s != %s " % (k, ParameterDist, P[k])
                     else:
                        P[k] = UniformDist(min=P[k], max=P[k])
                else:
                    assert isinstance(P[k], v) or (v == ParameterSet and P[k] == None) or (v == float and isinstance(P[k],int)) or (v == int and isinstance(P[k],float)), "Type mismatch for parameter %s: %s != %s " % (k, v, P[k])
        try:
            # we first need to collect the required parameters from all the classes along the parent path
            new_param_dict = {}
            for cls in self.__class__.__mro__:
            # some parents might not define required_parameters
            # if they do not require one or they are the object class
                if hasattr(cls, 'required_parameters'):
                    new_param_dict.update(cls.required_parameters.as_dict())
            walk(ParameterSet(new_param_dict), parameters)
        except AssertionError as err:
            raise Exception("%s\nInvalid parameters.\nNeed %s\nSupplied %s" % (
                                err, ParameterSet(new_param_dict), parameters))


    def __init__(self, parameters):
            self.check_parameters(parameters)
            self.parameters = parameters


class BaseComponent(ParametrizedObject):
    """
    Base class for mozaik model components.
    
    Parameters
    ----------
    model : Model
          Reference to the model to which the component will belong.
    """

    def __init__(self, model, parameters):
        ParametrizedObject.__init__(self, parameters)
        self.model = model


class SensoryInputComponent(BaseComponent):
    """
    Abstract API of sensory input component. Each mozaik sensory input component should 
    inherit from this class and implement its two abstrac methods.
    
    See Also
    --------
    mozaik.models.vision : the implementation of retinal input 
    """

    def process_input(self, input_space, stimulus_id, duration=None,
                             offset=0):
        """
        This method is responsible for presenting the content of input_space
        to the sensory input component, and all the mechanisms that are responsible to
        passing the output of the retina (in whatever form desired) to the Sheet
        objects that are connected to it and thus represent the interface
        between the input space component and the rest of the model.

        The method should return the sensory input that has been effectivitly presented to 
        the model, currently the format of it is not specified.
        """
        raise NotImplementedError

    def provide_null_input(self, input_space, duration=None, offset=0):
        """
        This method is responsible generating sensory input in the case of no
        stimulus. This method should correspond to the special case of
        process_input method where the input_space contains 'zero'
        input. This methods exists for optimization purposes as the 'zero' input
        is presented often due to it's presentation between different sensory
        stimuli to allow for models to return to spontaneous activity state.
        """
        raise NotImplementedError
