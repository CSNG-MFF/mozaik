# coding: utf-8
"""
Definition of the component interfaces. These interfaces are not currently
checked or enforced.
"""
from mozaik import __version__
from NeuroTools.parameters import ParameterSet
from NeuroTools.signals.spikes import SpikeList
import mozaik
import os
from string import Template
import numpy
from mozaik.tools.mozaik_parametrized import SNumber
from mozaik.tools.units import lux
import quantities as qt

logger = mozaik.getMozaikLogger("Mozaik")

 
class MozaikParametrizeObject(object):
    """Base class for for all MozailLite objects using the dynamic parametrization framwork."""
    
    required_parameters = ParameterSet({})
    version = __version__

    def check_parameters(self, parameters):
            def walk(tP, P, section=None):
                if set(tP.keys()) != set(P.keys()):
                    raise KeyError("Invalid parameters for %s.%s Required: %s. Supplied: %s. Difference: %s" % (self.__class__.__name__, section or '', tP.keys(), P.keys(), set(tP.keys()) ^ set(P.keys())))
                for k,v in tP.items():
                    if isinstance(v, ParameterSet):
                        if P[k] != None:
                            assert isinstance(P[k], ParameterSet), "Type mismatch for parameter %s: %s !=  ParameterSet, for %s " % (k,type(P[k]),P[k]) 
                            walk(v, P[k],section=k)
                    else:
                        assert isinstance(P[k], v), "Type mismatch for parameter %s: %s !=  ParameterSet, for %s " % (k,type(P[k]),P[k]) 
            try:
                # we first need to collect the required parameters from all the classes along the parent path
                new_param_dict={}
                for cls in self.__class__.__mro__:
                # some parents might not define required_parameters 
                # if they do not require one or they are the object class
                    if hasattr(cls, 'required_parameters'):
                        new_param_dict.update(cls.required_parameters.as_dict())
                walk(ParameterSet(new_param_dict), parameters)
            except AssertionError as err:
                raise Exception("%s\nInvalid parameters.\nNeed %s\nSupplied %s" % (err,ParameterSet(new_param_dict),
                                                                               parameters))  
                                                                               
                                                                               
    def __init__(self, parameters):
            """
            """
            self.check_parameters(parameters)
            self.parameters = parameters
                                                                                   
class MozaikComponent(MozaikParametrizeObject):
    """Base class for visual system components and connectors."""
    
    def __init__(self, model, parameters):
        """
        """
        MozaikParametrizeObject.__init__(self, parameters)
        self.model = model

class MozaikRetina(MozaikComponent):
      
      def process_visual_input(self, visual_space, stimulus_id,duration=None, offset=0):  
          """
          This method is responsible for presenting the content of visual_space
          the retina it represents, and all the mechanisms that are responsible to
          passing the output of the retina (in whatever form desired) to the Sheet objects
          that are connected to it and thus represent the interface between the 
          retina and the rest of the model.
          
          The method should return the list of 2d numpy arrays containing the 
          raw frames of the  visual input to the retina.
          """
          raise NotImplementedError
          pass
        
      def provide_null_input(self, visual_space, duration=None, offset=0):  
          """
          This method is responsible generating retinal input in the case of no visual stimulus.
          This method should correspond to the special case of process_visual_input method where
          the visual_space contains 'zero' input. This methods exists for optimization purposes
          as the 'zero' input is presented often due to it's presentation between different visual
          stimuli to allow for models to return to spontaneous activity state.
          """
          raise NotImplementedError
          pass

class Connector(MozaikComponent):
    """Base class for objects that connect mozaik sheets."""
    version = __version__
    
    def __init__(self, model, name,source, target, parameters):
        logger.info("Creating %s between %s and %s" % (self.__class__.__name__,
                                                       source.__class__.__name__,
                                                       target.__class__.__name__))
        MozaikComponent.__init__(self, model,parameters)
        self.name = name
        self.model.register_connector(self)
        self.sim = self.model.sim
        self.source = source
        self.target = target
        self.input = source
        self.target.input = self
        
        
    def describe(self, template='default', render=lambda t,c: Template(t).safe_substitute(c)):
        context = {
            'name': self.__class__.__name__,
            'source': {
                'name': self.source.__class__.__name__,
            },
            'target': {
                'name': self.target.__class__.__name__,
            },
            'projections': [prj.describe(template=None) for prj in self.projections]
        }
        if template:
            render(template, context)
        else:
            return context
        
  
