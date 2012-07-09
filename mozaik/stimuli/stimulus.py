"""
This package defines the API for:
    - implementation of stimuli as input to models (see class Stimulus)
    - identification of stimulus identity (without the actual 'data' of the stimulus) throughout *mozaik* (see class StimulusID)
      (NOTE: throughout most mozaik (particularly all post processing i.e. analysis and plotting) user will interact with StimulusIDs rather then Stimulus instances!!)
    - function helpers for common manipulation with collections of stimuli (or rather their IDs)

Each stimulus is expected to have a dictionary of parameters which have to uniquely identify the stimulus.
We implement this by using the Parametrized package that allows to specify parameters with the all above requirements.
For Stimuli objects we will allow only SNumber, SInteger and SString parameters. These extend the
corresponding parametrized parameters to allow specification of units (see tools/mozaik_parametrized.py).

Note that *all* such parameters defined in the class (and its ancestors) will be considered as parameters of the Stimulus.
"""

import quantities as qt
import numpy 
from operator import * # don't do import *
from mozaik.tools.mozaik_parametrized import * # don't do import *
from mozaik.framework.interfaces import MozaikParametrizeObject
import inspect
from NeuroTools.parameters import ParameterSet


class StimulusID():
      """
      StimulusID is a light object that contains all the parameter info of a given Stimulus object.
      There is one to one mapping between StimuluID instances and Stimulus instances.
      
      The main purpose for StimulusID is to stand for Stimulus whenever the stimulus identitity is required
      without the need to move around the heavy Stimulus objects.
      
      Unlike Stimulus, StimulusID allows any parameter to be also assigned None. This is often 
      used in data analysis to mark parameters that has been 'computed out' by the analysis (such as averaged out).
      """  

      def __str__(self):
            """
            Saves the parameter names and values as a dict
            """
            settings = ['\"%s\":%s' % (name,repr(val))
                    for name,val in self.get_param_values()]
            
            r= "{ \"name\" :" + "\"" + self.name + "\""+ "," + "\"module_path\" :" + "\"" + self.module_path + "\"" +',' + ", ".join(settings) + "}"
            return r

      def __eq__(self, other):
            return (self.name == other.name) and (self.get_param_values() == other.get_param_values())
            
      def get_param_values(self):
          z = self.params.items()
          z.sort(key=itemgetter(0))
          return z
          
      def number_of_parameters(self):
          return len(self.params.keys())
          
      def load_stimulus(self):
          cls = self.getStimulusClass()
          return cls(**self.params)  
      
      def getStimulusClass(self):
          z = __import__(self.module_path,globals(),locals(),self.name)

          return  getattr(z,self.name)
          
      def __init__(self,obj):
          self.units = {}
          self.params = {}          
          
          if isinstance(obj,Stimulus):
              self.name = obj.__class__.__name__
              self.module_path = inspect.getmodule(obj).__name__
              par = obj.params()
              for n,v in obj.get_param_values():
                  if n != 'name' and n != 'print_level':
                      self.params[n] = v
                      self.units[n] = par[n].units
          elif isinstance(obj,dict):
              self.name = obj.pop("name")
              self.module_path = obj.pop("module_path")
              par = self.getStimulusClass().params()
              for n,v in obj.items():
                  self.params[n] = v
                  self.units[n] = par[n].units
          elif isinstance(obj,str):
               d = eval(obj)
               self.name = d.pop("name")
               self.module_path = d.pop("module_path")
               par = self.getStimulusClass().params()
               for n,v in d.items():
                  self.params[n] = v
                  self.units[n] = par[n].units
          else:
              raise ValueError("obj is not of recognized type (recognized: str,dict or Stimulus)")

class Stimulus(MozaikParametrized):
    """
    Abstract class.
    """
    frame_duration = SNumber(qt.ms,doc="""The duration of single frame""")
    duration = SNumber(qt.ms,doc="""The duration of stimulus""")
    trial = SInteger(doc="""The duration of stimulus""")

    def __init__(self, **params):
        MozaikParametrized.__init__(self,**params)
        self.input = None
#        self._frames = None
        self._frames = self.frames()
        self.n_frames = numpy.inf # possibly very dangerous. Don't do 'for i in range(stim.n_frames)'!
        self.update()

    def __str__(self):
        return str(StimulusID(self))

    def __eq__(self, other):
        return self.equalParams(other) and (self.__class__ == other.__class__)

    def frames(self):
        """
        Return a generator which yields the frames of the stimulus in sequence.
        Each frame is returned as a tuple `(img, variables)` where
        `img` is a numpy array containing the image and `variables` is a list of
        variable values (e.g., orientation) associated with that frame.

        See topographica_based for examples.
        """
        raise NotImplementedError("Must be implemented by child class.")

    def update(self):
        """
        Sets the current frame to the next frame in the sequence.
        """
        raise NotImplementedError("Must be implemented by child class.")

    def reset(self):
        """
        Reset to the first frame in the sequence.
        """
        raise NotImplementedError("Must be implemented by child class.")

    def export(self, path=None):
        """
        Save the frames to disk. Returns a list of paths to the individual
        frames.
        
        path - the directory in which the individual frames will be saved. If
               path is None, then a temporary directory is created.
        """
        raise NotImplementedError("Must be implemented by child class.")





def _colapse(dd,param):
    d = {}
    for s in dd:
        s1 = StimulusID(s)
        s1.params[param]=None
        s1 = str(s1)
        if d.has_key(s1):
           d[s1].extend(dd[s])
        else:
           d[s1] = dd[s]
    return d
    
def colapse(value_list,stimuli_list,parameter_list=[]):
    ## it colapses the value_list acording to stimuli with the same value 
    ## of parameters whose indexes are listed in the <parameter_indexes> and 
    ## replaces the collapsed parameters in the 
    ## stimuli_list with *
    d = {}
    for v,s in zip(value_list,stimuli_list):
        d[str(s)]=[v]

    for param in parameter_list:
        d = _colapse(d,param)
    
    return ([d[k] for k in d.keys()] ,d.keys())


def find_stimulus(stimulus_list,**kwargs):
    """
    Find all stimuli in stimulus_list which match of the kwargs pairs parameter_name=value.
    """
    #for s in stimuli
    a  = 2
    

      
