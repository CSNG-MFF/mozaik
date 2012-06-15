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

from mozaik.framework.interfaces import VisualStimulus
import quantities as qt
import numpy
from operator import *
from mozaik.tools.mozaik_parametrized import *
import inspect


class StimulusID():
      """
      StimulusID is a light object that contains all the parameter info of a given Stimulus object.
      There is one to one mapping between StimuluID instances and Stimulus instances.
      
      The main purpose for StimulusID is to stand for Stimulus whenever the stimulus identitity is required
      without the need to move around the heavy Stimulus objects.
      
      Unlike Stimulus, StimulusID allows any parameter to be also assigned None. This is often 
      used in data analysis to mark parameters that has been 'computed out' by the analysis (such as averaged out).
      
      To access the parameter values refer to the param member variable
      To access the units values refer to the units member variable
      To access the periodicity (or lack of it) values refer to the periods member variable
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
          
      def copy(self):
          return StimulusID(str(self))
          
      def __init__(self,obj):
          self.units = {}
          self.periods = {}
          self.params = {}
          if isinstance(obj,Stimulus):
              self.name = obj.__class__.__name__
              self.module_path = inspect.getmodule(obj).__name__
              par = obj.params()
              for n,v in obj.get_param_values():
                  if n != 'name' and n != 'print_level':
                      self.params[n] = v
                      self.units[n] = par[n].units
                      self.periods[n] = par[n].period
          elif isinstance(obj,dict):
              self.name = obj.pop("name")
              self.module_path = obj.pop("module_path")
              par = self.getStimulusClass().params()
              for n,v in obj.items():
                  self.params[n] = v
                  self.units[n] = par[n].units
                  self.periods[n] = par[n].period                  
          elif isinstance(obj,str):
               d = eval(obj)
               self.name = d.pop("name")
               self.module_path = d.pop("module_path")
               par = self.getStimulusClass().params()
               for n,v in d.items():
                  self.params[n] = v
                  self.units[n] = par[n].units
                  self.periods[n] = par[n].period
          elif isinstance(obj,StimulusID):
               self.name = obj.name
               self.module_path = obj.module_path
               self.units = obj.units.copy()
               self.periods = obj.periods.copy()
               self.params = obj.params.copy()
          else:
              raise ValueError("obj is not of recognized type (recognized: str,dict, StimulusID or Stimulus)")
              

class Stimulus(VisualStimulus):
        duration = SNumber(qt.ms,doc="""The duration of stimulus""")
        density = SNumber(1/(qt.degree),doc="""The density of stimulus - units per degree""")
        trial = SInteger(doc="""The duration of stimulus""")

        def __str__(self):
            return str(StimulusID(self))
            
        def __eq__(self, other):
            return self.equalParams(other) and (self.__class__ == other.__class__)

        def __init__(self, **params):
            VisualStimulus.__init__(self,**params)
            self.n_frames = numpy.inf # possibly very dangerous. Don't do 'for i in range(stim.n_frames)'!

def _colapse(dd,param):
    d = {}
    for s in dd:
        s1 = StimulusID(s)
        
        if param not in s1.params.keys():
           raise KeyError('colaps: only stimuli containing parameter [%s] can be collapsed again parameter [%s]' % (param,param)) 
        
        s1.params[param]=None
        s1 = str(s1)
        if d.has_key(s1):
           d[s1].extend(dd[s])
        else:
           d[s1] = dd[s]
    return d
    
def colapse(value_list,stimuli_list,parameter_list=[]):
    """
    It colapses the value_list acording to stimuli with the same value 
    of parameters whose indexes are listed in the <parameter_indexes> and 
    replaces the collapsed parameters in the stimuli_list with None.
    
    It returns a tuple of lists (v,stimuli_id), where stimuli_id is
    the new list of stimuli where the stimuli in parameter_list were 'colapsed out'
    and replaced with None. v is a list of lists. The outer list corresponding in order to the stimuli_id list.
    the inner list corresponds to the list of values from value_list that mapped on the given
    stimuli_id.
    """
    d = {}
    for v,s in zip(value_list,stimuli_list):
        d[str(s)]=[v]

    for param in parameter_list:
        d = _colapse(d,param)
    
    return ([d[k] for k in d.keys()] ,[StimulusID(idd) for idd in d.keys()])


def colapse_to_dictionary(value_list,stimuli_list,parameter_name):
    """
    Returns dictionary D where D.keys() correspond to stimuli_ids 
    with the dimension 'parameter name' colapsed out (and replaced with None).
        
    The D.values() are tuple of lists (keys,values), where keys is the list of 
    values that the parameter_name had in the stimuli_list, and the values are the 
    values from value_list that correspond to the keys.
    """
    d = {}
    for (v,s) in zip(value_list,stimuli_list):
        s=s.copy()
        val = s.params[parameter_name]
        s.params[parameter_name] = None
        if d.has_key(str(s)):
           (a,b) = d[str(s)] 
           a.append(val)
           b.append(v)
        else:
           d[str(s)]  = ([val],[v]) 
    dd = {}
    for k in d:
        (a,b) = d[k]
        dd[k] = (a,b)
    return dd
    
def identical_stimulus_type(stimuli_list):
    """
    Returns true if all stimuli in stimulus_list are of the same type, else returns False.
    """
    stimulus_type = stimuli_list[0].name
    for sid in stimuli_list:
        if sid.name != stimulus_type:
            return False
    return True
           
                  
