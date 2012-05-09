"""
This package defines the API for:
    - implementation of stimuli as input to models (see class Stimulus)
    - identification of stimulus identity (without the actual 'data' of the stimulus) throughout *mozaik* (see class StimulusID)
    - function helpers for common manipulation with collections of stimuli (or rather their IDs)

Each stimulus is expected to have a dictionary of parameters which have to uniquely identify the stimulus.

We implement this by using the Parametrized package that allows to specify parameters with the all above requirements.
Note that *all* such parameters defined in the class (and its ancestors) will be considered as parameters of the Stimulus.
"""
from mozaik.framework.interfaces import VisualStimulus
import quantities as qt
import numpy

class StimulusID():
      """
      StimulusID is a light object that contains all the parameter info of a given Stimulus object.
      There is one to one mapping between StimuluID instances and Stimulus instances.
      
      The main purpose for StimulusID is to stand for Stimulus whenever the stimulus identitity is required
      without the need to move around the heavy Stimulus objects.
      """  

      def __str__(self):
            """
            Provide a nearly valid Python representation that could be used to recreate
            the item with its parameters, if executed in the appropriate environment.
            
            Returns 'classname(parameter1=x,parameter2=y,...)', listing
            all the parameters of this object.
            """
            settings = ['%s:%s' % (name,repr(val))
                    for name,val in self.get_param_values()]
            return "{name:" + self.name  + ", ".join(settings) + "}"

      def __eq__(self, other):
            return (self.name == other.name) and (self.get_param_values() == other.get_param_values())
            
      def get_param_values(self):
          return self.params.items().sort(key=itemgetter(0))
          
      def load_stimulus(self):
          cls = globals()[self.name]
          return cls(**self.params)  
        
      class sid_param():
            """Just a hack to allow parameters in StimulusID have units"""
            def __init__(value,units):
                self.units = units
                self.value = value

            def __get__(self, instance, owner):
                return self.value
    
      def __init__(obj):
          self.params = {}          
          
          if isinstance(obj,Stimulus):
              self.name = obj.__class__
              par = obj.params()
              for n,v in obj.get_param_values():
                  setattr(self, n, sid_param(v,par[n].units))
                  params[n] = v
          elif isinstance(obj,dict):
              self.name = obj.pop(name)
              par = globals()[self.name].params()
              for n,v in obj.items():
                  setattr(self, n, sid_param(v,par[n].units))
                  params[n] = v
          elif isinstance(obj,str):
               d = eval(str)
               self.name = d.pop(name)
               par = globals()[self.name].params()
               for n,v in d.items():
                  setattr(self, n, sid_param(v,par[n].units))
                  params[n] = v
          else:
              raise ValueError("obj is not of recognized type (recognized: str,dict or Stimulus)")
              
class Stimulus(VisualStimulus):
        duration = param.SNumber(units=qt.ms,instantiate=True,doc="""The duration of stimulus""")
        density = param.SNumber(units=1/(qt.degree),instantiate=True,doc="""The density of stimulus - units per degree""")
        trial = param.Integer(instantiate=True,doc="""The duration of stimulus""")

        def __str__(self):
            self.script_repr()
            
        def __eq__(self, other):
            return self.equalParams(other) and (self.__class__ == other.__class__)

        def __init__(self, **params):
            self.vparams = parameters
            VisualStimulus.__init__(self,**params)
            self.n_frames = numpy.inf # possibly very dangerous. Don't do 'for i in range(stim.n_frames)'!




def _colapse(dd,param):
    d = {}
    for s in dd:
        s1 = StimulusID(s)
        s1.parameters[axis]='*'
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

      
