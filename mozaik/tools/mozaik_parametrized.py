from param.parameterized import Parameterized
from param import Number, Integer, String
import logging

logger = logging.getLogger("mozaik")

class MozaikParametrized(Parameterized):
      
      def __init__(self,**params):
          Parameterized.__init__(self,**params)
          
          for (name,value) in self.get_param_values():
              if value == None:
                 logger.error("The parameter %s was not initialized" % name) 
                 raise ValueError("The parameter %s was not initialized" % name) 
            
      def equalParams(self,other):
          """
          Returns True if self and other have the same parameters and all their values match.
          False otherwise.
          
          JACOMMENT: This seems to work only because get_param_values sorts the list by names
          which is undocumented! 
          """
          return self.get_param_values() == other.get_param_values()
         

"""
For Stimuli objects we will allow only SNumber, SInteger and SString parameters.
These are extension of corresponding parametrized parameters that automaticall allow None
value, are instatiated and allow for definition of units.
"""
class SNumber(Number):
      __slots__ = ['units']
      def __init__(self,units,**params):
            super(SNumber,self).__init__(default=None,allow_None=True,instantiate=True,**params)
            self.units = units

class SInteger(Integer):
      __slots__ = ['units']    
      def __init__(self,**params):
            super(SInteger,self).__init__(default=None,allow_None=True,instantiate=True,**params)
            self.units = None        

class SString(String):
       __slots__ = ['units']
       def __init__(self,**params):
            super(SString,self).__init__(default=None,allow_None=True,instantiate=True,**params)
            self.units = None
