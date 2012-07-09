from param.parameterized import Parameterized
from param import Number, Integer, String
import logging

logger = logging.getLogger("mozaik")

class MozaikParametrized(Parameterized):
      
      name = String(default=None,constant=True,precedence=-1,doc="""String identifier for this object.""")
      
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
      
      def equalParamsExcept(self,other,exceptt):
          """
          Returns True if self and other have the same parameters and all their values match with
          the exception of the parameter in exceptt.
          False otherwise.
          """
          a = self.get_param_values()
          b = self.get_param_values()
          for k in exceptt:
              for i,(key,value) in enumerate(a):
                  if key == k:
                     break
              a.pop(i)
 
              for i,(key,value) in enumerate(b):
                  if key == k:
                     break
              b.pop(i)
          
          return a == b
         
      @classmethod
      def params(cls,parameter_name=None):
          """
          In MozaikParametrized we hide parameters with precedence below 0 from users.
          """
          d = super(MozaikParametrized, cls).params(parameter_name).copy()
          for k in d.keys():
              if d[k].precedence < 0 and d[k].precedence != None:
                 del d[k] 
          
          return d  
"""
For Stimuli objects we will allow only SNumber, SInteger and SString parameters.
These are extension of corresponding parametrized parameters that automaticall allow None
value, are instatiated and allow for definition of units.
"""
class SNumber(Number):
      __slots__ = ['units','period']
      def __init__(self,units,period=None,**params):
            super(SNumber,self).__init__(default=None,allow_None=True,instantiate=True,**params)
            self.units = units
            self.period = period

class SInteger(Integer):
      __slots__ = ['units','period']    
      def __init__(self,period=None,**params):
            super(SInteger,self).__init__(default=None,allow_None=True,instantiate=True,**params)
            self.units = None        
            self.period = period

class SString(String):
       __slots__ = ['units','period']
       def __init__(self,**params):
            super(SString,self).__init__(default=None,allow_None=True,instantiate=True,**params)
            self.units = None
            self.period = None
