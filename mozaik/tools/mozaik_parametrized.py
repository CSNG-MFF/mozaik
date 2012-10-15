from param.parameterized import Parameterized
from param import Number, Integer, String
import logging

logger = logging.getLogger("mozaik")

"""
For BaseStimulus and ADS objects we will allow only SNumber, SInteger and SString parameters.
These are extension of corresponding parametrized parameters that automatically
allow None value, are instantiated and allow for definition of units.
"""

class SNumber(Number):
    __slots__ = ['units','period']

    def __init__(self, units, period=None, **params):
        params.setdefault('default',None)
        super(SNumber, self).__init__(allow_None=True,
                                      instantiate=True, **params)
        self.units = units
        self.period = period
        


class SInteger(Integer):
    __slots__ = ['units','period']

    def __init__(self, period=None, **params):
        params.setdefault('default',None)        
        super(SInteger, self).__init__(allow_None=True,
                                       instantiate=True, **params)
        self.units = None
        self.period = period


class SString(String):
    __slots__ = ['units','period']

    def __init__(self, **params):
        params.setdefault('default',None)        
        super(SString, self).__init__(allow_None=True,
                                      instantiate=True, **params)
        self.units = None
        self.period = None



class MozaikParametrized(Parameterized):
    """
    We extend the topographica Parametrized package to constrain the parametrization.
    We allow only three parameter types (SNumber or SInteger or SString) that we have 
    extended with further information. 
    
    This allows us to define several useful operations over such parametrized objects that
    we will use extensively (see above). 
    
    Currently the main use of this parametrization is for defining stimuli and 
    analysis data structures. It allows us to write general and powerfull qurying 
    functions, automatically handle parameter units and parameter names.
    """

    name = SString(precedence=-1,doc="String identifier for this object.")

    def __init__(self, **params):
        Parameterized.__init__(self, **params)
        for name in self.params():
            o = self.params()[name]
            if not (isinstance(o,SNumber) or isinstance(o,SInteger) or isinstance(o,SString)):
               raise ValueError("The parameter %s is not of type SNumber or SInteger or SString" % name)
 
        for (name, value) in self.get_param_values():
            if value == None and self.params()[name].allow_None==False:                
                logger.error("The parameter %s was not initialized" % name)
                raise ValueError("The parameter %s was not initialized" % name)

    def equalParams(self, other):
        """
        Returns True if self and other have the same parameters and all their
        values match. False otherwise.

        JACOMMENT: This seems to work only because get_param_values sorts the
        list by names which is undocumented!
        """
        return self.get_param_values() == other.get_param_values()

    def equalParamsExcept(self, other, exceptt):
        """
        Returns True if self and other have the same parameters and all their
        values match with the exception of the parameter in exceptt.
        False otherwise.
        """
        a = self.get_param_values()
        b = self.get_param_values()
        for k in exceptt:
            for i, (key, value) in enumerate(a):
                if key == k:
                    break
            a.pop(i)

            for i, (key, value) in enumerate(b):
                if key == k:
                    break
            b.pop(i)

        return a == b

    @classmethod
    def params(cls, parameter_name=None):
        """
        In MozaikParametrized we hide parameters with precedence below 0 from
        users.
        """
        d = super(MozaikParametrized, cls).params(parameter_name).copy()
        for k in d.keys():
            if d[k].precedence < 0 and d[k].precedence != None:
                del d[k]

        return d

"""
Helper functions that allow querying lists of MozaikParametrized objects.
"""
def filter_query(object_list, extra_data_list=None,allow_non_existent_parameters=False,**kwargs):
    """
    Returns list of MozaikParametrized (and associated data if data_list!=None) of
    for which the parameters in kwargs match.

    object_list - the list of MozaikParametrized objects to filter
    data_list - the list of values corresponding to stimuli in object_list
    **kwargs - the parameter names and values that have to match
    allow_non_existent_parameters - if True it will allow objects in object list that 
                                    miss some of the parameters listed in kwargs, 
                                    and include them in the results as long the remaining 
                                    parameters match
                                    if False it will exclude them
    returns:
            if data_list == None :  subset of object_list containing elements that match the kwargs parameters
            if data_list != None :  tuple (a,b) where a is a subset of object_list containing elements that match the kwargs parameters and b is the corresponding subset of data_list
                                            
    """
    no_data = False
    if data_list == None:
        data_list = [[] for z in xrange(0, len(object_list))]
        no_data = True
    else:
        assert(len(data_list) == len(object_list))
    
    def fl(x,kwargs,allow): 
        x = x[0]
        if not allow and not (set(kwargs.keys()) <= set(x.params())):
           return False 
        keys = set(kwargs.keys()) & set(x.params())
        return x.params()[keys] == kwargs[keys]

    res = zip(*filter(fl,zip(object_list,data_list)))
    
    if no_data:
       return res[0]
    else:
       return res 
