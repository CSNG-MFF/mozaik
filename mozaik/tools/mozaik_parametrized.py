from param.parameterized import Parameterized
from param import Number, Integer, String
from sets import Set
import logging
import inspect
import collections

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

    name = SString(doc="String identifier for this object that is set to it's class name DO NOT CHANGE.")

    def __init__(self, **params):
        Parameterized.__init__(self, **params)
        self.module_path = inspect.getmodule(self).__name__
        self.name = self.__class__.__name__
        
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

    def __str__(self):
        """
        Turn the MozaikParametrized instance into string - this stores ONLY the names and values of each parameter and the module path from which this instance class came from.
        """
        settings = ['\"%s\":%s' % (name, repr(val)) for name, val in self.get_param_values()]
        r = "{ \"name\" :" + "\"" + self.name + "\""+ "," + "\"module_path\" :" + "\"" + self.module_path + "\"" +',' + ", ".join(settings) + "}"
        return r

    @classmethod
    def idd(cls,obj):
        """
        This class method is used in concjuction with the MozaikParametrized.__str__ function that stores all the parameters and the class and module of an object.
        This method restores a 'Shell' object out of this str. The returned object will be of the same type as the original object and will contain all its original parameters
        and their values, BUT WILL NOT BE INITIALIZED and so should not be used for anything else other than examining it's parameters!!!!
        
        Furthermore if given an instance of MozaikParametrized instead it will convert it into the 'Shell' object.
        """
        if isinstance(obj,MozaikParametrized):
           return MozaikParametrized.idd(str(obj))
        assert isinstance(obj,str)
        
        params = eval(obj)
        name = params.pop("name")
        module_path = params.pop("module_path")
        z = __import__(module_path, globals(), locals(), name)
        
        cls = getattr(z,name)
        
        obj = cls.__new__(cls,**params)
        MozaikParametrized.__init__(obj,**params)
        return obj
    
    
"""
Helper functions that allow querying lists of MozaikParametrized objects.
"""

def filter_query(object_list, extra_data_list=None,allow_non_existent_parameters=False,**kwargs):
    """
    Returns list of MozaikParametrized (and associated data if data_list!=None) of
    for which the parameters in kwargs match.

    object_list - the list of MozaikParametrized objects to filter
    data_list - the list of values corresponding to objects in object_list
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
    if extra_data_list == None:
        extra_data_list = [[] for z in xrange(0, len(object_list))]
        no_data = True
    else:
        assert(len(extra_data_list) == len(object_list))
    
    def fl(x,kwargs,allow): 
        x = x[0]
        if not allow and not (set(kwargs.keys()) <= set(x.params().keys())):
           return False 
        keys = set(kwargs.keys()) & set(x.params().keys())
        return [getattr(x,k) for k in keys] == [kwargs[k] for k in keys]
    
    res = zip(*filter(lambda x : fl(x,kwargs,allow_non_existent_parameters),zip(object_list,extra_data_list)))
    
    if no_data:
       if len(res)==0:
          return []
       return res[0]
    else:
       if len(res)==0:
          return [[],[]]
       return res 




def _colapse(dd, param):
    d = collections.OrderedDict()
    for s in dd:
        s1 = MozaikParametrized.idd(s)

        if param not in s1.params().keys():
            raise KeyError('colapse: MozaikParametrized object ' + str(s1) + ' does not contain parameter [%s]' % (param))

        setattr(s1,param,None)
        s1 = str(s1)
        if s1 in d:
           d[s1].extend(dd[s])
        else:
           d[s1] = dd[s]
            
    return d


def colapse(data_list, object_list, func=None, parameter_list=[],
            allow_non_identical_objects=False):
    """
    It collapses the data_list against parameters of objects in object_list that are in parameter_list. 
    This means that the new list of parameters (and associated datalist) will contain one object for each
    combination of parameter values not among the paramters against which to
    collapse. Each such object will be associated with a list of data that
    corresponded to object with the same parameter values, but any values for
    the parameters against which one is collapsing.
    The collapsed parameters in the object_list will be replaced with None.

    The function returns a tuple of lists (v,object_id), where object_id is
    the new list of objects where the object in parameter_list were
    'collapsed out' and replaced with None. v is a list of lists. The outer list
    corresponding in order to the object_id list. The inner list corresponds
    to the list of data from data_list that mapped on the given object_id.

    If func != None, func is applied to each member of v.

    data_list - the list of data corresponding to objects in object_list
    object_list - the list of object corresponding to data in data_list
    parameter_list - the list of parameter names against which to collapse the data
    func - the func to be applied to the lists formed by data associated with the
           same object parametrizations with exception of the parameters in parameter_list
    allow_non_identical_objects - (default=False) unless set to True, it will
                                  not allow running this operation on
                                  StimulusDependentData that does not contain
                                  only object of the same type
    """
    assert(len(data_list) == len(object_list))
    if (not allow_non_identical_objects and not identical_parametrized_object_params(object_list)):
        raise ValueError("colapse accepts only object lists of the same type")

    d = collections.OrderedDict()
    for v, s in zip(data_list, object_list):
        d[str(s)]=[v]

    for param in parameter_list:
        d = _colapse(d, param)

    values = d.values()
    st = [MozaikParametrized.idd(idd) for idd in d.keys()]
    
    
    if func != None:
        return ([func(v) for v in values], st)
    else:
        return (values, st)
        
def varying_parameters(parametrized_objects):
    """
    Find the varying list of params. Can be only applied
    on list of MozaikParametrized that have the same parameter set.
    """
    if not identical_parametrized_object_params(parametrized_objects):
        raise ValueError("varying_parameters: accepts only MozaikParametrized lists with the same parameters")

    p = parametrized_objects[0].params().keys()
    varying_params = collections.OrderedDict()
    for n in p.keys():
        for o in parametrized_objects:
            if getattr(o,n) != getattr(parametrized_objects[0],n):
                varying_params[n] = True
                break
    return varying_params.keys()


def identical_parametrized_object_params(parametrized_objects):
    for o in parametrized_objects:
        if set(o.params().keys()) != set(parametrized_objects[0].params().keys()):
                return False
    return True
                
                
def matching_parametrized_object_params(parametrized_objects,params=None,except_params=None):
    """
    if params != None
        Returns true if all MozaikParametrized objects in parametrized_objects have the same parameter values for parameters in params, otherwise returns False.
    
    if except_params != None
        Returns true if all MozaikParametrized objects in parametrized_objects have the same parameters except those in except_params, otherwise returns False.
        
    if except_params == None and except_params == None:
        Returns true if all MozaikParametrized objects in parametrized_objects have the same parameters, otherwise returns False.
        
    if except_params != None and except_params != None:
        Throws exception
    
    """
    if except_params != None and params != None:
        raise ValueError('identical_parametrized_object_params cannot be called with both params and except_params equal to None')
    
    if except_params == None and params == None:
        params = parametrized_objects[0].params().keys()
    
    if len(parametrized_objects) == 0:
        return True
    else:
        first =  parametrized_objects[0].params()

    for o in parametrized_objects:
        if except_params == None:    
           if set([first[k] for k in params]) != set([o.params()[k] for k in params]):
                return False
        else:
           if set([first[k] for k in (set(first.keys())-set(except_params))]) != set([o.params()[k] for k in (set(o.params().keys())-set(except_params))]):
                return False
                    
    return True
    
    
def colapse_to_dictionary(value_list, parametrized_objects, parameter_name):
    """
    Returns dictionary D where D.keys() correspond to parametrized_objects ids
    with the dimension 'parameter name' colapsed out (and replaced with None).

    The D.values() are tuple of lists (keys,values), where keys is the list of
    values that the parameter_name had in the parametrized_objects, and the values are the
    values from value_list that correspond to the keys.
    """
    assert(len(value_list) == len(parametrized_objects))
    d = collections.OrderedDict()

    for (v, s) in zip(value_list, parametrized_objects):
        s = MozaikParametrized.idd(s)
        val = getattr(s,parameter_name)
        setattr(s,parameter_name,None)
        if str(s) in d:
            (a, b) = d[str(s)]
            a.append(val)
            b.append(v)
        else:
            d[str(s)] = ([val], [v])
    dd = {}
    for k in d:
        (a, b) = d[k]
        dd[k] = (a, b)
    return dd


    

