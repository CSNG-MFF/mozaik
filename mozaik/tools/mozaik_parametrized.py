"""
This module contains extension of the `param <http://ioam.github.io/param/>`_ package,
and a collection of functions that allow for powerfull filtering of sets
of :class:`.MozaikParametrized` instances based on the values of their parameters.

When parametrizing *mozaik* objects we will allow only SNumber, SInteger and SString parameters.
These are extension of corresponding parametrized parameters that automatically
allow None value, are instantiated and allow for definition of units and period.
"""


from param.parameterized import Parameterized
from param import Number, Integer, String, produce_value, ClassSelector
from sets import Set
from parameters import ParameterSet
import logging
import inspect
import collections
from mozaik.tools.distribution_parametrization import ParameterWithUnitsAndPeriod, MozaikExtendedParameterSet


import param.parameterized
param.parameterized.docstring_signature=False

logger = logging.getLogger("mozaik")

class SNumber(Number):
    """
    A mozaik parameter that can hold a number. For the full range of options the 
    parameter offers reffer to the `Number` class in `param <http://ioam.github.io/param/>`_ package.
    
    On top of the Number class it adds the ability to specify units and period in constructor.
    The units should be `quantities.units`.
    """
    __slots__ = ['units','period']

    def __init__(self, units, period=None, **params):
        params.setdefault('default',None)
        super(SNumber, self).__init__(allow_None=True,
                                      instantiate=True, **params)
        self.units = units
        self.period = period
        


class SInteger(Integer):
    """
    A mozaik parameter that can hold an integer. For the full range of options the 
    parameter offers reffer to the `Integer` class in `param <http://ioam.github.io/param/>`_ package.
    
    On top of the Integer class it adds the ability to specify period in constructor. The units should be `quantities.units`.
    """    
    __slots__ = ['units','period']

    def __init__(self, period=None, **params):
        params.setdefault('default',None)        
        super(SInteger, self).__init__(allow_None=True,
                                       instantiate=True, **params)
        self.units = None
        self.period = period


class SString(String):
    """
    A mozaik parameter that can hold an string. For the full range of options the 
    parameter offers reffer to the `Integer` class in `param <http://ioam.github.io/param/>`_ package.
    
    This class is here for consistency reasons as it defines the units and period properties, 
    just like SInteger and SNumber, but automatically sets them to None.
    """

    __slots__ = ['units','period']    
    def __init__(self, **params):
        params.setdefault('default',None)        
        super(SString, self).__init__(allow_None=True,
                                      instantiate=True, **params)
        self.units = None
        self.period = None


class SNumber(Number):
    """
    A mozaik parameter that can hold a number. For the full range of options the 
    parameter offers reffer to the `Number` class in `param <http://ioam.github.io/param/>`_ package.
    
    On top of the Number class it adds the ability to specify units and period in constructor.
    The units should be `quantities.units`.
    """
    __slots__ = ['units','period']

    def __init__(self, units, period=None, **params):
        params.setdefault('default',None)
        super(SNumber, self).__init__(allow_None=True,
                                      instantiate=True, **params)
        self.units = units
        self.period = period

class SParameterSet(ClassSelector):
    """
    A mozaik parameter that can hold a ParameterSet. Such parameter is treated in a special way: see MozaikParametrized for more details.
    """
    def __init__(self, **params):
        params.setdefault('default',None)
        super(SParameterSet, self).__init__(MozaikExtendedParameterSet,allow_None=True,
                                      instantiate=True, **params)



class MozaikParametrized(Parameterized):
    """
    We extend the topographica Parametrized package to constrain the parametrization.
    We allow only four parameter types (SNumber or SInteger or SString or SParameterSet) that we have 
    extended with further information. 
    
    This allows us to define several useful operations over such parametrized objects that
    we will use extensively (see above). 
    
    Currently the main use of this parametrization is for defining stimuli and 
    analysis data structures. It allows us to write general and powerfull querying 
    functions, automatically handle parameter units and parameter names.
    
    For the usage of the :class:`.MozaikParametrized` class refer to the `Parameterized`
    in `param <http://ioam.github.io/param/>`_ package, as the behaviour of this class is 
    identical, with the exception that it restricts the possbile parameters used to the
    trio of SNumber or SInteger or SString.
    
    This class also adds several utility functions documented below.
    """

    name = SString(doc="String identifier for this object that is set to it's class name DO NOT TOUCH!")
    
    # we will chache imported modules due to the idd statement here, as module imports seem to be extremely costly.
    _module_cache = {}
    
    def __init__(self, **params):
        
        self.cached_get_param_values = None
        Parameterized.__init__(self, **params)
        self.module_path = inspect.getmodule(self).__name__
        self.name = self.__class__.__name__
        

        for name in self.params():
            o = self.params()[name]
            if not (isinstance(o,SNumber) or isinstance(o,SInteger) or isinstance(o,SString) or isinstance(o,SParameterSet)):
               raise ValueError("The parameter %s is not of type SNumber or SInteger or SString or SParameterSet but of type %s." % (name,type(o)))
 
        for (name, value) in self.get_param_values():
            if value == None and self.params()[name].allow_None==False:                
                logger.error("The parameter %s was not initialized" % name)
                raise ValueError("The parameter %s was not initialized" % name)

        def _expand_parameter_set(ps):
            """
            Helper function that expands parameter set.
            """
            ret = []
            for k in ps.keys():
                if isinstance(ps[k],MozaikExtendedParameterSet):
                   ret += [(k + '_' + p,v)  for p,v in  _expand_parameter_set(ps[k])]
                else:
                   ret.append((k,ps[k]))       
            return ret

        # find out which params are SParameterSet
        self.paramset_params_names =  [ps for ps in self.params().keys() if isinstance(self.params()[ps],SParameterSet)]

        # store expanded SParameterSet parameters
        self.expanded_paramset_params = []
        for ps in self.paramset_params_names:
                if getattr(self,ps) != None:
                    self.expanded_paramset_params+=_expand_parameter_set(getattr(self,ps))
        self.expanded_paramset_params.sort(key=lambda tup: tup[0])



        if self.expanded_paramset_params != []:
            self.expanded_params_names = zip(*self.expanded_paramset_params)[0]
        else:
            self.expanded_params_names= []

        assert ((set([t[0] for t in self.expanded_paramset_params]) & set(self.params().keys())) == set([])) , ("Conflict between MozaikParametrized and expanded ParameterSet parameters. Intersection: %s " % str(set([t[0] for t in self.expanded_paramset_params]) & set(self.params().keys())))

        # and lets also have a dictionary version of the parameters
        self.expanded_paramset_params_dict = self.params().copy()

        # remove SParemeterSet parameters 
        for key in self.expanded_paramset_params_dict.keys():
            if isinstance(self.expanded_paramset_params_dict[key],SParameterSet):
               del self.expanded_paramset_params_dict[key]
        
        self.expanded_paramset_params_dict.update(dict(self.expanded_paramset_params))


    

    def __setattr__(self,attribute_name,value):
        """
        We need to override the Parametrized __setattr__ to handle setting of SParameterSet parameters.
        """
        def set_in_dict(path, dt,value):
            keys = path.split('_')
            for key in keys[:-1]:
                dt = dt[key]
            dt[keys[-1]]=value

        if hasattr(self, 'expanded_params_names'):
            if attribute_name in self.expanded_params_names:
                self.expanded_paramset_params_dict[attribute_name] = value;

                n =[]
                for z in self.expanded_paramset_params:
                    if z[0] == attribute_name:
                       n.append((z[0],value)) 
                    else:
                       n.append(z)   
                self.expanded_paramset_params = n

                #HAAAAAAAAAAAAACK
                if attribute_name == 'stimulating_signal_parameters_orientation':
                    getattr(self,'direct_stimulation_parameters')['stimulating_signal_parameters']['orientation']=value

                if attribute_name == 'stimulating_signal_parameters_scale':
                    getattr(self,'direct_stimulation_parameters')['stimulating_signal_parameters']['scale']=value

                if attribute_name == 'stimulating_signal_parameters_contrast':
                    getattr(self,'direct_stimulation_parameters')['stimulating_signal_parameters']['contrast']=value

                if attribute_name == 'stimulating_signal_parameters_size':
                    getattr(self,'direct_stimulation_parameters')['stimulating_signal_parameters']['size']=value


                #set_in_dict(attribute_name,getattr(self,'direct_stimulation_parameters'),value)
                #set_in_dict(path[1:,],self.params()[path[0]],value)
                return
        Parameterized.__setattr__(self,attribute_name,value)
        Parameterized.__setattr__(self,'cached_get_param_values',None)
    

    def get_param_values(self,onlychanged=False):
        if self.cached_get_param_values == None:
           Parameterized.__setattr__(self,'cached_get_param_values',Parameterized.get_param_values(self,onlychanged))
        return self.cached_get_param_values
        
    def equalParams(self, other):
        """
        Returns True if self and other have the same parameters and all their
        values match. False otherwise.

        JACOMMENT: This seems to work only because get_param_values sorts the
        list by names which is undocumented!
        """
        return (self.get_param_values() == other.get_param_values()) and (self.expanded_paramset_params == other.expanded_paramset_params)

    def getParams(self):
        """
        This is the function that MozaikParametrized objects should use. It returns all the MozaikParametrized, + the expanded parameters from SParameterSet parameters, - the SParameterSet.
        In Mozaik all comparison and filtering operations should always be done based on this.
        """
        return self.expanded_paramset_params_dict

    def getParamValue(self,name):
        """
        This is the function that MozaikParametrized objects should use to retrieve a value of a parameter.
        """
        if name in self.expanded_params_names:
            if isinstance(self.expanded_paramset_params_dict[name],ParameterWithUnitsAndPeriod):
                return  self.expanded_paramset_params_dict[name].value
            else: 
                return self.expanded_paramset_params_dict[name]
        else:
            return getattr(self,name)

    def __str__(self):
        """
        Turn the MozaikParametrized instance into string - this stores ONLY the names and values of each parameter and the module path from which this instance class came from.
        """
        settings =[]

        for name, val in self.get_param_values():
            if isinstance(val,MozaikExtendedParameterSet):
                settings.append('\"%s\":MozaikExtendedParameterSet(%s)' % (name, repr(val)))
            else:
                settings.append('\"%s\":%s' % (name, repr(val)))
        
        r = "{\"module_path\" :" + "\"" + self.module_path + "\"" +',' + ", ".join(settings) + "}"
        return r

    def __repr__(self):
        """
        Returns the description of the MozaikParametrized instance - its class name and the list of its parameters and their values.
        """
        param_str = "\n".join(['   \"%s\":%s' % (name, repr(val))
                               for name, val in self.get_param_values()])
        return self.__class__.__name__ + "\n" + param_str + "\n"

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
        assert isinstance(obj,str) , "The object passed to the idd class method is not string: %s" % (type(obj)) 
        
        params = eval(obj)
        name = params.pop("name")
        module_path = params.pop("module_path")
        
        if (module_path,name) in MozaikParametrized._module_cache:
            z = MozaikParametrized._module_cache[(module_path,name)]
        else:
            z = __import__(module_path, globals(), locals(), name)
            MozaikParametrized._module_cache[(module_path,name)] = z
            
        cls = getattr(z,name)
        
        obj = cls.__new__(cls,**params)
        MozaikParametrized.__init__(obj,**params)
        return obj

    @classmethod
    def idd_to_instance(cls,obj):
        """
        
        """
        if isinstance(obj,MozaikParametrized):
           return MozaikParametrized.idd(str(obj))
        assert isinstance(obj,str)
        
        params = eval(obj)
        name = params.pop("name")
        module_path = params.pop("module_path")
        z = __import__(module_path, globals(), locals(), name)
        
        cls = getattr(z,name)
        return cls(**params)        
    
"""
Helper functions that allow querying lists of MozaikParametrized objects.
"""

def filter_query(object_list, extra_data_list=None,allow_non_existent_parameters=False,**kwargs):
    """
    Returns a subset of `object_list` containing MozaikParametrized instances (and associated data if data_list!=None)
    for which the parameters in kwargs match.

    Parameters
    ----------
    
    object_list : list(MozaikParametrized)
                The list of MozaikParametrized objects to filter.
                
    extra_data_list : list(object)
                    The list of values corresponding to objects in object_list. 
                    The same subset of the vales will be returned as for the `object_list`.
                    
    allow_non_existent_parameters : bool
                                    If True it will allow objects in object list that 
                                    miss some of the parameters listed in kwargs, 
                                    and include them in the results as long the remaining 
                                    parameters match if False it will exclude them.
                                    
    \*\*kwargs : dict
             The parameter names and values that have to match.                                    
    
    Returns
    -------
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
        if not allow and not (set(kwargs.keys()) <= set(x.getParams().keys())):
           return False 
        keys = set(kwargs.keys()) & set(x.getParams().keys())
        for k in keys:
            if not isinstance(kwargs[k],list): 
                if kwargs[k] != x.getParamValue(k):
                   return False
            elif not (x.getParamValue(k) in kwargs[k]):
               return False
        return True
    
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

        if param not in s1.getParams().keys():
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



    Parameters
    ----------
    
    data_list : list
              The list of data corresponding to objects in object_list.
    
    object_list : list(MozaikParametrized)
                The list of object corresponding to data in data_list.
    
    parameter_list : list
                   The list of parameter names against which to collapse the data.
    
    func : func()
         If not None, the function to be applied to the lists formed by data associated
         with the same object parametrizations with exception of the parameters in parameter_list
         
    allow_non_identical_objects : bool, optional
                                unless set to True, it will
                                not allow running this operation on
                                StimulusDependentData that does not contain
                                only object of the same type
                                  
    Returns
    -------
    (v,object_id) : list,list
                  Where object_id is the new list of objects where the objects in `parameter_list` were
                  'collapsed out' and replaced with None. v is a list of lists. The outer list
                  corresponds in order to the object_id list. The inner list corresponds
                  to the list of data from data_list that mapped on the given object_id.
    """
    assert(len(data_list) == len(object_list))
    if (not allow_non_identical_objects and not identical_parametrized_object_params(object_list)):
        raise ValueError("colapse accepts only object lists of the same type")

    d = collections.OrderedDict()
    for v, s in zip(data_list, object_list):
        if str(s) in d:
            d[str(s)].append(v)
        else:
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
    
    Returns
    -------
    
    List of parameters of the `MozaikParamterrized` objects in `parametrized_objects` that have at least 2 different values within the list.
    """
    if not identical_parametrized_object_params(parametrized_objects):
        raise ValueError("varying_parameters: accepts only MozaikParametrized lists with the same parameters")

    varying_params = collections.OrderedDict()
    for n in parametrized_objects[0].getParams().keys():
        for o in parametrized_objects:
            if o.getParamValue(n) != parametrized_objects[0].getParamValue(n):
                varying_params[n] = True
                break

    return varying_params.keys()

def parameter_value_list(parametrized_objects,param):
    """
    Returns the list of values the given parameter has in the list of MozaikParametrized instances.
    
    Parameters
    ----------
    parametrized_objects : list
                         A list of MozaikParametrized instances.
    
    param : str
          Name of the parameter.
    
    Returns
    -------
    
    A list of different values that the parameter param has across the `MozaikParametrized` objects in `parametrized_objects` list.
    """
    return set([obj.getParamValue(param) for obj in parametrized_objects])
    
def identical_parametrized_object_params(parametrized_objects):
    """
    Check whether the objects have the same parameters.
    
    Returns
    -------
    
    True if all instances of `MozaikParametrized` in  `parametrized_objects` have the same set of parameters (not values!).
    Otherwise returns False.
    """
    for o in parametrized_objects:
        if set(o.getParams().keys()) != set(parametrized_objects[0].getParams().keys()):
                return False
    return True
                
def matching_parametrized_object_params(parametrized_objects,params=None,except_params=None):
    """
    Checks whether `parametrized_objects` have the same parameter values for parameters in `params` or not in `except_params`.
    It is assumed all the parameterized_objects have the same parameters.
    
    Parameters
    ----------
    
    parametrized_objects : list 
                         List of `MozaikParametrized` object to test.
    
    params : list,optional
           List of parameters whose values have to match.

    except_params : list, optional
                  List of parameters that do not have to match.  
                  
    Returns
    -------
                  
    if params != None
        Returns True if all `MozaikParametrized` objects in `parametrized_objects` have the same parameter values for parameters in `params`, otherwise returns False.
    
    if except_params != None
        Returns True if all `MozaikParametrized` objects in `arametrized_objects` have the same parameters except those in `except_params`, otherwise returns False.
        
    if except_params == None and except_params == None
        Returns True if all `MozaikParametrized` objects in `parametrized_objects` have the same parameters, otherwise returns False.
        
    if except_params != None and except_params != None
        Throws exception
    """
    
    if except_params != None and params != None:
        raise ValueError('identical_parametrized_object_params cannot be called with both params and except_params equal to None')
    
    if except_params == None and params == None:
        params = parametrized_objects[0].getParams().keys()
    
    if len(parametrized_objects) == 0:
        return True
    else:
        first =  parametrized_objects[0].getParams()

    if not all([set(first.keys()) == set(p.getParams().keys()) for p in parametrized_objects]):
       raise ValueError('all ADS in the data store do not have the same parameters')

    if except_params != None:
       params = list((set(first.keys())-set(except_params)))
    
    for o in parametrized_objects:
        if [parametrized_objects[0].getParamValue(k) for k in params] != [o.getParamValue(k) for k in params]:
           return False
                    
    return True
    
    
def colapse_to_dictionary(value_list, parametrized_objects, parameter_name):
    """
    Colapse out a parameter `parameter_name` of a list of  `MozaikParametrized` instances.
    
    Parameters
    ----------
    
    value_list : list
               The list of values that correspond to instances of `MozaikParametrized` in `parametrized_objects`.

    parametrized_objects : list
                         The list of values that correspond to instances of `MozaikParametrized`.
                         
    parameter_name : str
                   The parameter  against which to colapse.
                   
    Returns
    -------
    
    D : dict
      Where D.keys() correspond to `parametrized_objects` ids
      with the dimension `parameter_name` colapsed out (and replaced with None).
      The D.values() are tuple of lists (keys,values), where keys is the list of
      values that the parameter_name had in the parametrized_objects, and the values are the
      values from value_list that correspond to the keys.          
    """
    assert(len(value_list) == len(parametrized_objects))
    d = collections.OrderedDict()

    for (v, s) in zip(value_list, parametrized_objects):
        s = MozaikParametrized.idd(s)
        val = s.getParamValue(parameter_name)
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


    

