"""
This package defines the API for:
    - implementation of stimuli as input to models (see class BaseStimulus)
    - identification of stimulus identity 
    - function helpers for common manipulation with collections of stimuli

Each stimulus is expected to have a dictionary of parameters which have to
uniquely identify the stimulus.
We implement this by using the Parametrized package that allows to specify
parameters with the all above requirements.
For Stimuli objects we will allow only SNumber, SInteger and SString parameters.
These extend the corresponding parameterized parameters to allow specification
of units (see tools/mozaik_parametrized.py).

Note that each stimulus can be converted back and forth into a string via the 
str operator and the load_stimulus class function. The allows for efficient storing 
and manipulation of string identities. 

Note that *all* such parameters defined in the class (and its ancestors) will
be considered as parameters of the BaseStimulus.
"""

import quantities as qt
import numpy
import mozaik
from operator import itemgetter
from mozaik.tools.mozaik_parametrized import MozaikParametrized, SNumber, SInteger, SString
import inspect

logger = mozaik.getMozaikLogger("Mozaik")


def Stimulus(string):
        """
        This function mimics a class constructor. If give a string that was created by the str operators
        on a BaseStimulus object it creates a new instance of the same stimulus.
        
        NOTE: If the string is actually instance of BaseStimulus it is directly returned without raising error.
        """
        if isinstance(string,BaseStimulus):
           return string  
        assert isinstance(string,str)
        
        params = eval(string)
        name = params.pop("name")
        module_path = params.pop("module_path")
        z = __import__(module_path, globals(), locals(), name)
        cls = getattr(z,name)
        return cls(**params)


class BaseStimulus(MozaikParametrized):
    """
    The abstract stimulus class. See the module documentation for more details
    """
    frame_duration = SNumber(qt.ms, doc="The duration of single frame")
    duration = SNumber(qt.ms, doc="The duration of stimulus")
    trial = SInteger(doc="The trial of the stimulus")
    name = SString(doc="The name of the stimulus that is by default set to the name of the class. DO NOT CHANGE")
    
    def __init__(self, **params):
        MozaikParametrized.__init__(self, **params)
        self.input = None
        self._frames = self.frames()
        self.name = self.__class__.__name__
        self.module_path = inspect.getmodule(self).__name__
        self.n_frames = numpy.inf  # possibly very dangerous. Don't do 'for i in range(stim.n_frames)'!

    def __str__(self):
        """
        Turn the stimulus to string - this can be used to store the stimulus as a simple string and it can be restored (minus the state) 
        via the load_stimulus function.
        """
        settings = ['\"%s\":%s' % (name, repr(val)) for name, val in self.get_param_values()]
        r = "{ \"name\" :" + "\"" + self.name + "\""+ "," + "\"module_path\" :" + "\"" + self.module_path + "\"" +',' + ", ".join(settings) + "}"
        return r
        
    def __eq__(self, other):
        """
        Are the name and all parameters of two stimuli are equivallent?
        """
        return self.equalParams(other) and (self.__class__ == other.__class__)

    def number_of_parameters(self):
        """
        Does what it says.
        """
        return len(self.get_param_values())

    def copy(self):
        """
        Make a copy of the stimulus (note this does not preserve state).
        """
        return Stimulus(str(self))

    def frames(self):
        """
        Return a generator which yields the frames of the stimulus in sequence.
        Each frame is returned as a tuple `(frame, variables)` where
        `frame` is a numpy array containing the stimulus at the given time and
        `variables` is a list of variable values (e.g., orientation) associated
        with that frame.

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

"""
Various common operations over BaseStimulus lists (and associated lists of data) follow.
"""

def _colapse(dd, param):
    d = {}
    for s in dd:
        s1 = Stimulus(s)

        if param not in s1.params().keys():
            raise KeyError('colaps: only stimuli containing parameter [%s] can be collapsed again parameter [%s]' % (param, param))

        setattr(s1,param,None)
        s1 = str(s1)
        if s1 in d:
           d[s1].extend(dd[s])
        else:
           d[s1] = dd[s]
            
    return d


def colapse(data_list, stimuli_list, func=None, parameter_list=[],
            allow_non_identical_stimuli=False):
    """
    It collapses the stimuli_list and associated data_list acording to stimuli
    against parameters in parameter_list. This means that the new list of
    parameters (and associated datalist) will contain one stimulus for each
    combination of parameter values not among the paramters against which to
    collapse. Each such stimulus will be associated with a list of data that
    corresponded to stimuli with the same parameter values, but any values for
    the parameters against which one is collapsing.
    The collapsed parameters in the stimuli_list will be replaced with None.

    The function returns a tuple of lists (v,stimuli_id), where stimuli_id is
    the new list of stimuli where the stimuli in parameter_list were
    'collapsed out' and replaced with None. v is a list of lists. The outer list
    corresponding in order to the stimuli_id list. The inner list corresponds
    to the list of data from data_list that mapped on the given stimuli_id.

    If func != None, func is applied to each member of v.

    data_list - the list of data corresponding to stimuli in stimuli_list
    stimuli_list - the list of stimuli corresponding to data in data_list
    parameter_list - the list of parameter names against which to collapse the data
    func - the func to be applied to the lists formed by data associated with the
           same stimuli parametrizations with exception of the parameters in parameter_list
    allow_non_identical_stimuli - (default=False) unless set to True, it will
                                  not allow running this operation on
                                  StimulusDependentData that does not contain
                                  only stimuli of the same type
    """
    assert(len(data_list) == len(stimuli_list))
    if (not allow_non_identical_stimuli and not identical_stimulus_type(stimuli_list)):
        raise ValueError("colapse accepts only stimuli lists of the same type")

    d = {}
    for v, s in zip(data_list, stimuli_list):
        d[str(s)]=[v]

    for param in parameter_list:
        d = _colapse(d, param)

    values = d.values()
    st = [Stimulus(idd) for idd in d.keys()]
    
    
    if func != None:
        return ([func(v) for v in values], st)
    else:
        return (values, st)


def varying_parameters(stimulus_ids):
    """
    Find the varying list of params. Can be only applied
    on a stimulus list containing identical type of stimuli.
    """
    if not identical_stimulus_type(stimulus_ids):
        raise ValueError("varying_parameters: accepts only stimulus lists of the same type")

    p = stimulus_ids[0].params().keys()
    varying_params = {}
    for n in p.keys():
        for sid in stimulus_ids:
            if getattr(sid,n) != getattr(stimulus_ids[0],n):
                varying_params[n] = True
                break
    return varying_params


def colapse_to_dictionary(value_list, stimuli_list, parameter_name):
    """
    Returns dictionary D where D.keys() correspond to stimuli_ids
    with the dimension 'parameter name' colapsed out (and replaced with None).

    The D.values() are tuple of lists (keys,values), where keys is the list of
    values that the parameter_name had in the stimuli_list, and the values are the
    values from value_list that correspond to the keys.
    """
    assert(len(value_list) == len(stimuli_list))
    d = {}

    for (v, s) in zip(value_list, stimuli_list):
        s = s.copy()
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


def identical_stimulus_type(stimuli_list):
    """
    Returns true if all stimuli in stimulus_list are of the same type, else returns False.
    """
    stimulus_type = Stimulus(stimuli_list[0]).name
    for st in stimuli_list:
        if Stimulus(st).name != stimulus_type:
            return False
    return True
