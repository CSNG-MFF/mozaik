"""
This package defines the API for:
    - implementation of stimuli as input to models (see class Stimulus)
    - identification of stimulus identity (without the actual 'data' of the
      stimulus) throughout *mozaik* (see class StimulusID)
      (NOTE: throughout most mozaik (particularly all post processing i.e.
      analysis and plotting) user will interact with StimulusIDs rather than
      Stimulus instances!!)
    - function helpers for common manipulation with collections of stimuli
      (or rather their IDs)

Each stimulus is expected to have a dictionary of parameters which have to
uniquely identify the stimulus.
We implement this by using the Parametrized package that allows to specify
parameters with the all above requirements.
For Stimuli objects we will allow only SNumber, SInteger and SString parameters.
These extend the corresponding parameterized parameters to allow specification
of units (see tools/mozaik_parametrized.py).

Note that *all* such parameters defined in the class (and its ancestors) will
be considered as parameters of the Stimulus.
"""

import quantities as qt
import numpy
import mozaik
from operator import *  # don't do import *
from mozaik.tools.mozaik_parametrized import *  # don't do import *
from mozaik.framework.interfaces import MozaikParametrizeObject
import inspect
from NeuroTools.parameters import ParameterSet


logger = mozaik.getMozaikLogger("Mozaik")


class Stimulus(MozaikParametrized):
    """
    Abstract class.
    """
    frame_duration = SNumber(qt.ms, doc="The duration of single frame")
    duration = SNumber(qt.ms, doc="The duration of stimulus")
    trial = SInteger(doc="The trial of the stimulus")

    def __init__(self, **params):
        MozaikParametrized.__init__(self, **params)
        self.input = None
        self._frames = self.frames()
        self.n_frames = numpy.inf  # possibly very dangerous. Don't do 'for i in range(stim.n_frames)'!
        self.update()

    def __str__(self):
        return str(StimulusID(self))

    def __eq__(self, other):
        return self.equalParams(other) and (self.__class__ == other.__class__)

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


class StimulusID():
    """
    StimulusID is a lightweight object that contains all the parameter info
    of a given Stimulus object. There is a one to one mapping between
    StimulusID instances and Stimulus instances.

    The main purpose for StimulusID is to stand for Stimulus whenever the
    stimulus identitity is required without the need to move around the heavy
    Stimulus objects.

    Unlike Stimulus, StimulusID allows any parameter to be also assigned None.
    This is often used in data analysis to mark parameters that have been
    'computed out' by the analysis (such as averaged out).

    To access the parameter values refer to the param member variable
    To access the units values refer to the units member variable
    To access the periodicity (or lack of it) values refer to the periods
    member variable
    """

    def __str__(self):
        """
        Saves the parameter names and values as a dict
        """
        settings = ['\"%s\":%s' % (name, repr(val))
                    for name, val in self.get_param_values()]
        r = "{ \"name\" :" + "\"" + self.name + "\""+ "," + "\"module_path\" :" + "\"" + self.module_path + "\"" +',' + ", ".join(settings) + "}"
        return r

    def __eq__(self, other):
        return (self.name == other.name
                and self.get_param_values() == other.get_param_values())

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
        z = __import__(self.module_path, globals(), locals(), self.name)
        return  getattr(z, self.name)

    def copy(self):
        return StimulusID(str(self))

    def __init__(self, obj):
        self.units = {}
        self.periods = {}
        self.params = {}
        if isinstance(obj, Stimulus):
            self.name = obj.__class__.__name__
            self.module_path = inspect.getmodule(obj).__name__
            par = obj.params()
            for n, v in obj.get_param_values():
                if n != 'name' and n != 'print_level':
                    self.params[n] = v
                    self.units[n] = par[n].units
                    self.periods[n] = par[n].period
        elif isinstance(obj, dict):
            self.name = obj.pop("name")
            self.module_path = obj.pop("module_path")
            par = self.getStimulusClass().params()
            for n, v in obj.items():
                self.params[n] = v
                self.units[n] = par[n].units
                self.periods[n] = par[n].period
        elif isinstance(obj, str):
            d = eval(obj)
            self.name = d.pop("name")
            self.module_path = d.pop("module_path")
            par = self.getStimulusClass().params()
            for n, v in d.items():
                self.params[n] = v
                self.units[n] = par[n].units
                self.periods[n] = par[n].period
        elif isinstance(obj, StimulusID):
            self.name = obj.name
            self.module_path = obj.module_path
            self.units = obj.units.copy()
            self.periods = obj.periods.copy()
            self.params = obj.params.copy()
        else:
            raise ValueError("obj is not of recognized type (recognized: str, dict, StimulusID or Stimulus)")


"""
Various common operations over StimulusID lists (and associated lists of data) follow.
"""


def _colapse(dd, param):
    d = {}
    for s in dd:
        s1 = StimulusID(s)

        if param not in s1.params.keys():
            raise KeyError('colaps: only stimuli containing parameter [%s] can be collapsed again parameter [%s]' % (param, param))

        s1.params[param]=None
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

    values = [d[k] for k in d.keys()]
    st = [StimulusID(idd) for idd in d.keys()]

    if func != None:
        return ([func(v) for v in values], st)
    else:
        return (values, st)


def varying_parameters(stimulus_ids):
    """
    Find the varying list of params. Can be only applied
    on a stimulus list containing identical type of stimuli.
    """
    if not identical_stimulus_type(stimuli_ids):
        raise ValueError("varying_parameters: accepts only stimuli lists of the same type")

    p = stimuli_ids[0].params.copy()
    varying_params = {}
    for n in p.keys():
        for sid in stimuli_ids:
            if sid.params[n] != p[n]:
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
        val = s.params[parameter_name]
        s.params[parameter_name] = None
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
    stimulus_type = StimulusID(stimuli_list[0]).name
    for st in stimuli_list:
        if StimulusID(st).name != stimulus_type:
            return False
    return True


def find_stimuli(stimulus_name, stimuli_list, data_list=None, **kwargs):
    """
    Returns list of stimuli (and associated data if data_list!=None) of
    stimulus_name type and for which the parameters in kwargs match.

    stimulus_name - the name of the stimulus to filter out
    data_list - the list of values corresponding to stimuli in stimuli_list
    stimuli_list - the list of stimuli corresponding to values in value_list
    **kwargs - the parameter names and values that have to match

    """
    new_st = []
    new_d = []

    no_data = False
    if data_list == None:
        data_list = [[] for z in xrange(0, len(stimuli_list))]
        no_data = True
    else:
        assert(len(data_list) == len(stimuli_list))
    for sid, data in (stimuli_list, data_list):
        sid = StimulusID(sid)
        if sid.name == stimulus_name:
            flag=True
            for n, f in kwargs.items():
                if (not n in sid.params) or f != sid.params[n]:
                    flag = False
                    break
            if flag:
                new_st.append(sid)
                new_d.append(data)

    if no_data:
        return new_st
    else:
        return (new_st, new_d)
