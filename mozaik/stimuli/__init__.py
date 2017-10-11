"""
This module defines the API for:
    - implementation of stimuli as input to models (see class BaseStimulus)
    - identification of stimulus identity 
    - function helpers for common manipulation with collections of stimuli

Each stimulus is expected to have a list of parameters which have to uniquely identify the stimulus.
This parameterization is done via the MozaikParametrized package (see :class:`mozaik.tools.mozaik_parametrized.MozaikParametrized`)
that allows to specify parameters with the all above requirements.
For Stimuli objects we will allow only SNumber, SInteger and SString parameter types (see :py:mod:`mozaik.tools.mozaik_parametrized`).
These extend the corresponding parameterized parameters to allow specification of units.

Note that each stimulus can be converted back and forth into a string via the str operator and the :func:`mozaik.tools.mozaik_parametrized.MozaikParametrized.idd` function. 
This allows for efficient storing  and manipulation of stimulus identities. 

Note that *all* such parameters defined in the class (and its ancestors) will
be considered as parameters of the BaseStimulus.
"""
import quantities as qt
import numpy
import mozaik
from operator import itemgetter
from mozaik.tools.mozaik_parametrized import MozaikParametrized, SNumber, SInteger, SString, SParameterSet
import collections


logger = mozaik.getMozaikLogger()

class BaseStimulus(MozaikParametrized):
    """
    The abstract stimulus class. It defines the parameters common to all stimuli and
    the list of function each stimulus has to provide.
    """
    frame_duration = SNumber(qt.ms, doc="The duration of single frame")
    duration = SNumber(qt.ms, doc="The duration of stimulus")
    trial = SInteger(doc="The trial of the stimulus")
    direct_stimulation_name = SString(default=None,doc="The name of the artifical stimulation protocol")
    direct_stimulation_parameters = SParameterSet(default=None,doc="The parameters with which the direct stimulation protocol has been initialized")
    
    def __init__(self, **params):
        MozaikParametrized.__init__(self, **params)
        self.input = None
        self._frames = self.frames()
        self.n_frames = numpy.inf  # possibly very dangerous. Don't do 'for i in range(stim.n_frames)'!
        
    def __eq__(self, other):
        """
        Are the name and all parameters of two stimuli are equivallent?
        """
        return self.equalParams(other) and (self.__class__ == other.__class__)

    def number_of_parameters(self):
        """
        Returns number of parameters of the stimulus.
        """
        return len(self.getParams().keys())

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


class InternalStimulus(BaseStimulus):
    """
    Technical stimulus corresponding to no sensory stimulation.

    It is used exclusively in the case when the model receives no sensory stimulus. 
    In such case it is still possible to stimulate the network 'artificially' via the Experiment's direct stimulation 
    facilities (see exc_spike_stimulators etc. in Experiment class).
    
    In such case this stimulus should be associated with the experiment, as it will allow for all the other parts of mozaik to work
    consistently, and will be useful in that it will record the duration of the experiment, the possible information about multiple trials,
    and the identity of the artificial stimulation used. Note that in that case the frame_duration should be set to duration time.
    """
    def __init__(self,**params):
        BaseStimulus.__init__(self,**params)
        assert self.frame_duration == self.duration , "Mozaik requires that frame_duration and duration for InternalStimulus are set to equal values"
    
    def frames(self):
        return None
