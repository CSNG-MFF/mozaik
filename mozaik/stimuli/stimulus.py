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
import collections


logger = mozaik.getMozaikLogger("Mozaik")

class BaseStimulus(MozaikParametrized):
    """
    The abstract stimulus class. See the module documentation for more details
    """
    frame_duration = SNumber(qt.ms, doc="The duration of single frame")
    duration = SNumber(qt.ms, doc="The duration of stimulus")
    trial = SInteger(doc="The trial of the stimulus")
    stimulation_name = SString(default="None",doc="The name of the artifical stimulation protocol")
    
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
        Does what it says.
        """
        return len(self.get_param_values())

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


