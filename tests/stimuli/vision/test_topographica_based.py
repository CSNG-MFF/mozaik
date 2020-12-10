from mozaik.stimuli.vision.visual_stimulus import VisualStimulus
import imagen
import imagen.random
from imagen.transferfn import TransferFn
import param
from imagen.image import BoundingBox
import pickle
import numpy
from mozaik.tools.mozaik_parametrized import SNumber, SString
from mozaik.tools.units import cpd
from numpy import pi
from quantities import Hz, rad, degrees, ms, dimensionless
import mozaik.stimuli.vision.topographica_based as topo

import pytest

# Dummy class to get rid of NotImplementedError
class DummyTBVS(topo.TopographicaBasedVisualStimulus):
    def frames(self):
        return []


class TestTopographicaBasedVisualStimulus:
    def test_nontransparent(self):
        t = DummyTBVS(size_x=1, size_y=1, location_x=0.0, location_y=0.0)
        assert not t.transparent


class TopographicaBasedVisualStimulusTester(object):

    num_frames = 100  # Number of frames to test
    default = {}

    @classmethod
    def setup_class(cls):
        cls.default = {
            "duration": 100,
            "frame_duration": 1,
            "background_luminance": 50.0,
            "density": 10.0,
            "location_x": 0.0,
            "location_y": 0.0,
            "size_x": 11.0,
            "size_y": 11.0,
        }

    def reference_frames(self, **params):
        """
        output = tuple(numpy.array : frame, list : optional parameter(s), e.g. orientation))
        """
        raise NotImplementedError("Must be implemented by child class.")

    def actual_frames(self, **params):
        """
        output = tuple(numpy.array : frame, list : optional parameter(s), e.g. orientation))
        """
        raise NotImplementedError("Must be implemented by child class.")

    def test_frames(self, **params):
        pytest.skip("Must be implemented in child class and call evaluate.")

    def evaluate(self, *params):
        rf = self.reference_frames(*params)
        af = self.actual_frames(*params)
        assert self.compare_frames(rf, af, self.num_frames)

    # Compare frames of 2 frame generators
    def compare_frames(self, g0, g1, num_frames):
        for i in range(num_frames):
            f0 = g0.next()
            f1 = g1.next()
            if not (numpy.array_equal(f0[0], f1[0]) and f0[1] == f1[1]):
                return False
        return True
