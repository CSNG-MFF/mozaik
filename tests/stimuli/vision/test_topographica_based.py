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
