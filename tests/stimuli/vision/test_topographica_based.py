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


class TestNoise(TopographicaBasedVisualStimulusTester):

    experiment_seed = 0

    @classmethod
    def setup_class(cls):
        super(TestNoise, cls).setup_class()
        cls.default["time_per_image"] = 2


# grid_size, size_x, grid, background_luminance, density
sparse_noise_params = [
    (10, 10, True, 50, 5.0),
    (15, 15, False, 60, 6.0),
    (5, 5, False, 0.0, 15),
]


class TestSparseNoise(TestNoise):
    def test_init_assert(self):
        with pytest.raises(AssertionError):
            t = topo.SparseNoise(time_per_image=1.4, frame_duration=1.5)

    def reference_frames(self, grid_size, size_x, grid, background_luminance, density):
        time_per_image = self.default["time_per_image"]
        frame_duration = self.default["frame_duration"]
        aux = imagen.random.SparseNoise(
            grid_density=grid_size * 1.0 / size_x,
            grid=grid,
            offset=0,
            scale=2 * background_luminance,
            bounds=BoundingBox(radius=size_x / 2),
            xdensity=density,
            ydensity=density,
            random_generator=numpy.random.RandomState(seed=self.experiment_seed),
        )
        while True:
            aux2 = aux()
            for i in range(time_per_image / frame_duration):
                yield (aux2, [0])

    def actual_frames(self, grid_size, size_x, grid, background_luminance, density):
        snclass = topo.SparseNoise(
            grid_size=grid_size,
            grid=grid,
            background_luminance=background_luminance,
            density=density,
            size_x=size_x,
            size_y=self.default["size_y"],
            location_x=self.default["location_x"],
            location_y=self.default["location_y"],
            time_per_image=self.default["time_per_image"],
            frame_duration=self.default["frame_duration"],
            experiment_seed=self.experiment_seed,
        )
        return snclass._frames

    @pytest.mark.parametrize(
        "grid_size, size_x, grid, background_luminance, density", sparse_noise_params
    )
    def test_frames(self, grid_size, size_x, grid, background_luminance, density):
        self.evaluate(grid_size, size_x, grid, background_luminance, density)


# grid_size, size_x, background_luminance, density
dense_noise_params = [
    (10, 10, 50, 5.0),
    (15, 15, 60, 6.0),
    (5, 5, 0.0, 15),
]


class TestDenseNoise(TestNoise):
    def test_init_assert(self):
        with pytest.raises(AssertionError):
            t = topo.DenseNoise(time_per_image=1.4, frame_duration=1.5)

    def reference_frames(self, grid_size, size_x, background_luminance, density):
        time_per_image = self.default["time_per_image"]
        frame_duration = self.default["frame_duration"]
        aux = imagen.random.DenseNoise(
            grid_density=grid_size * 1.0 / size_x,
            offset=0,
            scale=2 * background_luminance,
            bounds=BoundingBox(radius=size_x / 2),
            xdensity=density,
            ydensity=density,
            random_generator=numpy.random.RandomState(seed=self.experiment_seed),
        )
        while True:
            aux2 = aux()
            for i in range(time_per_image / frame_duration):
                yield (aux2, [0])

    def actual_frames(self, grid_size, size_x, background_luminance, density):
        snclass = topo.DenseNoise(
            grid_size=grid_size,
            grid=False,
            background_luminance=background_luminance,
            density=density,
            size_x=size_x,
            size_y=self.default["size_y"],
            location_x=self.default["location_x"],
            location_y=self.default["location_y"],
            time_per_image=self.default["time_per_image"],
            frame_duration=self.default["frame_duration"],
            experiment_seed=self.experiment_seed,
        )
        return snclass._frames

    @pytest.mark.parametrize(
        "grid_size, size_x, background_luminance, density", dense_noise_params
    )
    def test_frames(self, grid_size, size_x, background_luminance, density):
        self.evaluate(grid_size, size_x, background_luminance, density)
