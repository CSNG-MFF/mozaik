import pickle
import pytest
import numpy as np
import quantities as qt
from copy import deepcopy
from mozaik.models import Model
from parameters import ParameterSet
from mozaik.sheets.vision import VisualCorticalUniformSheet3D
from mozaik.tools.distribution_parametrization import (
    load_parameters,
    PyNNDistribution,
    MozaikExtendedParameterSet,
)
from mozaik.connectors.vision import MapDependentModularConnectorFunction
from mozaik.tools.circ_stat import circular_dist
import scipy.stats
import pathlib

test_dir = None


class TestCorticalStimulationWithOptogeneticArray:
    model = None
    sheet = None

    @classmethod
    def setup_class(cls):
        from pyNN import nest
        from mozaik.experiments.optogenetic import (
            CorticalStimulationWithOptogeneticArray,
            SingleOptogeneticArrayStimulus,
            OptogeneticArrayStimulusCircles,
            OptogeneticArrayStimulusHexagonalTiling,
            OptogeneticArrayImageStimulus,
            OptogeneticArrayStimulusOrientationTuningProtocol,
        )

        global test_dir, CorticalStimulationWithOptogeneticArray, SingleOptogeneticArrayStimulus, OptogeneticArrayStimulusCircles, OptogeneticArrayStimulusHexagonalTiling, OptogeneticArrayImageStimulus, OptogeneticArrayStimulusOrientationTuningProtocol
        test_dir = str(pathlib.Path(__file__).parent.parent)
        model_params = load_parameters(test_dir + "/sheets/model_params")
        cls.sheet_params = load_parameters(test_dir + "/sheets/exc_sheet_params")
        cls.sheet_params.min_depth = 100
        cls.sheet_params.max_depth = 400
        cls.opt_array_params = load_parameters(test_dir + "/sheets/opt_array_params")
        cls.set_sheet_size(cls, 400)
        cls.model = Model(nest, 8, model_params)
        cls.sheet = VisualCorticalUniformSheet3D(
            cls.model, ParameterSet(cls.sheet_params)
        )
        cls.sheet.record()

    def set_sheet_size(self, size):
        self.sheet_params["sx"] = size
        self.sheet_params["sy"] = size
        self.sheet_params["recorders"]["1"]["params"]["size"] = size
        self.opt_array_params["size"] = size

    def get_coords(self, neuron_ids):
        ac = np.array(list(self.sheet.pop.all()))
        pos = self.sheet.pop.positions[0:2, :]
        return pos[:, [np.where(ac == i)[0][0] for i in neuron_ids]] * 1000  # in µms

    def get_experiment(self, **params):
        pass

    def get_experiment_direct_stimulators(self, **params):
        dss = self.get_experiment(**params).direct_stimulation
        dss = [ds["exc_sheet"][0] for ds in dss]
        for ds in dss:
            ds.stimulator_signals = ds.decompress_array(ds.stimulator_signals)
            ds.mixed_signals_photo = ds.decompress_array(ds.mixed_signals_photo)
        return dss

    def stimulated_neuron_in_radius(self, ds, invert=False):
        ssp = ds.parameters.stimulating_signal_parameters
        center = np.array(ssp.coords).T
        coords = self.get_coords(ds.stimulated_cells)
        d = np.sqrt(((coords - center) ** 2).sum(axis=0))
        if invert:
            return np.all(d >= ssp.radius - ds.parameters.spacing)
        else:
            return np.all(d <= ssp.radius + ds.parameters.spacing)

    def test_initial_asserts(self):
        p = MozaikExtendedParameterSet(
            {
                "sheet_list": ["exc_sheet"],
                "sheet_intensity_scaler": [1.0],
                "sheet_transfection_proportion": [1.0],
                "num_trials": 1,
                "stimulator_array_parameters": deepcopy(self.opt_array_params),
            }
        )

        for param in ["sheet_intensity_scaler", "sheet_transfection_proportion"]:
            with pytest.raises(AssertionError):
                p[param].append(1.0)
                CorticalStimulationWithOptogeneticArray(self.model, p)
            p[param].pop()

            with pytest.raises(AssertionError):
                p[param][0] = -1
                CorticalStimulationWithOptogeneticArray(self.model, p)
            p[param][0] = 1

            if param == "sheet_transfection_proportion":
                with pytest.raises(AssertionError):
                    p[param][0] = 2
                    CorticalStimulationWithOptogeneticArray(self.model, p)
                p[param][0] = 1


class TestSingleOptogeneticArrayStimulus(TestCorticalStimulationWithOptogeneticArray):
    def get_experiment(self, x, y):
        return SingleOptogeneticArrayStimulus(
            self.model,
            MozaikExtendedParameterSet(
                {
                    "sheet_list": ["exc_sheet"],
                    "sheet_intensity_scaler": [1.0],
                    "sheet_transfection_proportion": [1.0],
                    "num_trials": 1,
                    "stimulator_array_parameters": deepcopy(self.opt_array_params),
                    "stimulating_signal": "mozaik.sheets.direct_stimulator.single_pixel",
                    "stimulating_signal_parameters": ParameterSet(
                        {"x": x, "y": y, "intensity": 1, "duration": 2}
                    ),
                }
            ),
        )

    # Cortical to stimulator array coordinates)
    def c2a(self, c):
        return int((c + self.opt_array_params.size / 2) / self.opt_array_params.spacing)

    @pytest.mark.parametrize("x", np.random.randint(-20, 20, 7) * 10)
    @pytest.mark.parametrize("y", np.random.randint(-20, 20, 7) * 10)
    def test_random_pixels(self, x, y):
        size, spacing = self.opt_array_params.size, self.opt_array_params.spacing
        assert size == 400 and spacing == 10
        dss = self.get_experiment_direct_stimulators(x=x, y=y)
        assert dss[0].stimulator_signals[self.c2a(x), self.c2a(y), 0] == 1
        if dss[0].mixed_signals_current.shape[0] > 0:
            coords = self.get_coords(dss[0].stimulated_cells)
            assert np.all(np.isclose(coords[0, :], x, atol=spacing // 2))
            assert np.all(np.isclose(coords[1, :], y, atol=spacing // 2))


class TestOptogeneticArrayStimulusCircles(TestCorticalStimulationWithOptogeneticArray):
    def get_experiment(self, center, inverted):
        return OptogeneticArrayStimulusCircles(
            self.model,
            MozaikExtendedParameterSet(
                {
                    "sheet_list": ["exc_sheet"],
                    "sheet_intensity_scaler": [1.0],
                    "sheet_transfection_proportion": [1.0],
                    "num_trials": 1,
                    "stimulator_array_parameters": deepcopy(self.opt_array_params),
                    "x_center": center[0],
                    "y_center": center[1],
                    "radii": [25, 50, 100, 150],
                    "intensities": [0.5, 1.0],
                    "duration": 150,
                    "onset_time": 0,
                    "offset_time": 75,
                    "inverted": inverted,
                }
            ),
        )

    @pytest.mark.parametrize("center", [[0, 0], [0, 1], [1, 0], [1, 1]])
    @pytest.mark.parametrize("inverted", [False, True])
    def test_stimulated_neurons_in_radius(self, center, inverted):
        dss = self.get_experiment_direct_stimulators(center=center, inverted=inverted)
        for ds in dss:
            assert self.stimulated_neuron_in_radius(ds, inverted)


class TestOptogeneticArrayStimulusHexagonalTiling(
    TestCorticalStimulationWithOptogeneticArray
):
    def get_experiment(self, center, radius):
        return OptogeneticArrayStimulusHexagonalTiling(
            self.model,
            MozaikExtendedParameterSet(
                {
                    "sheet_list": ["exc_sheet"],
                    "sheet_intensity_scaler": [1.0],
                    "sheet_transfection_proportion": [1.0],
                    "num_trials": 1,
                    "stimulator_array_parameters": deepcopy(self.opt_array_params),
                    "x_center": center[0],
                    "y_center": center[1],
                    "radius": radius,
                    "intensities": [0.5],
                    "duration": 150,
                    "onset_time": 0,
                    "offset_time": 75,
                    "angle": 0,
                    "shuffle": False,
                }
            ),
        )

    @pytest.mark.parametrize("center", [[0, 0], [0, 1], [1, 0], [1, 1]])
    @pytest.mark.parametrize("radius", [25, 50])
    def test_stimulated_neurons_in_radius(self, center, radius):
        dss = self.get_experiment_direct_stimulators(center=center, radius=radius)
        for ds in dss:
            assert self.stimulated_neuron_in_radius(ds)

    @pytest.mark.parametrize("radius", [25, 50, 75])
    def test_hexagon_centers(self, radius):
        # Check that all hexagon centers are at least 2*sqrt(3)/2*r distance
        # and that there is at least one hexagon at precisely that distance
        dss = self.get_experiment_direct_stimulators(center=[0, 0], radius=radius)
        centers = np.array(
            [ds.parameters.stimulating_signal_parameters.coords for ds in dss]
        ).squeeze()
        for i in range(centers.shape[0]):
            d = np.sqrt(((centers[i] - centers) ** 2).sum(axis=1))
            d[i] = np.infty
            a = np.isclose(d, radius * np.sqrt(3))
            assert np.any(a)
            d[a] = np.infty
            assert np.all(d >= radius * np.sqrt(3))


class TestOptogeneticArrayImageStimulus(TestCorticalStimulationWithOptogeneticArray):
    def get_experiment(self, im_path, intensity_scaler):
        return OptogeneticArrayImageStimulus(
            self.model,
            MozaikExtendedParameterSet(
                {
                    "sheet_list": ["exc_sheet"],
                    "sheet_intensity_scaler": [intensity_scaler],
                    "sheet_transfection_proportion": [1.0],
                    "num_trials": 1,
                    "stimulator_array_parameters": deepcopy(self.opt_array_params),
                    "intensities": [1.0],
                    "duration": 150,
                    "onset_time": 0,
                    "offset_time": 75,
                    "images_path": im_path,
                }
            ),
        )

    def test_or_map_activation(self):
        MapDependentModularConnectorFunction(
            self.sheet,
            self.sheet,
            ParameterSet(
                {
                    "map_location": test_dir + "/sheets/or_map",
                    "map_stretch": 1,
                    "sigma": 0,
                    "periodic": True,
                }
            ),
        )

        f = open(test_dir + "/sheets/or_map", "rb")
        or_map = pickle.load(f, encoding="latin1")
        f.close()
        np.save(test_dir + "/sheets/or_map.npy", circular_dist(0, or_map, 1))

        dss = self.get_experiment_direct_stimulators(
            im_path=test_dir + "/sheets/or_map.npy", intensity_scaler=1.0
        )
        anns = self.model.neuron_annotations()["exc_sheet"]
        ids = self.model.neuron_ids()["exc_sheet"]
        ors = [circular_dist(0, ann["LGNAfferentOrientation"], np.pi) for ann in anns]
        assert len(dss) == 1
        msp = dss[0].mixed_signals_photo[:, 0]
        assert len(msp) == len(ors)
        corr, _ = scipy.stats.pearsonr(msp, ors)
        assert corr > 0.85

    @pytest.mark.parametrize("intensity_scaler", [0.5, 1.0, 1.5])
    def test_intensity_scaler(self, intensity_scaler):
        MapDependentModularConnectorFunction(
            self.sheet,
            self.sheet,
            ParameterSet(
                {
                    "map_location": test_dir + "/sheets/or_map",
                    "map_stretch": 1,
                    "sigma": 0,
                    "periodic": True,
                }
            ),
        )
        f = open(test_dir + "/sheets/or_map", "rb")
        or_map = pickle.load(f, encoding="latin1")
        f.close()
        np.save(test_dir + "/sheets/or_map.npy", circular_dist(0, or_map, 1))
        dss = self.get_experiment_direct_stimulators(
            im_path=test_dir + "/sheets/or_map.npy", intensity_scaler=1.0
        )
        msp_full = dss[0].mixed_signals_photo.sum()
        dss = self.get_experiment_direct_stimulators(
            im_path=test_dir + "/sheets/or_map.npy", intensity_scaler=intensity_scaler
        )
        msp_is = dss[0].mixed_signals_photo.sum()
        assert np.isclose(msp_is / msp_full, intensity_scaler)


class TestOptogeneticArrayStimulusOrientationTuningProtocol(
    TestCorticalStimulationWithOptogeneticArray
):
    def get_experiment(self, n_orientations):
        return OptogeneticArrayStimulusOrientationTuningProtocol(
            self.model,
            MozaikExtendedParameterSet(
                {
                    "sheet_list": ["exc_sheet"],
                    "sheet_intensity_scaler": [1.0],
                    "sheet_transfection_proportion": [1.0],
                    "num_trials": 1,
                    "stimulator_array_parameters": deepcopy(self.opt_array_params),
                    "num_orientations": n_orientations,
                    "sharpness": 1,
                    "intensities": [1.0],
                    "duration": 150,
                    "onset_time": 0,
                    "offset_time": 75,
                }
            ),
        )

    @pytest.mark.parametrize("n_orientations", range(1, 7))
    def test_or_map_activation(self, n_orientations):
        MapDependentModularConnectorFunction(
            self.sheet,
            self.sheet,
            ParameterSet(
                {
                    "map_location": "tests/sheets/or_map",
                    "map_stretch": 1,
                    "sigma": 0,
                    "periodic": True,
                }
            ),
        )

        dss = self.get_experiment_direct_stimulators(n_orientations=n_orientations)
        orientations = np.linspace(0, np.pi, n_orientations, endpoint=False)
        for i in range(len(orientations)):
            anns = self.model.neuron_annotations()["exc_sheet"]
            ids = self.model.neuron_ids()["exc_sheet"]
            dist = [
                circular_dist(orientations[i], a["LGNAfferentOrientation"], np.pi)
                for a in anns
            ]
            inv_dist = 1 - np.array(dist) / np.pi
            msp = dss[i].mixed_signals_photo[:, 0]
            assert len(msp) == len(inv_dist)
            corr, _ = scipy.stats.pearsonr(msp, inv_dist)
            assert corr > 0.9


class TestOptogeneticArrayStimulusContrastBasedOrientationTuningProtocol:
    # TODO: Enforce small activation difference between this and fullfield gratings
    pass
