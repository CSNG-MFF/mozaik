from collections import OrderedDict

import numpy as np
from parameters import ParameterSet
from pyNN import space

import mozaik
from mozaik.sheets.vision import VisualCorticalGridSheet, VisualCorticalUniformSheet


class FakePopulation:
    def __init__(
        self, size, cellclass, cellparams, structure=None, initial_values=None, label=None
    ):
        self.size = size
        self.structure = structure
        self.label = label
        self.all_cells = np.arange(size)
        self.positions = structure.generate_positions(size)

    def __len__(self):
        return self.size

    def all(self):
        return self.all_cells

    def initialize(self, **kwargs):
        self.initial_values = kwargs

    def set(self, **kwargs):
        self.variable_parameters = kwargs


class FakeSim:
    IF_cond_exp = object()

    class state:
        dt = 1.0

    def __init__(self):
        self.populations = []

    def Population(self, *args, **kwargs):
        population = FakePopulation(*args, **kwargs)
        self.populations.append(population)
        return population

    def native_cell_type(self, cell_model):
        return cell_model


class FakeRNG:
    def uniform(self, low, high, size):
        return np.zeros(size)


class FakeModel:
    def __init__(self):
        self.sim = FakeSim()
        self.sheets = OrderedDict()

    def register_sheet(self, sheet):
        self.sheets[sheet.name] = sheet


def sheet_parameters(**overrides):
    parameters = {
        "name": "grid_sheet",
        "sx": 400.0,
        "sy": 400.0,
        "density": 100.0,
        "mpi_safe": False,
        "magnification_factor": 1000.0,
        "cell": {
            "model": "IF_cond_exp",
            "native_nest": False,
            "params": {},
            "receptors": {},
            "initial_values": {},
        },
        "artificial_stimulators": {},
        "recording_interval": 1.0,
        "recorders": {},
    }
    parameters.update(overrides)
    return ParameterSet(parameters)


def unique_axis_values(values):
    return np.unique(np.round(values, 12))


def test_visual_cortical_grid_sheet_places_square_sheet_on_centered_grid():
    sheet = VisualCorticalGridSheet(FakeModel(), sheet_parameters())

    population = sheet.pop
    positions = population.positions
    xs = unique_axis_values(positions[0])
    ys = unique_axis_values(positions[1])

    assert isinstance(population.structure, space.Grid2D)
    assert population.size == 16
    assert len(xs) == 4
    assert len(ys) == 4
    np.testing.assert_allclose(np.diff(xs), 0.1)
    np.testing.assert_allclose(np.diff(ys), 0.1)
    np.testing.assert_allclose(xs.mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(ys.mean(), 0.0, atol=1e-12)
    assert xs.min() > -sheet.size_x / 2.0
    assert xs.max() < sheet.size_x / 2.0
    assert ys.min() > -sheet.size_y / 2.0
    assert ys.max() < sheet.size_y / 2.0


def test_visual_cortical_grid_sheet_uses_same_linear_density_on_both_axes():
    sheet = VisualCorticalGridSheet(
        FakeModel(), sheet_parameters(sx=600.0, sy=300.0)
    )

    positions = sheet.pop.positions
    xs = unique_axis_values(positions[0])
    ys = unique_axis_values(positions[1])

    assert sheet.pop.size == 18
    assert len(xs) == 6
    assert len(ys) == 3
    np.testing.assert_allclose(np.diff(xs), np.diff(ys)[0])
    np.testing.assert_allclose(np.diff(xs), 0.1)


def test_visual_cortical_uniform_sheet_still_uses_random_structure():
    old_pynn_rng = mozaik.pynn_rng
    mozaik.pynn_rng = FakeRNG()
    try:
        sheet = VisualCorticalUniformSheet(FakeModel(), sheet_parameters(name="uniform"))
    finally:
        mozaik.pynn_rng = old_pynn_rng

    assert isinstance(sheet.pop.structure, space.RandomStructure)
    assert sheet.pop.size == 16
