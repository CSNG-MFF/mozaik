from parameters import ParameterSet
from collections import OrderedDict
from copy import deepcopy

from mozaik.experiments import Experiment
from mozaik.stimuli import InternalStimulus
from mozaik.tools.distribution_parametrization import MozaikExtendedParameterSet
from mozaik.sheets.direct_stimulator import IntraCorticalMicroStimulation


class IntraCorticalMicroStimulationStimulus(Experiment):
    """

    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------

    duration : float
                Duration of the microstimulation.

    sheet_list : int
                The list of sheets in which to do stimulation.

    stimulator_parameters: ParameterSet
                See class IntraCorticalMicroStimulation in direct_stimulator.py
    """

    required_parameters = ParameterSet({
        'duration': float,
        'sheet_list': list,
        "stimulator_parameters": ParameterSet,
    })

    def __init__(self, model, parameters):
        Experiment.__init__(self, model, parameters)

        stimulators = OrderedDict()
        for sheet in parameters.sheet_list:
            _params = MozaikExtendedParameterSet(deepcopy(parameters))
            stimulators[sheet] = [
                IntraCorticalMicroStimulation(model.sheets[sheet], _params.stimulator_parameters)
            ]
        self.direct_stimulation = [stimulators]

        _params = MozaikExtendedParameterSet(deepcopy(parameters))
        self.stimuli.append(
            InternalStimulus(
                frame_duration=_params.duration, 
                duration=_params.duration,
                trial=0,
                direct_stimulation_name='IntraCorticalMicroStimulation',
                direct_stimulation_parameters=_params.stimulator_parameters
            )
        )
