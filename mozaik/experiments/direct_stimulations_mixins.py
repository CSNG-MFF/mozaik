from mozaik.tools.distribution_parametrization import ParameterWithUnitsAndPeriod, MozaikExtendedParameterSet
from mozaik.sheets.direct_stimulator import Depolarization
from collections import OrderedDict



def add_per_stimulus_current_injection(exp,stimulation_configuration,stimulation_sheet,stimulation_current):
    r"""
    To experiment *exp*, add an injection of current of magnitude *stimulation_current* to neurons from sheet *stimulation_sheet*
    selected based on population selector *stimulation_configuration*.
    """
    # this will work correctly only if no direct stimulators were already added.
    assert exp.direct_stimulation==None
    exp.direct_stimulation = []
    for s in exp.stimuli:
            d  = OrderedDict()
            p = MozaikExtendedParameterSet({
                                'population_selector' : stimulation_configuration,
                                'current' : stimulation_current
                               })

            d[stimulation_sheet] = [Depolarization(exp.model.sheets[stimulation_sheet],p)]

            exp.direct_stimulation.append(d)     

            p['sheet'] = stimulation_sheet
            s.direct_stimulation_name='Injection'
            s.direct_stimulation_parameters = p

    return exp
