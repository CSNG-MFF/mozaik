import sys
from parameters import ParameterSet
from mozaik.models import Model
from mozaik.connectors.modular import ModularSamplingProbabilisticConnector
from mozaik import load_component
from mozaik.space import VisualRegion
import mozaik


class ModelMapDependentModularConnectorFunction(Model):
    required_parameters = ParameterSet(
        {
            "sheets": ParameterSet(
                {
                    "sheet": ParameterSet,
                }
            ),
        }
    )

    def __init__(self, sim, num_threads, parameters):
        Model.__init__(self, sim, num_threads, parameters)
        # Load components
        Sheet = load_component(self.parameters.sheets.sheet.component)
        sheet = Sheet(self, self.parameters.sheets.sheet.params)

        ModularSamplingProbabilisticConnector(
            self,
            "RecurrentConnection",
            sheet,
            sheet,
            self.parameters.sheets.sheet.RecurrentConnection,
        ).connect()
