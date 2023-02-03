import sys
from parameters import ParameterSet
from mozaik.models import Model
from mozaik.connectors.meta_connectors import GaborConnector
from mozaik.connectors.modular import (
    ModularSamplingProbabilisticConnector,
    ModularSamplingProbabilisticConnectorAnnotationSamplesCount,
)
from mozaik import load_component
from mozaik.space import VisualRegion
import mozaik


class ModelLocalModule(Model):
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
        Sheet_lm = load_component(self.parameters.sheets.sheet_lm.component)

        sheet = Sheet(self, self.parameters.sheets.sheet.params)
        sheet_lm = Sheet_lm(self, self.parameters.sheets.sheet_lm.params)
        ModularSamplingProbabilisticConnector(
            self,
            "RecurrentConnection",
            sheet,
            sheet,
            self.parameters.sheets.sheet.RecurrentConnection,
        ).connect()
        ModularSamplingProbabilisticConnector(
            self,
            "RecurrentConnectionLM",
            sheet_lm,
            sheet_lm,
            self.parameters.sheets.sheet_lm.RecurrentConnectionLM,
        ).connect()
