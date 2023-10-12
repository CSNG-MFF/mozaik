import sys
from parameters import ParameterSet
from mozaik.models import Model
from mozaik.connectors.meta_connectors import GaborConnector
from mozaik import load_component
from mozaik.space import VisualRegion
import mozaik


class ModelGaborConnector(Model):
    required_parameters = ParameterSet(
        {
            "sheets": ParameterSet(
                {
                    "sheet_input": ParameterSet,
                    "sheet": ParameterSet,
                }
            ),
        }
    )

    def __init__(self, sim, num_threads, parameters):
        Model.__init__(self, sim, num_threads, parameters)
        # Load components
        SheetInput = load_component(self.parameters.sheets.sheet_input.component)
        Sheet = load_component(self.parameters.sheets.sheet.component)

        self.input_layer = SheetInput(self, self.parameters.sheets.sheet_input.params)
        sheet = Sheet(self, self.parameters.sheets.sheet.params)

        GaborConnector(
            self,
            self.input_layer.sheets["X_ON"],
            self.input_layer.sheets["X_OFF"],
            sheet,
            self.parameters.sheets.sheet.AfferentConnection,
            "AfferentConnection",
        )


class ModelGaborConnectorStretch(Model):
    required_parameters = ParameterSet(
        {
            "sheets": ParameterSet(
                {
                    "sheet_input": ParameterSet,
                    "sheet": ParameterSet,
                }
            ),
        }
    )

    def __init__(self, sim, num_threads, positions, parameters):
        Model.__init__(self, sim, num_threads, parameters)
        # Load components
        SheetInput = load_component(self.parameters.sheets.sheet_input.component)
        Sheet = load_component(self.parameters.sheets.sheet.component)

        self.input_layer = SheetInput(self, self.parameters.sheets.sheet_input.params)
        sheet = Sheet(self, self.parameters.sheets.sheet.params)

        sheet.pop.positions = positions

        GaborConnector(
            self,
            self.input_layer.sheets["X_ON"],
            self.input_layer.sheets["X_OFF"],
            sheet,
            self.parameters.sheets.sheet.AfferentConnection,
            "AfferentConnection",
        )
