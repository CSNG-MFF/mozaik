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


class SelfSustainedPushPull(Model):

    required_parameters = ParameterSet(
        {
            "sheets": ParameterSet(
                {
                    "l4_cortex_exc": ParameterSet,
                    "l4_cortex_inh": ParameterSet,
                    "l23_cortex_exc": ParameterSet,
                    "l23_cortex_inh": ParameterSet,
                    "retina_lgn": ParameterSet,
                }
            ),
            "visual_field": ParameterSet,
            "only_afferent": bool,
            "l23": bool,
            "feedback": bool,
            "trial": int,
        }
    )

    def __init__(self, sim, num_threads, parameters):
        Model.__init__(self, sim, num_threads, parameters)
        # Load components
        CortexExcL4 = load_component(self.parameters.sheets.l4_cortex_exc.component)
        CortexInhL4 = load_component(self.parameters.sheets.l4_cortex_inh.component)
        if not self.parameters.only_afferent and self.parameters.l23:
            CortexExcL23 = load_component(
                self.parameters.sheets.l23_cortex_exc.component
            )
            CortexInhL23 = load_component(
                self.parameters.sheets.l23_cortex_inh.component
            )

        RetinaLGN = load_component(self.parameters.sheets.retina_lgn.component)

        # Build and instrument the network
        self.visual_field = VisualRegion(
            location_x=self.parameters.visual_field.centre[0],
            location_y=self.parameters.visual_field.centre[1],
            size_x=self.parameters.visual_field.size[0],
            size_y=self.parameters.visual_field.size[1],
        )
        self.input_layer = RetinaLGN(self, self.parameters.sheets.retina_lgn.params)
        cortex_exc_l4 = CortexExcL4(self, self.parameters.sheets.l4_cortex_exc.params)
        cortex_inh_l4 = CortexInhL4(self, self.parameters.sheets.l4_cortex_inh.params)

        if not self.parameters.only_afferent and self.parameters.l23:
            cortex_exc_l23 = CortexExcL23(
                self, self.parameters.sheets.l23_cortex_exc.params
            )
            cortex_inh_l23 = CortexInhL23(
                self, self.parameters.sheets.l23_cortex_inh.params
            )

        # initialize afferent layer 4 projections
        GaborConnector(
            self,
            self.input_layer.sheets["X_ON"],
            self.input_layer.sheets["X_OFF"],
            cortex_exc_l4,
            self.parameters.sheets.l4_cortex_exc.AfferentConnection,
            "V1AffConnection",
        )
        GaborConnector(
            self,
            self.input_layer.sheets["X_ON"],
            self.input_layer.sheets["X_OFF"],
            cortex_inh_l4,
            self.parameters.sheets.l4_cortex_inh.AfferentConnection,
            "V1AffInhConnection",
        )

        # initialize lateral layer 4 projections
        if not self.parameters.only_afferent:

            ModularSamplingProbabilisticConnectorAnnotationSamplesCount(
                self,
                "V1L4ExcL4ExcConnection",
                cortex_exc_l4,
                cortex_exc_l4,
                self.parameters.sheets.l4_cortex_exc.L4ExcL4ExcConnection,
            ).connect()
            ModularSamplingProbabilisticConnectorAnnotationSamplesCount(
                self,
                "V1L4ExcL4InhConnection",
                cortex_exc_l4,
                cortex_inh_l4,
                self.parameters.sheets.l4_cortex_exc.L4ExcL4InhConnection,
            ).connect()

            ModularSamplingProbabilisticConnector(
                self,
                "V1L4InhL4ExcConnection",
                cortex_inh_l4,
                cortex_exc_l4,
                self.parameters.sheets.l4_cortex_inh.L4InhL4ExcConnection,
            ).connect()
            ModularSamplingProbabilisticConnector(
                self,
                "V1L4InhL4InhConnection",
                cortex_inh_l4,
                cortex_inh_l4,
                self.parameters.sheets.l4_cortex_inh.L4InhL4InhConnection,
            ).connect()

            if self.parameters.l23:

                # initialize afferent layer 4 to layer 2/3 projection
                ModularSamplingProbabilisticConnector(
                    self,
                    "V1L4ExcL23ExcConnection",
                    cortex_exc_l4,
                    cortex_exc_l23,
                    self.parameters.sheets.l23_cortex_exc.L4ExcL23ExcConnection,
                ).connect()
                ModularSamplingProbabilisticConnector(
                    self,
                    "V1L4ExcL23InhConnection",
                    cortex_exc_l4,
                    cortex_inh_l23,
                    self.parameters.sheets.l23_cortex_inh.L4ExcL23InhConnection,
                ).connect()

                ModularSamplingProbabilisticConnector(
                    self,
                    "V1L23ExcL23ExcConnection",
                    cortex_exc_l23,
                    cortex_exc_l23,
                    self.parameters.sheets.l23_cortex_exc.L23ExcL23ExcConnection,
                ).connect()
                ModularSamplingProbabilisticConnector(
                    self,
                    "V1L23ExcL23InhConnection",
                    cortex_exc_l23,
                    cortex_inh_l23,
                    self.parameters.sheets.l23_cortex_exc.L23ExcL23InhConnection,
                ).connect()
                ModularSamplingProbabilisticConnector(
                    self,
                    "V1L23InhL23ExcConnection",
                    cortex_inh_l23,
                    cortex_exc_l23,
                    self.parameters.sheets.l23_cortex_inh.L23InhL23ExcConnection,
                ).connect()
                ModularSamplingProbabilisticConnector(
                    self,
                    "V1L23InhL23InhConnection",
                    cortex_inh_l23,
                    cortex_inh_l23,
                    self.parameters.sheets.l23_cortex_inh.L23InhL23InhConnection,
                ).connect()
                if self.parameters.feedback:
                    ModularSamplingProbabilisticConnector(
                        self,
                        "V1L23ExcL4ExcConnection",
                        cortex_exc_l23,
                        cortex_exc_l4,
                        self.parameters.sheets.l23_cortex_exc.L23ExcL4ExcConnection,
                    ).connect()
                    ModularSamplingProbabilisticConnector(
                        self,
                        "V1L23ExcL4InhConnection",
                        cortex_exc_l23,
                        cortex_inh_l4,
                        self.parameters.sheets.l23_cortex_exc.L23ExcL4InhConnection,
                    ).connect()
