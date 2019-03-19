# -*- coding: utf-8 -*-
import sys
import numpy
import mozaik
from parameters import ParameterSet
from mozaik.models import Model
from mozaik.connectors.meta_connectors import GaborConnector
from mozaik.connectors.modular import ModularSamplingProbabilisticConnector
from mozaik import load_component
from mozaik.space import VisualRegion

logger = mozaik.getMozaikLogger()

class PushPullCCModel(Model):
    """
    This model is a re-implementation of a model by Jens Kremkow presented in following paper:
    
    Jens Kremkow, Laurent U. Perrinet, Cyril Monier, Jose-Manuel Alonso, Ad Aertsen, Yves Frégnac and Guillaume S. Masson

    [Push-Pull Receptive Field Organization and Synaptic Depression: 
    Mechanisms for Reliably Encoding Naturalistic Stimuli in V1AffInhConnection](http://journal.frontiersin.org/article/10.3389/fncir.2016.00037/full)
    
    Front. Neural Circuits, 11 May 2016

    DOI: https://doi.org/10.3389/fncir.2016.00037

    The architecture of the model can be found in figure 3C of the above paper:
    ![fig3C](http://www.frontiersin.org/files/Articles/190318/fncir-10-00037-HTML/image_m/fncir-10-00037-g003.jpg)
    """
    
    required_parameters = ParameterSet({
	'sheets' : ParameterSet({
		        'l4_cortex_exc' : ParameterSet, 
        		'l4_cortex_inh' : ParameterSet, 
	        	'retina_lgn' : ParameterSet ,
		}),
        'visual_field' : ParameterSet 
    })
    
    def __init__(self, sim, num_threads, parameters):
        Model.__init__(self, sim, num_threads, parameters)        
        # Load components
        CortexExcL4 = load_component(self.parameters.sheets.l4_cortex_exc.component)
        CortexInhL4 = load_component(self.parameters.sheets.l4_cortex_inh.component)
        
        RetinaLGN = load_component(self.parameters.sheets.retina_lgn.component)
      
        # Build and instrument the network
        self.visual_field = VisualRegion(location_x=self.parameters.visual_field.centre[0],location_y=self.parameters.visual_field.centre[1],size_x=self.parameters.visual_field.size[0],size_y=self.parameters.visual_field.size[1])
        self.input_layer = RetinaLGN(self, self.parameters.sheets.retina_lgn.params)
        cortex_exc_l4 = CortexExcL4(self, self.parameters.sheets.l4_cortex_exc.params)
        cortex_inh_l4 = CortexInhL4(self, self.parameters.sheets.l4_cortex_inh.params)

        # initialize projections
        GaborConnector(self,self.input_layer.sheets['X_ON'],self.input_layer.sheets['X_OFF'],cortex_exc_l4,self.parameters.sheets.l4_cortex_exc.AfferentConnection,'V1AffConnection')
        GaborConnector(self,self.input_layer.sheets['X_ON'],self.input_layer.sheets['X_OFF'],cortex_inh_l4,self.parameters.sheets.l4_cortex_inh.AfferentConnection,'V1AffInhConnection')
        ModularSamplingProbabilisticConnector(self,'V1L4ExcL4ExcConnection',cortex_exc_l4,cortex_exc_l4,self.parameters.sheets.l4_cortex_exc.L4ExcL4ExcConnection).connect()
        ModularSamplingProbabilisticConnector(self,'V1L4ExcL4InhConnection',cortex_exc_l4,cortex_inh_l4,self.parameters.sheets.l4_cortex_exc.L4ExcL4InhConnection).connect()
        ModularSamplingProbabilisticConnector(self,'V1L4InhL4ExcConnection',cortex_inh_l4,cortex_exc_l4,self.parameters.sheets.l4_cortex_inh.L4InhL4ExcConnection).connect()
        ModularSamplingProbabilisticConnector(self,'V1L4InhL4InhConnection',cortex_inh_l4,cortex_inh_l4,self.parameters.sheets.l4_cortex_inh.L4InhL4InhConnection).connect()

