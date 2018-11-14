# -*- coding: utf-8 -*-

from parameters import ParameterSet
from mozaik.models import Model
from mozaik.connectors.fast import UniformProbabilisticArborization
from mozaik import load_component

class VogelsAbbott(Model):
    """
    This is implementation of model of self-sustained activity in balanced networks from: 
    Vogels, T. P., & Abbott, L. F. (2005). 
    
    [Signal propagation and logic gating in networks of integrate-and-fire neurons.](http://www.jneurosci.org/content/25/46/10786)
    
    The Journal of neuroscience : the official journal of the Society for Neuroscience, 25(46), 10786–95. 

    DOI: https://doi.org/10.1523/JNEUROSCI.3508-05.2005

    
    """
    required_parameters = ParameterSet({
        'sheets' : ParameterSet({
            'exc_layer' : ParameterSet, 
            'inh_layer' : ParameterSet, 
        })
    })
    
    def __init__(self, sim, num_threads, parameters):
        Model.__init__(self, sim, num_threads, parameters)
        # Load components
        ExcLayer = load_component(self.parameters.sheets.exc_layer.component)
        InhLayer = load_component(self.parameters.sheets.inh_layer.component)
        
        exc = ExcLayer(self, self.parameters.sheets.exc_layer.params)
        inh = InhLayer(self, self.parameters.sheets.inh_layer.params)

        # initialize projections
        UniformProbabilisticArborization(self,'ExcExcConnection',exc,exc,self.parameters.sheets.exc_layer.ExcExcConnection).connect()
        #UniformProbabilisticArborization(self,'ExcInhConnection',exc,inh,self.parameters.sheets.exc_layer.ExcInhConnection).connect()
        #UniformProbabilisticArborization(self,'InhExcConnection',inh,exc,self.parameters.sheets.inh_layer.InhExcConnection).connect()
        #UniformProbabilisticArborization(self,'InhInhConnection',inh,inh,self.parameters.sheets.inh_layer.InhInhConnection).connect()

