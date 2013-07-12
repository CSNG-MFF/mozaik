from parameters import ParameterSet
from mozaik.models import Model
from mozaik.connectors.fast import UniformProbabilisticArborization
from mozaik import load_component

class VogelsAbbott(Model):
    
    required_parameters = ParameterSet({
        'exc_layer' : ParameterSet, 
        'inh_layer' : ParameterSet, 
    })
    
    def __init__(self, sim, num_threads, parameters):
        Model.__init__(self, sim, num_threads, parameters)
        # Load components
        ExcLayer = load_component(self.parameters.exc_layer.component)
        InhLayer = load_component(self.parameters.inh_layer.component)
        
        exc = ExcLayer(self, self.parameters.exc_layer.params)
        inh = InhLayer(self, self.parameters.inh_layer.params)

        # initialize projections
        UniformProbabilisticArborization(self,'ExcExcConnection',exc,exc,self.parameters.exc_layer.ExcExcConnection).connect()
        UniformProbabilisticArborization(self,'ExcInhConnection',exc,inh,self.parameters.exc_layer.ExcInhConnection).connect()
        UniformProbabilisticArborization(self,'InhExcConnection',inh,exc,self.parameters.inh_layer.InhExcConnection).connect()
        UniformProbabilisticArborization(self,'InhInhConnection',inh,inh,self.parameters.inh_layer.InhInhConnection).connect()

