from NeuroTools.parameters import ParameterSet
from NeuroTools import  visualization, visual_logging, datastore

from mozaik.framework import load_component
from mozaik.framework.interfaces import MozaikComponent
from mozaik.framework.space import VisualSpace, VisualRegion
from mozaik.framework.connectors import ExponentialProbabilisticArborization,UniformProbabilisticArborization,GaborConnector, V1PushPullProbabilisticArborization
from mozaik.framework.sheets import Sheet

import logging

logger = logging.getLogger("mozaik")



class Model(MozaikComponent):
    
    required_parameters = ParameterSet({
        'name':str,
        'screen' : ParameterSet({
                                    'update_interval':float,
                                    'background_luminance':float,
                               }),
        'results_dir' : str, 
        'cortex_inh' : ParameterSet, 
        'cortex_exc' : ParameterSet, 
        'retina_lgn' : ParameterSet,   
        'reset'      : bool,
        'null_stimulus_period' : float,
        'visual_field' : ParameterSet({
                                    'centre':tuple,
                                    'size': tuple,
                               })
    })

    """
    Model encapsulates a mozaik model and defines interfaces 
    with which one can do experiments to the model.
    
    It has to be able to present stimulus  to the visual space
    record the activity in the model to this stimulus sequence 
    and return it as a neo object.
    """
    def present_stimulus_and_record(self,stimulus):
        self.visual_space.add_object(str(stimulus),stimulus)
        
        # create empty arrays in annotations to store the sheet identity of stored data
        sh = []
        for sheet in self.sheets:   
            if self.first_time:
                sheet.record()
            sh.append(sheet) 
        retinal_input = self.retina.process_visual_input(self.visual_space,stimulus.duration,self.simulator_time)        
        self.run(stimulus.duration)
                
        segments = []
        for sheet in self.sheets:    
            if sheet.to_record != None:
                if self.parameters.reset:
                    s = sheet.write_neo_object()
                    segments.append(s)
                else:
                    s = sheet.write_neo_object(stimulus.duration)
                    segments.append(s)
                    
        self.visual_space.clear()
        self.reset()
        self.first_time = False
        return (segments,retinal_input)


    def __init__(self,sim,parameters):
        MozaikComponent.__init__(self, self, parameters);        
        self.first_time=True
        self.sim = sim
        self.node = self.sim.setup() # should have some parameters here
        self.sheets = []
        self.connectors = []

        # Set-up visual stimulus
        self.visual_space = VisualSpace(self.parameters.screen.update_interval,self.parameters.screen.background_luminance)
        self.simulator_time = 0
        
    def run(self, tstop):
        logger.info("Simulating the network for %s ms" % tstop)
        self.sim.run(tstop)
        logger.info("Finished simulating the network for %s ms" % tstop)
        self.simulator_time += tstop
        
    def reset(self):
        logger.info("Resetting the network")
        if self.parameters.reset:
            self.sim.reset()
            self.simulator_time=0
        else:
            self.retina.provide_null_input(self.visual_space,self.parameters.null_stimulus_period,self.simulator_time)
            logger.info("Simulating the network for %s ms with blank stimulus" % self.parameters.null_stimulus_period)
            self.sim.run(self.parameters.null_stimulus_period)
            self.simulator_time+=self.parameters.null_stimulus_period
    
    def register_sheet(self, sheet):
        self.sheets.append(sheet)
        
    def register_connector(self, connector):
        self.connectors.append(connector)

    def neuron_positions(self):
        pos = {}
        for s in self.sheets:
            pos[s.name] = s.pop.positions
        return pos
        
    def neuron_annotations(self):
        neuron_annotations = {}
        for s in self.sheets:
             neuron_annotations[s.name] = s.get_neuron_annotations()
        return neuron_annotations
        
        

class JensModel(Model):
    def __init__(self,simulator,parameters):
        Model.__init__(self,simulator,parameters)        
        # Load components
        CortexExc = load_component(self.parameters.cortex_exc.component)
        CortexInh = load_component(self.parameters.cortex_inh.component)
        RetinaLGN = load_component(self.parameters.retina_lgn.component)
      
        # Build and instrument the network
        self.visual_field = VisualRegion(self.parameters.visual_field.centre, self.parameters.visual_field.size)
        self.retina = RetinaLGN(self, self.parameters.retina_lgn.params)
        
        cortex_exc = CortexExc(self, self.parameters.cortex_exc.params)
        cortex_inh = CortexInh(self, self.parameters.cortex_inh.params)
        
        # which neurons to record
        
        tr = {'spikes' : 'all', 
              'v' : [0,1,2,3,4,5,6,7,8,9,10],
              'gsyn_exc' :[0,1,2,3,4,5,6,7,8,9,10],
              'gsyn_inh' : [0,1,2,3,4,5,6,7,8,9,10],
        }
        
        cortex_exc.to_record = tr #'all'
        cortex_inh.to_record = tr #'all'
        self.retina.sheets['X_ON'].to_record = tr #'all'
        self.retina.sheets['X_OFF'].to_record = tr #'all'

        # initialize projections
        #UniformProbabilisticArborization(self,cortex_exc,cortex_exc,self.parameters.cortex_exc.ExcExcConnection,'V1ExcExcConnection').connect()
        #UniformProbabilisticArborization(self,cortex_exc,cortex_inh,self.parameters.cortex_exc.ExcInhConnection,'V1ExcInhConnection').connect()
        #UniformProbabilisticArborization(self,cortex_inh,cortex_exc,self.parameters.cortex_inh.InhExcConnection,'V1InhExcConnection').connect()
        #UniformProbabilisticArborization(self,cortex_inh,cortex_inh,self.parameters.cortex_inh.InhInhConnection,'V1InhInhConnection').connect()
        
        GaborConnector(self,self.retina.sheets['X_ON'],self.retina.sheets['X_OFF'],cortex_exc,self.parameters.cortex_exc.AfferentConnection,'V1AffConnection')
        GaborConnector(self,self.retina.sheets['X_ON'],self.retina.sheets['X_OFF'],cortex_inh,self.parameters.cortex_inh.AfferentConnection,'V1AffInhConnection')
        
        #V1PushPullProbabilisticArborization(self,cortex_exc,cortex_exc,self.parameters.cortex_exc.ExcExcConnection,'V1ExcExcConnection')
        #V1PushPullProbabilisticArborization(self,cortex_exc,cortex_inh,self.parameters.cortex_exc.ExcInhConnection,'V1ExcInhConnection')
        
        #V1PushPullProbabilisticArborization(self,cortex_inh,cortex_exc,self.parameters.cortex_inh.InhExcConnection,'V1InhExcConnection')
        #V1PushPullProbabilisticArborization(self,cortex_inh,cortex_inh,self.parameters.cortex_inh.InhInhConnection,'V1InhInhConnection')
        
