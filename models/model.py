from datetime import datetime
import sys,os
import logging

from NeuroTools import init_logging
from NeuroTools import visual_logging
from NeuroTools.parameters import ParameterSet

from MozaikLite.framework import load_component
from MozaikLite.framework.interfaces import MozaikComponent
from MozaikLite.framework.space import VisualSpace, VisualRegion

from MozaikLite.framework.connectors import ExponentialProbabilisticArborization,UniformProbabilisticArborization,GaborConnector
from MozaikLite.stimuli.stimulus_generator import Null

from neo.core.segment import Segment
from NeuroTools import signals, plotting, visualization, visual_logging, datastore

from MozaikLite.framework.sheets import Sheet

logger = logging.getLogger("MozaikLite")

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
        p={'stimulus':'','sheets':[]}

 
        for sheet in self.sheets:   
            if sheet.to_record:
                sheet.record('spikes')
                sheet.record('v')
                p[sheet.name+'_spikes']=[]
                p[sheet.name+'_vm']=[]
                p['sheets'].append(sheet.name)

        s = Segment(**p) 

        self.retina.present_stimulus(self.visual_space,stimulus.duration)        
        self.run(stimulus.duration)
        
        for sheet in self.sheets:    
            if sheet.to_record:
                sheet.write_neo_object(s,stimulus.duration)

        self.retina.end()
        self.visual_space.clear()
        self.reset()
        for sheet in self.sheets:    
            if sheet.to_record:
                sheet.pop._record(None)
        return s


    def __init__(self,sim):
        logger.info("Creating Model object using the %s simulator." % sim.__name__)
        self.sim = sim
        self.visual_field = VisualRegion((0,0), (1,1))
        self.node = self.sim.setup() # should have some parameters here
        self.sheets = []
        self.connectors = []

        
        # Read parameters
        if len(sys.argv) > 1:
            parameters_url = sys.argv[1]
        else:
            parameters_url = "param-ffi/" + self.__class__.__name__ + "/defaults"
        self.parameters = ParameterSet(parameters_url) 
        MozaikComponent.__init__(self, self, self.parameters);        
        
        # Create results directory
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.results_dir = self.parameters.results_dir + timestamp + '/'
        os.mkdir(self.results_dir)
        self.parameters.save(self.results_dir + "parameters", expand_urls=True)
        
        # Set-up logging
        init_logging(self.results_dir + "log", file_level=logging.DEBUG, console_level=logging.DEBUG) # NeuroTools version
        visual_logging.basicConfig(self.results_dir + "visual_log.zip", level=logging.DEBUG)
        
        # Set-up visual stimulus
        self.visual_space = VisualSpace(self.parameters.screen.update_interval,self.parameters.screen.background_luminance)
        self.t = 0
        
    def run(self, tstop):
        #should check if this network is in the cache and, if so,
        #if the desired quantities have already been recorded
        #for tstop or greater
        #if not, call
        logger.info("Simulating the network for %s ms" % tstop)
        self.sim.run(tstop)
        self.t += tstop
        
    def reset(self):
        # can this interferre with cache ?
        logger.info("Resetting the network")
        self.sim.reset()
        self.t=0
    
    def register_sheet(self, sheet):
        self.sheets.append(sheet)
        
    def register_connector(self, connector):
        self.connectors.append(connector)


class JensModel(Model):
    def __init__(self,simulator):
        Model.__init__(self,simulator)        
        # Load components
        CortexExc = load_component(self.parameters.cortex_exc.component)
        CortexInh = load_component(self.parameters.cortex_inh.component)
        RetinaLGN = load_component(self.parameters.retina_lgn.component)
      
        # Build and instrument the network
        self.visual_field = VisualRegion(self.parameters.visual_field.centre, self.parameters.visual_field.size)
        self.retina = RetinaLGN(self, self.parameters.retina_lgn.params)
        
        cortex_exc = CortexExc(self, self.parameters.cortex_exc.params)
        cortex_inh = CortexInh(self, self.parameters.cortex_inh.params)
        
        cortex_exc.to_record = True
        cortex_inh.to_record = True
        

        # initialize projections
        UniformProbabilisticArborization(self,cortex_exc,cortex_exc,self.parameters.cortex_exc.ExcExcConnection,'V1ExcExcConnection')
        UniformProbabilisticArborization(self,cortex_exc,cortex_inh,self.parameters.cortex_exc.ExcInhConnection,'V1ExcInhConnection')
        UniformProbabilisticArborization(self,cortex_inh,cortex_exc,self.parameters.cortex_inh.InhExcConnection,'V1InhExcConnection')
        UniformProbabilisticArborization(self,cortex_inh,cortex_inh,self.parameters.cortex_inh.InhInhConnection,'V1InhInhConnection')

        GaborConnector(self,self.retina.layers["A"].populations['X_ON'],self.retina.layers["A"].populations['X_OFF'],cortex_exc,self.parameters.cortex_exc.AfferentConnection,'V1AffConnection')
        GaborConnector(self,self.retina.layers["A"].populations['X_ON'],self.retina.layers["A"].populations['X_OFF'],cortex_inh,self.parameters.cortex_inh.AfferentConnection,'V1AffInhConnection')
        
        
