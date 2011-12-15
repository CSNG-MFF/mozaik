from NeuroTools.parameters import ParameterSet

from MozaikLite.framework import load_component
from MozaikLite.framework.interfaces import MozaikComponent
from MozaikLite.framework.space import VisualSpace, VisualRegion
from MozaikLite.framework.connectors import ExponentialProbabilisticArborization,UniformProbabilisticArborization,GaborConnector, V1RFSpecificProbabilisticArborization
from NeuroTools import signals, plotting, visualization, visual_logging, datastore

from MozaikLite.framework.sheets import Sheet


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
        sh = []
        for sheet in self.sheets:   
            #if self.first_time:
            sheet.record(['spikes', 'v', 'g_syn'])
            #sheet.record('spikes')
            #sheet.record('v')
            #sheet.record('g_syn')
            sh.append(sheet) 
        retinal_input = self.retina.process_visual_input(self.visual_space,stimulus.duration)        
        
        self.run(stimulus.duration)
        
        segments = []
        
        for sheet in self.sheets:    
            if sheet.to_record != None:
                s = sheet.write_neo_object(stimulus.duration)
                segments.append(s)

        self.visual_space.clear()
        self.reset()
        
        for sheet in self.sheets:    
            if sheet.to_record != None:
               sheet.pop._record(None)
        
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
        self.t = 0
        
    def run(self, tstop):
        print ("Simulating the network for %s ms" % tstop)
        self.sim.run(tstop)
        self.t += tstop
        
    def reset(self):
        print ("Resetting the network")
        self.sim.reset()
        self.t=0
    
    def register_sheet(self, sheet):
        self.sheets.append(sheet)
        
    def register_connector(self, connector):
        self.connectors.append(connector)


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
        
        cortex_exc.to_record = 'all'
        cortex_inh.to_record = 'all'
        self.retina.sheets['X_ON'].to_record = 'all' #[0,1,2,3,4,5,6,7,8,9,10]
        self.retina.sheets['X_OFF'].to_record = 'all' #[0,1,2,3,4,5,6,7,8,9,10]
        # initialize projections
        
        UniformProbabilisticArborization(self,cortex_exc,cortex_exc,self.parameters.cortex_exc.ExcExcConnection,'V1ExcExcConnection').connect()
        UniformProbabilisticArborization(self,cortex_exc,cortex_inh,self.parameters.cortex_exc.ExcInhConnection,'V1ExcInhConnection').connect()
        UniformProbabilisticArborization(self,cortex_inh,cortex_exc,self.parameters.cortex_inh.InhExcConnection,'V1InhExcConnection').connect()
        UniformProbabilisticArborization(self,cortex_inh,cortex_inh,self.parameters.cortex_inh.InhInhConnection,'V1InhInhConnection').connect()
        
        #V1RFSpecificProbabilisticArborization(self,cortex_exc,cortex_exc,self.parameters.cortex_exc.ExcExcConnection,'V1ExcExcConnection')
        #V1RFSpecificProbabilisticArborization(self,cortex_exc,cortex_inh,self.parameters.cortex_exc.ExcInhConnection,'V1ExcInhConnection')
        #V1RFSpecificProbabilisticArborization(self,cortex_inh,cortex_exc,self.parameters.cortex_inh.InhExcConnection,'V1InhExcConnection')
        #V1RFSpecificProbabilisticArborization(self,cortex_inh,cortex_inh,self.parameters.cortex_inh.InhInhConnection,'V1InhInhConnection')

        GaborConnector(self,self.retina.sheets['X_ON'],self.retina.sheets['X_OFF'],cortex_exc,self.parameters.cortex_exc.AfferentConnection,'V1AffConnection')
        GaborConnector(self,self.retina.sheets['X_ON'],self.retina.sheets['X_OFF'],cortex_inh,self.parameters.cortex_inh.AfferentConnection,'V1AffInhConnection')
        
        
