from NeuroTools.parameters import ParameterSet
from mozaik.stimuli.stimulus import StimulusID
from mozaik.stimuli.visual_stimuli import VisualRegion
from mozaik.framework import load_component
from mozaik.framework.interfaces import MozaikComponent
from mozaik.framework.space import VisualSpace, InputSpace
from mozaik.framework.connectors import ExponentialProbabilisticArborization,UniformProbabilisticArborization,GaborConnector, V1PushPullProbabilisticArborization
from mozaik.framework.sheets import Sheet
import logging

try:
    from mpi4py import MPI
except ImportError:
    MPI = None
if MPI:
    mpi_comm = MPI.COMM_WORLD
MPI_ROOT = 0
    

logger = logging.getLogger("mozaik")

class Model(MozaikComponent):
    """
    Model encapsulates a mozaik model and defines interfaces 
    with which one can do experiments to the model.
    
    It has to be able to present stimuli to the respective input_space,
    record the activity in the model to this stimulus sequence 
    and return it as a neo object.
    """
    required_parameters = ParameterSet({
        'name':str,
        'results_dir' : str,
        'reset'      : bool,
        'null_stimulus_period' : float,
        'input_space' : ParameterSet,
        'input_space_type' : str, # defining the type of input space, visual/auditory/...
    })

    def __init__(self,sim,parameters):
        MozaikComponent.__init__(self, self, parameters);
        self.first_time=True
        self.sim = sim
        self.node = self.sim.setup() # should have some parameters here
        self.sheets = {}
        self.connectors = {}

        # Set-up the input space
        input_space_type = load_component(self.parameters.input_space_type)
        self.input_space = input_space_type(self.parameters.input_space)
        self.simulator_time = 0

    def present_stimulus_and_record(self,stimulus):
        self.input_space.add_object(str(stimulus),stimulus)

        # create empty arrays in annotations to store the sheet identity of stored data
        for sheet in self.sheets.values():
            if self.first_time:
                sheet.record()
        sensory_input = self.input_layer.process_input(self.input_space, StimulusID(stimulus), stimulus.duration, self.simulator_time)
        self.run(stimulus.duration)

        segments = []
        if (not MPI) or (mpi_comm.rank == MPI_ROOT):
            for sheet in self.sheets.values():    
                if sheet.to_record != None:
                    if self.parameters.reset:
                        s = sheet.write_neo_object()
                        segments.append(s)
                    else:
                        s = sheet.write_neo_object(stimulus.duration)
                        segments.append(s)

        self.input_space.clear()
        self.reset()
        self.first_time = False
        print 'DEBUG len(segments):', len(segments)
        return (segments, sensory_input)



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
            self.input_layer.provide_null_input(self.input_space, self.parameters.null_stimulus_period,self.simulator_time)
            logger.info("Simulating the network for %s ms with blank stimulus" % self.parameters.null_stimulus_period)
            self.sim.run(self.parameters.null_stimulus_period)
            self.simulator_time+=self.parameters.null_stimulus_period

    def register_sheet(self, sheet):
        if self.sheets.has_key(sheet.name):
           raise ValueError("ERROR: Sheet %s already registerd" % sheet.name)
        self.sheets[sheet.name] = sheet

    def register_connector(self, connector):
        if self.connectors.has_key(connector.name):
           raise ValueError("ERROR: Connector %s already registerd" % connector.name) 
        self.connectors[connector.name] = connector

    def neuron_positions(self):
        pos = {}
        for s in self.sheets.values():
            pos[s.name] = s.pop.positions
        return pos

    def neuron_annotations(self):
        neuron_annotations = {}
        for s in self.sheets.values():
             neuron_annotations[s.name] = s.get_neuron_annotations()
        return neuron_annotations

class JensModel(Model):

    required_parameters = ParameterSet({
        'input_space' : ParameterSet,
        'visual_field' : ParameterSet({
                                'centre':tuple,
                                'size': tuple,
                               }),
        'cortex_inh' : ParameterSet,
        'cortex_exc' : ParameterSet,
        'retina_lgn' : ParameterSet
    })

    def __init__(self,simulator,parameters):
        Model.__init__(self, simulator, parameters)
#        self.input_space = VisualSpace(self.parameters.input_space)

        # Load components
        CortexExc = load_component(self.parameters.cortex_exc.component)
        CortexInh = load_component(self.parameters.cortex_inh.component)
        RetinaLGN = load_component(self.parameters.retina_lgn.component)
      
        # Build and instrument the network
        self.visual_field = VisualRegion(location_x=self.parameters.visual_field.centre[0], \
                location_y=self.parameters.visual_field.centre[1], \
                size_x=self.parameters.visual_field.size[0], \
                size_y=self.parameters.visual_field.size[1], \
                duration=-1., frame_duration=1., trial=0) # to match the Stimulus requirements
        self.retina = RetinaLGN(self, self.parameters.retina_lgn.params)
        self.input_layer = self.retina
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
        GaborConnector(self,self.retina.sheets['X_ON'],self.retina.sheets['X_OFF'],cortex_exc,self.parameters.cortex_exc.AfferentConnection,'V1AffConnection')
        GaborConnector(self,self.retina.sheets['X_ON'],self.retina.sheets['X_OFF'],cortex_inh,self.parameters.cortex_inh.AfferentConnection,'V1AffInhConnection')

        V1PushPullProbabilisticArborization(self,cortex_exc,cortex_exc,self.parameters.cortex_exc.ExcExcConnection,'V1ExcExcConnection')
        V1PushPullProbabilisticArborization(self,cortex_exc,cortex_inh,self.parameters.cortex_exc.ExcInhConnection,'V1ExcInhConnection')
        V1PushPullProbabilisticArborization(self,cortex_inh,cortex_exc,self.parameters.cortex_inh.InhExcConnection,'V1InhExcConnection')
        V1PushPullProbabilisticArborization(self,cortex_inh,cortex_inh,self.parameters.cortex_inh.InhInhConnection,'V1InhInhConnection')

