from NeuroTools.parameters import ParameterSet
from mozaik.stimuli.stimulus_generator import StimulusID
from mozaik.framework.interfaces import MozaikComponent
from mozaik.framework.space import VisualSpace
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
    
    required_parameters = ParameterSet({
        'name':str,
        'screen' : ParameterSet({
                                    'update_interval':float,
                                    'background_luminance':float,
                               }),
        'results_dir' : str, 
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
        for sheet in self.sheets.values():   
            if self.first_time:
                sheet.record()
            sh.append(sheet) 
        retinal_input = self.retina.process_visual_input(self.visual_space,StimulusID(stimulus),stimulus.duration,self.simulator_time)        
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
                    
        self.visual_space.clear()
        self.reset()
        self.first_time = False
        return (segments,retinal_input)


    def __init__(self,sim,parameters):
        MozaikComponent.__init__(self, self, parameters);        
        self.first_time=True
        self.sim = sim
        self.node = self.sim.setup() # should have some parameters here
        self.sheets = {}
        self.connectors = {}

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

