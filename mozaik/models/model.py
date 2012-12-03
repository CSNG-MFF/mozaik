"""
docstring goes here

"""
from NeuroTools.parameters import ParameterSet
from mozaik.framework.interfaces import MozaikComponent
from mozaik.framework import load_component
import mozaik
import time

try:
    from mpi4py import MPI
except ImportError:
    MPI = None
if MPI:
    mpi_comm = MPI.COMM_WORLD
MPI_ROOT = 0


logger = mozaik.getMozaikLogger("Mozaik")


class Model(MozaikComponent):
    """
    Model encapsulates a mozaik model and defines interfaces
    with which one can do experiments to the model.

    It has to be able to present stimulus  to the input space
    record the activity in the model to this stimulus sequence
    and return it as a neo object.
    For this purpose, derive your model from this Model class and
    give it a member named self.input_layer in the constructor.
    """

    required_parameters = ParameterSet({
        'name': str,
        'results_dir': str,
        'reset': bool,
        'null_stimulus_period': float,
        'input_space': ParameterSet,
        'input_space_type': str,  # defining the type of input space, visual/auditory/...
    })

    def __init__(self, sim, parameters):
        MozaikComponent.__init__(self, self, parameters)
        self.first_time = True
        self.sim = sim
        self.node = sim.setup(timestep=0.1, min_delay=0.1, max_delay=100.0)  # should have some parameters here
        self.sheets = {}
        self.connectors = {}

        # Set-up the input space
        input_space_type = load_component(self.parameters.input_space_type)
        self.input_space = input_space_type(self.parameters.input_space)
        self.simulator_time = 0

    def present_stimulus_and_record(self, stimulus):

        # create empty arrays in annotations to store the sheet identity of stored data
        for sheet in self.sheets.values():
            sheet.prepare_input(stimulus.duration,self.simulator_time)
            if self.first_time:
                sheet.record()

        self.input_space.clear()
        sim_run_time = self.reset()
        self.input_space.add_object(str(stimulus), stimulus)

                
        sensory_input = self.input_layer.process_input(self.input_space, stimulus, stimulus.duration, self.simulator_time)
        sim_run_time += self.run(stimulus.duration)

        segments = []
        #if (not MPI) or (mpi_comm.rank == MPI_ROOT):
        for sheet in self.sheets.values():    
                if sheet.to_record != None:
                    if self.parameters.reset:
                        s = sheet.write_neo_object()
                        segments.append(s)
                    else:
                        s = sheet.write_neo_object(stimulus.duration)
                        segments.append(s)

        self.first_time = False
        return (segments, sensory_input,sim_run_time)
        
    def run(self, tstop):
        t0 = time.time()
        logger.info("Simulating the network for %s ms" % tstop)
        self.sim.run(tstop)
        logger.info("Finished simulating the network for %s ms" % tstop)
        self.simulator_time += tstop
        return time.time()-t0

    def reset(self):
        logger.info("Resetting the network")
        t0 = time.time()
        if self.parameters.reset:
            self.sim.reset()
            self.simulator_time = 0
        else:
            self.input_layer.provide_null_input(self.input_space,
                                                self.parameters.null_stimulus_period,
                                                self.simulator_time)
            logger.info("Simulating the network for %s ms with blank stimulus" % self.parameters.null_stimulus_period)
            self.sim.run(self.parameters.null_stimulus_period)
            self.simulator_time+=self.parameters.null_stimulus_period
            
            for sheet in self.sheets.values():    
                if sheet.to_record != None:
                   sheet.write_neo_object()
            
        return time.time()-t0    
    

    def register_sheet(self, sheet):
        if sheet.name in self.sheets:
            raise ValueError("ERROR: Sheet %s already registerd" % sheet.name)
        self.sheets[sheet.name] = sheet

    def register_connector(self, connector):
        if connector.name in self.connectors:
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
