"""
This module contains the implementation of a Model API. 

Each simulation contains one model, that overarches the neural network that has been built using 
the basic *mozaik* components (sheets and connectors) and some additional structures such as the recording configurations.
"""
from parameters import ParameterSet
from mozaik.core import BaseComponent
from mozaik import load_component
from mozaik.stimuli import InternalStimulus
import mozaik
import time
import numpy

logger = mozaik.getMozaikLogger()


class Model(BaseComponent):
    """
    Model encapsulates a mozaik model.
    
    Each mozaik model has to derive from this class,
    and in its constructor it has to construct the model from
    the basic *mozaik* building blocks (sheets and connectors),
    and set the variable `input_layer` to the sheet corresponding to the sensory input sheet.
    
    Other parameters
    ----------------
    name : str
         The name of the model.
    
    results_dir : str
                Path to a directory where to store the results.
    
    reset : bool
         If True the pyNN.reset() is used to reset the network between stimulus presentations. 
         Otherwise a blank stimulus is shown for a period of time defined by the parameter null_stimulus_period.
    
    null_stimulus_period : float
                         The length of blank stimulus presentation during the simulation.
    
    input_space : ParameterSet
                The parameters for the InputSpace object that will become the sensory input space for the model.
                
    input_space_type : str
                     The python class of the InputSpace object to use.
                     
    min_delay : float (ms)
                Minimum delay of connections allowed in the simulation. 

    max_delay : float (ms)
                Maximum delay of connections allowed in the simulation. 
    
    time_step : float (ms)
                Length of the single step of the simulation. 
    """

    required_parameters = ParameterSet({
        'name': str,
        'results_dir': str,
        'store_stimuli' : bool,
        'reset': bool,
        'null_stimulus_period': float,
        'input_space': ParameterSet, # can be none - in which case input_space_type is ignored
        'input_space_type': str,  # defining the type of input space, visual/auditory/... it is the class path to the class representing it
        'min_delay' : float,
        'max_delay' : float,
        'time_step' : float
    })

    def __init__(self, sim, num_threads, parameters):
        BaseComponent.__init__(self, self, parameters)
        self.first_time = True
        self.sim = sim
        self.node = sim.setup(timestep=self.parameters.time_step, min_delay=self.parameters.min_delay, max_delay=self.parameters.max_delay, threads=num_threads)  # should have some parameters here
        self.sheets = {}
        self.connectors = {}

        # Set-up the input space
        if self.parameters.input_space != None:
            input_space_type = load_component(self.parameters.input_space_type)
            self.input_space = input_space_type(self.parameters.input_space)
        else:
            self.input_space = None
            
        self.simulator_time = 0

    def present_stimulus_and_record(self, stimulus,artificial_stimulators):
        """
        This method is the core of the model execution control. It ensures that a `stimulus` is presented
        to the model, the simulation is ran for the duration of the stimulus, and all the data recorded during 
        this period are retieved from the simulator. It also makes sure a blank stimulus preceds each stimulus presntation.
        
        Parameters
        ----------
        stimulus : Stimulus
                 Stimulus to be presented.
                 
        artificial_stimulators : dict
                               Dictionary where keys are sheet names, and values are lists of DirectStimulator instances to be applied in the corresponding sheet.
        
        Returns
        -------
        segments : list
                 List of segments holding the recorded data, one per each sheet.
        
        sensory_input : object
                 The 'raw' sensory input that has been shown to the network - the structure of this object depends on the sensory component.
        
        sim_run_time : float (seconds)
                     The biological time of the simulation up to this point (including blank presentations).
                                          
        """
        for sheet in self.sheets.values():
            if self.first_time:
               sheet.record()
        sim_run_time = self.reset()
        for sheet in self.sheets.values():
            sheet.prepare_artificial_stimulation(stimulus.duration,self.simulator_time,artificial_stimulators.get(sheet.name,[]))
        if self.input_space:
            self.input_space.clear()
            if not isinstance(stimulus,InternalStimulus):
                self.input_space.add_object(str(stimulus), stimulus)
                sensory_input = self.input_layer.process_input(self.input_space, stimulus, stimulus.duration, self.simulator_time)
            else:
                self.input_layer.provide_null_input(self.input_space,stimulus.duration,self.simulator_time)
                sensory_input = None                                                    
        else:
            sensory_input = None
        sim_run_time += self.run(stimulus.duration)
        segments = []
        
        for sheet in self.sheets.values():    
            if sheet.to_record != None:
                if self.parameters.reset:
                    s = sheet.get_data()
                    if (not mozaik.mpi_comm) or (mozaik.mpi_comm.rank == mozaik.MPI_ROOT):
                        segments.append(s)
                else:
                    s = sheet.get_data(stimulus.duration)
                    if (not mozaik.mpi_comm) or (mozaik.mpi_comm.rank == mozaik.MPI_ROOT):
                        segments.append(s)

        self.first_time = False
        
        #remove any artificial stimulators 
        for sheet in self.sheets.values():
            for ds in artificial_stimulators.get(sheet.name,[]):
                ds.inactivate(self.simulator_time)
        
        return (segments, sensory_input,sim_run_time)
        
    def run(self, tstop):
        """
        Run's the simulation for tstop time.
        
        Parameters
        ----------
        tstop : float (seconds)
              The duration for which to run the simulation.
        
        Returns
        -------
        time : float (seconds)
             The wall clock time for which the simulator ran.
        """
        t0 = time.time()
        logger.info("Simulating the network for %s ms" % tstop)
        self.sim.run(tstop)
        logger.info("Finished simulating the network for %s ms" % tstop)
        self.simulator_time += tstop
        return time.time()-t0

    def reset(self):
        """
        Rests the network. Depending on the self.parameters.reset this is done either 
        by using the pyNN `reset` function or by presenting a blank stimulus for self.parameters.null_stimulus_period
        seconds.
        """
        logger.debug("Resetting the network")
        t0 = time.time()
        if self.parameters.reset:
            self.sim.reset()
            self.simulator_time = 0
        else:
            if self.parameters.null_stimulus_period != 0:
                for sheet in self.sheets.values():
                    sheet.prepare_artificial_stimulation(self.parameters.null_stimulus_period,self.simulator_time,[])
                
                if self.input_space:
                    self.input_layer.provide_null_input(self.input_space,
                                                        self.parameters.null_stimulus_period,
                                                        self.simulator_time)
                                                        
                logger.info("Simulating the network for %s ms with blank stimulus" % self.parameters.null_stimulus_period)
                self.sim.run(self.parameters.null_stimulus_period)
                self.simulator_time+=self.parameters.null_stimulus_period
                for sheet in self.sheets.values():    
                    if sheet.to_record != None:
                       sheet.get_data()
        return time.time()-t0    
    

    def register_sheet(self, sheet):
        """
        This functions has to called to add a new sheet is added to the model.
        """
        if sheet.name in self.sheets:
            raise ValueError("ERROR: Sheet %s already registerd" % sheet.name)
        self.sheets[sheet.name] = sheet

    def register_connector(self, connector):
        """
        This functions has to called to add a new connector to the model.
        """
        
        if connector.name in self.connectors:
            raise ValueError("ERROR: Connector %s already registerd" % connector.name)
        self.connectors[connector.name] = connector

    def neuron_ids(self):
        """
        Returns the list of ids of neurons in the model.
        """
        ids = {}
        for s in self.sheets.values():
            ids[s.name] = numpy.array([int(a) for a in s.pop.all()])
        return ids

    def sheet_parameters(self):
        """
        Returns the list of ids of neurons in the model.
        """
        p = {}
        for s in self.sheets.values():
            p[s.name] = s.parameters
        return p

        
    def neuron_positions(self):
        """
        Returns the positions of neurons in the model. 
        The positions are return as a dictionary where each key
        corresponds to a sheet name, and the value contains a 2D array of size (2,number_of_neurons)
        containing the x and y coordinates of the neurons in the given sheet.
        """
        pos = {}
        for s in self.sheets.values():
            pos[s.name] = s.pop.positions
        return pos

    def neuron_annotations(self):
        """
        Returns the neuron annotations, as a dictionary with sheet names as keys, and corresponding annotation
        dictionaries as values.
        """
        neuron_annotations = {}
        for s in self.sheets.values():
            neuron_annotations[s.name] = s.get_neuron_annotations()
        return neuron_annotations
