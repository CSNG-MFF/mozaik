# encoding: utf-8
"""
Module containing the implementation of sheets - one of the basic building blocks of *mozaik* models.
"""
import numpy
import mozaik
from mozaik.core import BaseComponent
from mozaik import load_component
from mozaik.tools.distribution_parametrization import PyNNDistribution
from parameters import ParameterSet, UniformDist
from pyNN import space
from pyNN.errors import NothingToWriteError
from string import Template
from neo.core.spiketrain import SpikeTrain
import quantities as pq


logger = mozaik.getMozaikLogger()

class Sheet(BaseComponent):
    """
    Sheet is an abstraction of a 2D continuouse sheet of neurons, roughly
    corresponding to the PyNN Population class with the added spatial structure.

    The spatial position of all cells is kept within the PyNN Population object.
    Each sheet is assumed to be centered around (0,0) origin, corresponding to
    whatever excentricity the model is looking at. The internal representation
    of space is degrees of visual field. Thus x,y coordinates of a cell in all
    sheets correspond to the degrees of visual field this cell is away from the
    origin. However, the sheet and derived classes/methods are supposed to
    accept parameters in units that are most natural for the given parameter and
    recalculate these into the internal degrees of visual field representation.

    As a rule of thumb in mozaik:
       * the units in visual space should be in degrees.
       * the units for cortical space should be in μm.
       
    Other parameters
    ----------------
    
    cell : ParameterSet
         The parametrization of the cell model that all neurons in this sheet will have.
         
    cell.model : str
               The name of the cell model.
    
    cell.params : ParameterSet
               The set of parameters that the given model requires.
               
    cell.initial_values : ParameterSet
                   It can contain a ParameterSet containing the initial values for some of the parameters in cell.params
                   
    mpi_safe : bool
             Whether to set the sheet up to be reproducible in MPI environment. 
             This is computationally less efficient that if it is set to false, but it will
             guaruntee the same results irrespective of the number of MPI process used.
             
    artificial_stimulators : ParameterSet
             Contains a list of ParameterSet objects, one per each :class:`.direct_stimulator.DirectStimulator` object to be created.
             Each contains a parameter 'component' that specifies which :class:`.direct_stimulator.DirectStimulator` to use, and  a 
             parameter 'params' which is a ParameterSet to be passed to that `DirectStimulator`.
    
    name : str
        Name of the sheet.
    
    recorders : ParameterSet
                Parametrization of recorders in this sheet. The recorders ParameterSet will contain as keys the names
                of the different recording configuration user want to have in this sheet. For the format of each recording configuration see notes.

    Notes
    -----
    
    Each recording configuration requires the following parameters:
    
    *variables* 
        tuple of strings specifying the variables to measure (allowd values are: 'spikes' , 'v','gsyn_exc' , 'gsyn_inh' )
    *componnent* 
        the path to the :class:`mozaik.sheets.population_selector.PopulationSelector` class
    *params*
        a ParameterSet containing the parameters for the given :class:`mozaik.sheets.population_selector.PopulationSelector` class 
        
    """

    required_parameters = ParameterSet({
        'cell': ParameterSet({
            'model': str,  # the cell type of the sheet
            'params': ParameterSet,
            'initial_values': ParameterSet,
        }),

        'mpi_safe': bool,
        'artificial_stimulators' : ParameterSet,
        'name': str,
        'recorders' : ParameterSet
    })

    def __init__(self, model, parameters):
        BaseComponent.__init__(self, model, parameters)
        self.sim = self.model.sim
        self.name = parameters.name  # the name of the population
        self.model.register_sheet(self)
        self._pop = None
        
        # We want to be able to define in cell.params the cell parameters as also PyNNDistributions so we can get variably parametrized populations
        # The problem is that the pyNN.Population can accept only scalar parameters. There fore we will remove from cell.params all parameters
        # that are PyNNDistributions, and will initialize them later just after the population is initialized (in property pop())
        self.dist_params = {}
        for k in self.parameters.cell.params.keys():
            if isinstance(self.parameters.cell.params[k],PyNNDistribution):
               self.dist_params[k]=self.parameters.cell.params[k]
               del self.parameters.cell.params[k]
        

    def setup_to_record_list(self):
        """
        Set up the recording configuration.
        """
        self.to_record = {}
        for k in  self.parameters.recorders.keys():
            recording_configuration = load_component(self.parameters.recorders[k].component)
            l = recording_configuration(self,self.parameters.recorders[k].params).generate_idd_list_of_neurons()
            if isinstance(self.parameters.recorders[k].variables,str):
               self.parameters.recorders[k].variables = [self.parameters.recorders[k].variables]
               
            for var in self.parameters.recorders[k].variables:
                self.to_record[var] = list(set(self.to_record.get(var,[])) | set(l))


        for k in self.to_record.keys():
            idds = self.pop.all_cells.astype(int)
            self.to_record[k] = [numpy.flatnonzero(idds == idd)[0] for idd in self.to_record[k]]
            
    def size_in_degrees(self):
        """Returns the x, y size in degrees of visual field of the given area."""
        raise NotImplementedError
        pass

    def pop():
        doc = "The PyNN population holding the neurons in this sheet."

        def fget(self):
            if not self._pop:
                logger.error('Population have not been yet set in sheet: ' +  self.name + '!')
            return self._pop

        def fset(self, value):
            if self._pop:
                raise Exception("Error population has already been set. It is not allowed to do this twice!")
            self._pop = value
            l = value.all_cells.astype(int)

	    
	    self._neuron_annotations = [{} for i in xrange(0, len(value))]
            self.setup_artificial_stimulation()
            self.setup_to_record_list()
            self.setup_initial_values()



        return locals()
    
    pop = property(**pop())  # this will be populated by PyNN population, in the derived classes

    def add_neuron_annotation(self, neuron_number, key, value, protected=True):
        """
        Adds annotation to neuron at index neuron_number.
        
        Parameters
        ----------
        neuron_number : int
                      The index of the neuron in the population to which the annotation will be added.  
        
        key : str
            The name of the annotation
        
        value : object
              The value of the annotation
        
        protected : bool (default=True)
                  If True, the annotation cannot be changed.
        """
        if not self._pop:
            logger.error('Population has not been yet set in sheet: ' + self.name + '!')
        if (key in self._neuron_annotations[neuron_number] and self._neuron_annotations[neuron_number][key][0]):
            logger.warning('The annotation<' + '> for neuron ' + str(neuron_number) + ' is protected. Annotation not updated')
        else:
            self._neuron_annotations[neuron_number][key] = (protected, value)

    def get_neuron_annotation(self, neuron_number, key):
        """
        Retrieve annotation for a given neuron.
        
        Parameters
        ----------
        neuron_number : int
                      The index of the neuron in the population to which the annotation will be added.  
        
        key : str
            The name of the annotation
        
        Returns
        -------
            value : object
                  The value of the annotation
        """

        if not self._pop:
            logger.error('Population has not been yet set in sheet: ' + self.name + '!')
        return self._neuron_annotations[neuron_number][key][1]

    def get_neuron_annotations(self):
        if not self._pop:
            logger.error('Population has not been yet set in sheet: ' +  self.name + '!')

        anns = []
        for i in xrange(0, len(self.pop)):
            d = {}
            for (k, v) in self._neuron_annotations[i].items():
                d[k] = v[1]
            anns.append(d)
        return anns

    def describe(self, template='default', render=lambda t, c: Template(t).safe_substitute(c)):
        context = {
            'name': self.__class__.__name__,
        }
        if template:
            render(template, context)
        else:
            return context

    def record(self):
        if self.to_record != None:
            for variable in self.to_record.keys():
                cells = self.to_record[variable]
                if cells != 'all':
                    self.pop[cells].record(variable)
                else:
                    self.pop.record(variable)

    def get_data(self, stimulus_duration=None):
        """
        Retrieve data recorded in this sheet from pyNN in response to the last presented stimulus.
        
        Parameters
        ----------
        stimulus_duration : float(ms)
                          The length of the last stimulus presentation.
        
        Returns
        -------
        segment : Segment
                The segment holding all the recorded data. See NEO documentation for detail on the format.
        """

        try:
            block = self.pop.get_data(['spikes', 'v', 'gsyn_exc', 'gsyn_inh'],
                                      clear=True)
        except NothingToWriteError, errmsg:
            logger.debug(errmsg)
        
        if (mozaik.mpi_comm) and (mozaik.mpi_comm.rank != mozaik.MPI_ROOT):
           return None
        s = block.segments[-1]
        s.annotations["sheet_name"] = self.name

        # lets sort spike train so that it is ordered by IDs and thus hopefully
        # population indexes
        def compare(a, b):
            return cmp(a.annotations['source_id'], b.annotations['source_id'])

        s.spiketrains = sorted(s.spiketrains, compare)
        if stimulus_duration != None:        
           for i in xrange(0, len(s.spiketrains)):
               s.spiketrains[i] -= s.spiketrains[i].t_start
               s.spiketrains[i].t_stop -= s.spiketrains[i].t_start
               s.spiketrains[i].t_start = 0 * pq.ms
           for i in xrange(0, len(s.analogsignalarrays)):
               s.analogsignalarrays[i].t_start = 0 * pq.ms
       
        return s

    def prepare_artificial_stimulation(self, duration, offset,additional_stimulators):
        """
        Prepares the background noise and artificial stimulation for the population for the stimulus that is 
        about to be presented. 
        
        Parameters
        ----------
        
        duration : float (ms)
                 The duration of the stimulus that will be presented.
        
        additional_stimulators : list
                               List of additional stimulators, defined by the experiment that should be applied during this stimulus. 
                
        offset : float (ms)
               The current time of the simulation.
        """
        for ds in self.artificial_stimulators + additional_stimulators:
            ds.prepare_stimulation(duration,offset)
        

    def setup_artificial_stimulation(self):
        """
        Called once population is created. Sets up the background noise.
        """
        self.artificial_stimulators = []
        for k in  self.parameters.artificial_stimulators.keys():
            direct_stimulator = load_component(self.parameters.artificial_stimulators[k].component)
            self.artificial_stimulators.append(direct_stimulator(self,self.parameters.artificial_stimulators[k].params))

        
    def setup_initial_values(self):
        """
        Called once population is set. Set's up the initial values of the neural model variables.
        """
        # Initial state variables
        self.pop.initialize(**self.parameters.cell.initial_values)
        # Variable cell parameters
        self.pop.set(**self.dist_params)
        #for k,v in self.dist_params.iteritems():
        #    self.pop.rset(k,v)