"""
This file contains the API for direct stimulation of neurons. 
By direct stimulation here we mean a artificial stimulation that 
would happen during electrophisiological experiment - such a injection
of spikes/currents etc into cells. In mozaik this happens at population level - i.e.
each direct stimulator specifies how the given population is stimulated. In general each population can have several
stimultors.
"""
from mozaik.core import ParametrizedObject
from parameters import ParameterSet
import numpy
import numpy.random
import mozaik
from NeuroTools import stgen
from mozaik import load_component
from pyNN.parameters import Sequence

class DirectStimulator(ParametrizedObject):
      """
      The API for direct stimulation.
      The DirectStimulator specifies how are cells in the assigned population directly stimulated. 
        
      Parameters
      ----------
      parameters : ParameterSet
                   The dictionary of required parameters.
                    
      sheet : Sheet
              The sheet in which to stimulate neurons.
              
      Notes
      -----
      
      By defalut the direct stimulation should ensure that it is mpi-safe - this is especially crucial for 
      stimulators that involve source of randomnes. However, the DirectSimulators also can inspect the mpi_safe
      value of the population to which they are assigned, and if it is False they can switch to potentially 
      more efficient implementation that will however not be reproducible across multi-process simulations.
      
      Important: the functiona inactivate should only temporarily inactivate the stimulator, a subsequent call to prepare_stimulation
      should activate the stimulator back!
      """

      def __init__(self, sheet, parameters):
          ParametrizedObject.__init__(self, parameters)
          self.sheet = sheet
     
      def prepare_stimulation(self,duration,offset):
          """
          Prepares the stimulation during the next period of model simulation lasting `duration` seconds.
          
          Parameters
          ----------
          duration : double (seconds)
                     The period for which to prepare the stimulation
          
          offset : double (seconds)
                   The current simulator time.
                     
          """
          raise NotImplemented 
          
      def inactivate(self,offset):
          """
          Ensures any influences of the stimulation are inactivated for subsequent simulation of the model.

          Parameters
          ----------
          offset : double (seconds)
                   The current simulator time.
          
          Note that a subsequent call to prepare_stimulation should 'activate' the stimulator again.
          """
          raise NotImplemented 



class BackgroundActivityBombardment(DirectStimulator):
    """
    The BackgroundActivityBombardment simulates the poisson distrubated background bombardment of spikes onto a 
    neuron due to the other 'unsimulated' neurons in its pre-synaptic population.
    
    Parameters
    ----------
    parameters : ParameterSet
               The dictionary of required parameters.
                
    sheet : Sheet
          The sheet in which to stimulate neurons.
    
    Other parameters
    ----------------
    
    exc_firing_rate : float
                     The firing rate of external neurons sending excitatory inputs to each neuron of this sheet.
 
    inh_firing_rate : float
                     The firing rate of external neurons sending inhibitory inputs to each neuron of this sheet.
    
    exc_weight : float
                     The weight of the synapses for the excitatory external Poisson input.    

    inh_weight : float
                     The weight of the synapses for the inh external Poisson input.    
    Notes
    -----
    
    Currently the mpi_safe version only works in nest!
    """
    
    
    required_parameters = ParameterSet({
            'exc_firing_rate': float,
            'exc_weight': float,
            'inh_firing_rate': float,
            'inh_weight': float,
    })
        
        
        
    def __init__(self, sheet, parameters):
        DirectStimulator.__init__(self, sheet,parameters)
        
        exc_syn = self.sheet.sim.StaticSynapse(weight=self.parameters.exc_weight)
        inh_syn = self.sheet.sim.StaticSynapse(weight=self.parameters.inh_weight)
        
        if not self.sheet.parameters.mpi_safe:
            from pyNN.nest import native_cell_type        
            if (self.parameters.exc_firing_rate != 0 or self.parameters.exc_weight != 0):
                self.np_exc = self.sheet.sim.Population(1, native_cell_type("poisson_generator"),{'rate': 0})
                self.sheet.sim.Projection(self.np_exc, self.sheet.pop,self.sheet.sim.AllToAllConnector(),synapse_type=exc_syn,receptor_type='excitatory')

            if (self.parameters.inh_firing_rate != 0 or self.parameters.inh_weight != 0):
                self.np_inh = self.sheet.sim.Population(1, native_cell_type("poisson_generator"),{'rate': 0})
                self.sheet.sim.Projection(self.np_inh, self.sheet.pop,self.sheet.sim.AllToAllConnector(),synapse_type=inh_syn,receptor_type='inhibitory')
        
        else:
            if (self.parameters.exc_firing_rate != 0 or self.parameters.exc_weight != 0):
                        self.ssae = self.sheet.sim.Population(self.sheet.pop.size,self.sheet.sim.SpikeSourceArray())
                        seeds=mozaik.get_seeds((self.sheet.pop.size,))
                        self.stgene = [stgen.StGen(rng=numpy.random.RandomState(seed=seeds[i])) for i in numpy.nonzero(self.sheet.pop._mask_local)[0]]
                        self.sheet.sim.Projection(self.ssae, self.sheet.pop,self.sheet.sim.OneToOneConnector(),synapse_type=exc_syn,receptor_type='excitatory')

            if (self.parameters.inh_firing_rate != 0 or self.parameters.inh_weight != 0):
                        self.ssai = self.sheet.sim.Population(self.sheet.pop.size,self.sheet.sim.SpikeSourceArray())
                        seeds=mozaik.get_seeds((self.sheet.pop.size,))
                        self.stgeni = [stgen.StGen(rng=numpy.random.RandomState(seed=seeds[i])) for i in numpy.nonzero(self.sheet.pop._mask_local)[0]]
                        self.sheet.sim.Projection(self.ssai, self.sheet.pop,self.sheet.sim.OneToOneConnector(),synapse_type=inh_syn,receptor_type='inhibitory')

    def prepare_stimulation(self,duration,offset):
        if not self.sheet.parameters.mpi_safe:
           self.np_exc[0].set_parameters(rate=self.parameters.exc_firing_rate)
           self.np_inh[0].set_parameters(rate=self.parameters.inh_firing_rate)
        else:
           if (self.parameters.exc_firing_rate != 0 or self.parameters.exc_weight != 0):
                for j,i in enumerate(numpy.nonzero(self.sheet.pop._mask_local)[0]):
                    pp = self.stgene[j].poisson_generator(rate=self.parameters.exc_firing_rate,t_start=0,t_stop=duration).spike_times
                    a = offset + numpy.array(pp)
                    self.ssae[i].set_parameters(spike_times=Sequence(a.astype(float)))
               
           if (self.parameters.inh_firing_rate != 0 or self.parameters.inh_weight != 0):
                for j,i in enumerate(numpy.nonzero(self.sheet.pop._mask_local)[0]):
                    pp = self.stgene[j].poisson_generator(rate=self.parameters.inh_firing_rate,t_start=0,t_stop=duration).spike_times
                    a = offset + numpy.array(pp)
                    self.ssai[i].set_parameters(spike_times=Sequence(a.astype(float)))
        

        
    def inactivate(self,offset):        
        if not self.sheet.parameters.mpi_safe:
           self.np_exc[0].set_parameters(rate=0)
           self.np_inh[0].set_parameters(rate=0)
            

class Kick(DirectStimulator):
    """
    This stimulator sends a kick of excitatory spikes into a specified subpopulation of neurons.
    
    Parameters
    ----------
    parameters : ParameterSet
               The dictionary of required parameters.
                
    sheet : Sheet
          The sheet in which to stimulate neurons.
    
    Other parameters
    ----------------
    
    exc_firing_rate : float
                     The firing rate of external neurons sending excitatory inputs to each neuron of this sheet.
 
    exc_weight : float
                     The weight of the synapses for the excitatory external Poisson input.    
    
    drive_period : float
                     Period over which the Kick will deposit the full drive defined by the exc_firing_rate, after this time the 
                     firing rates will be linearly reduced to reach zero at the end of stimulation.

    population_selector : ParemeterSet
                        Defines the population selector and its parameters to specify to which neurons in the population the 
                        background activity should be applied. 
                     
    Notes
    -----
    
    Currently the mpi_safe version only works in nest!
    """
    
    
    required_parameters = ParameterSet({
            'exc_firing_rate': float,
            'exc_weight': float,
            'drive_period' : float,
            'population_selector' : ParameterSet({
                    'component' : str,
                    'params' : ParameterSet
                    
            })
            
    })

    def __init__(self, sheet, parameters):
        DirectStimulator.__init__(self, sheet,parameters)
        population_selector = load_component(self.parameters.population_selector.component)
        self.ids = population_selector(sheet,self.parameters.population_selector.params).generate_idd_list_of_neurons()
        d = dict((j,i) for i,j in enumerate(self.sheet.pop.all_cells))
        self.to_stimulate_indexes = [d[i] for i in self.ids]
        
        exc_syn = self.sheet.sim.StaticSynapse(weight=self.parameters.exc_weight)
        if (self.parameters.exc_firing_rate != 0 or self.parameters.exc_weight != 0):
            self.ssae = self.sheet.sim.Population(self.sheet.pop.size,self.sheet.sim.SpikeSourceArray())
            seeds=mozaik.get_seeds((self.sheet.pop.size,))
            self.stgene = [stgen.StGen(rng=numpy.random.RandomState(seed=seeds[i])) for i in self.to_stimulate_indexes]
            self.sheet.sim.Projection(self.ssae, self.sheet.pop,self.sheet.sim.OneToOneConnector(),synapse_type=exc_syn,receptor_type='excitatory') 

    def prepare_stimulation(self,duration,offset):
        if (self.parameters.exc_firing_rate != 0 and self.parameters.exc_weight != 0):
           for j,i in enumerate(self.to_stimulate_indexes):
               if self.parameters.drive_period < duration:
                   z = numpy.arange(self.parameters.drive_period+0.001,duration-100,10)
                   times = [0] + z.tolist() 
                   rate = [self.parameters.exc_firing_rate] + ((1.0-numpy.linspace(0,1.0,len(z)))*self.parameters.exc_firing_rate).tolist()
               else:
                   times = [0]  
                   rate = [self.parameters.exc_firing_rate] 
               pp = self.stgene[j].inh_poisson_generator(numpy.array(rate),numpy.array(times),t_stop=duration).spike_times
               a = offset + numpy.array(pp)
               self.ssae[i].set_parameters(spike_times=Sequence(a.astype(float)))

    def inactivate(self,offset):        
        pass


class Depolarization(DirectStimulator):
    """
    This stimulator injects a constant current into neurons in the population.
    
    Parameters
    ----------
    parameters : ParameterSet
               The dictionary of required parameters.
                
    sheet : Sheet
          The sheet in which to stimulate neurons.
    
    Other parameters
    ----------------
    
    current : float (mA)
                     The current to inject into neurons.

    population_selector : ParemeterSet
                        Defines the population selector and its parameters to specify to which neurons in the population the 
                        background activity should be applied. 
                     
    Notes
    -----
    
    Currently the mpi_safe version only works in nest!
    """
    
    
    required_parameters = ParameterSet({
            'current': float,
            'population_selector' : ParameterSet({
                    'component' : str,
                    'params' : ParameterSet
                    
            })
            
    })
        
    def __init__(self, sheet, parameters):
        DirectStimulator.__init__(self, sheet,parameters)
        
        population_selector = load_component(self.parameters.population_selector.component)
        self.ids = population_selector(sheet,self.parameters.population_selector.params).generate_idd_list_of_neurons()
        self.scs = self.sheet.sim.StepCurrentSource(times=[0.0], amplitudes=[0.0])
        for cell in self.sheet.pop.all_cells:
            cell.inject(self.scs)

    def prepare_stimulation(self,duration,offset):
        self.scs.set_parameters(times=[offset+self.sheet.sim.state.dt*2], amplitudes=[self.parameters.current])
        
    def inactivate(self,offset):
        self.scs.set_parameters(times=[offset+self.sheet.sim.state.dt*2], amplitudes=[0.0])
