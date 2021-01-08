# encoding: utf-8
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
from mozaik.tools.stgen import StGen
from mozaik import load_component
from pyNN.parameters import Sequence
from mozaik import load_component
import math
from mozaik.tools.circ_stat import circular_dist ,circ_mean
import pylab
from scipy.integrate import odeint
import pickle
import scipy.interpolate
from mpl_toolkits.mplot3d import Axes3D
from mozaik.controller import Global
import pickle

logger = mozaik.getMozaikLogger()

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
        
        exc_syn = self.sheet.sim.StaticSynapse(weight=self.parameters.exc_weight,delay=self.sheet.model.parameters.min_delay)
        inh_syn = self.sheet.sim.StaticSynapse(weight=self.parameters.inh_weight,delay=self.sheet.model.parameters.min_delay)
        
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
                        self.stgene = [StGen(rng=numpy.random.RandomState(seed=seeds[i])) for i in numpy.nonzero(self.sheet.pop._mask_local)[0]]
                        self.sheet.sim.Projection(self.ssae, self.sheet.pop,self.sheet.sim.OneToOneConnector(),synapse_type=exc_syn,receptor_type='excitatory')

            if (self.parameters.inh_firing_rate != 0 or self.parameters.inh_weight != 0):
                        self.ssai = self.sheet.sim.Population(self.sheet.pop.size,self.sheet.sim.SpikeSourceArray())
                        seeds=mozaik.get_seeds((self.sheet.pop.size,))
                        self.stgeni = [StGen(rng=numpy.random.RandomState(seed=seeds[i])) for i in numpy.nonzero(self.sheet.pop._mask_local)[0]]
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
        
        exc_syn = self.sheet.sim.StaticSynapse(weight=self.parameters.exc_weight,delay=self.sheet.model.parameters.min_delay)
        if (self.parameters.exc_firing_rate != 0 or self.parameters.exc_weight != 0):
            self.ssae = self.sheet.sim.Population(self.sheet.pop.size,self.sheet.sim.SpikeSourceArray())
            seeds=mozaik.get_seeds((self.sheet.pop.size,))
            self.stgene = [StGen(rng=numpy.random.RandomState(seed=seeds[i])) for i in self.to_stimulate_indexes]
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
        ids = population_selector(sheet,self.parameters.population_selector.params).generate_idd_list_of_neurons()
        d = dict((j,i) for i,j in enumerate(self.sheet.pop.all_cells))
        to_stimulate_indexes = [d[i] for i in ids]
        
        self.scs = self.sheet.sim.StepCurrentSource(times=[0.0], amplitudes=[0.0])
        for i in to_stimulate_indexes:
            self.sheet.pop.all_cells[i].inject(self.scs)

    def prepare_stimulation(self,duration,offset):
        self.scs.set_parameters(times=[offset+self.sheet.dt*2], amplitudes=[self.parameters.current])
        
    def inactivate(self,offset):
        self.scs.set_parameters(times=[offset+self.sheet.dt*2], amplitudes=[0.0])


class LocalStimulatorArray(DirectStimulator):
    """
    This class assumes there is a regular grid of stimulators (parameters `size` and `spacing` control
    the geometry of the grid), with each stimulator stimulating indiscriminately the local population 
    of neurons in the given sheet. The intensity of stimulation falls of as Gaussian (parameter `itensity_fallof`), 
    and the stimulations from different stimulators add up linearly. 

    The temporal profile of the stimulator is given by function specified in the parameter `stimulating_signal`.
    This function receives the population to be stimulated, the list of coordinates of the stimulators, and any extra user parameters 
    specified in the parameter `stimulating_signal_parameters`. It should return the list of currents that 
    flow out of the stimulators. The function specified in `stimulating_signal` should thus look like this:

    def stimulating_signal_function(population,list_of_coordinates, parameters)

    The rate current changes that the stimulating_signal_function returns is specified by the `current_update_interval`
    parameter.

    Parameters
    ----------
    parameters : ParameterSet
               The dictionary of required parameters.
                
    sheet : Sheet
          The sheet in which to stimulate neurons.
    
    Other parameters
    ----------------
    
    size : float (μm) 
                     The size of the stimulator grid

    spacing : float (μm)
                     The distance between stimulators (the number of stimulators will thus be (size/distance)^2)

    itensity_fallof : float (μm)
                     The sigma of the Gaussian of the stimulation itensity falloff.

    stimulating_signal : str
                     The python path to a function that defines the stimulation.

    stimulating_signal_parameters : ParameterSet
                     The parameters passed to the function specified in  `stimulating_signal`

    current_update_interval : float
                     The interval at which the current is updated. Thus the length of the stimulation is current_update_interval times
                     the number of current values returned by the function specified in the `stimulating_signal` parameter.
    
    depth_sampling_step : float (μm)
                     For optimization reasons we will assume that neurons lie at descrete range of depth spaced at `depth_sampling_step`

    Notes
    -----

    For now this is not mpi optimized.
    """
    
    
    required_parameters = ParameterSet({
            'size': float,
            'spacing' : float,
            'itensity_fallof' : float,
            'stimulating_signal' : str,
            'stimulating_signal_parameters' : ParameterSet,
            'current_update_interval' : float,
            'depth_sampling_step' : float,
            'light_source_light_propagation_data' : str,
    })
    
    def __init__(self, sheet,parameters,shared_scs=None):
        DirectStimulator.__init__(self, sheet,parameters)

        assert math.fmod(self.parameters.size,self.parameters.spacing) < 0.000000001 , "Error the size has to be multiple of spacing!"
        assert math.fmod(self.parameters.size / self.parameters.spacing /2,2) < 0.000000001 , "Error the size and spacing have to be such that they give odd number of elements!"

        
        axis_coors = numpy.arange(0,self.parameters.size+self.parameters.spacing,self.parameters.spacing) - self.parameters.size/2.0 

        n = int(numpy.floor(len(axis_coors)/2.0))
        stimulator_coordinates = numpy.meshgrid(axis_coors,axis_coors)

        pylab.figure(figsize=(42,12))

        #let's load up disperssion data and setup interpolation
        f = open(self.parameters.light_source_light_propagation_data,'r')
        radprofs = pickle.load(f)
        #light_flux_lookup =  scipy.interpolate.RegularGridInterpolator((numpy.arange(0,1080,60),numpy.linspace(0,1,354)*149.701*numpy.sqrt(2)), radprofs, method='linear',bounds_error=False,fill_value=0)
        light_flux_lookup =  scipy.interpolate.RegularGridInterpolator((numpy.arange(0,1080,60),numpy.linspace(0,1,708)*299.7*numpy.sqrt(2)), radprofs, method='linear',bounds_error=False,fill_value=0)

        # the constant translating the data in radprofs to photons/s/cm^2
        K = 2.97e26
        W = 3.9e-10

        # now let's calculate mixing weights, this will be a matrix nxm where n is 
        # the number of neurons in the population and m is the number of stimulators
        x =  stimulator_coordinates[0].flatten()
        y =  stimulator_coordinates[1].flatten()
        xx,yy = self.sheet.vf_2_cs(self.sheet.pop.positions[0],self.sheet.pop.positions[1])
        zeros = numpy.zeros(len(x))
        f = open(Global.root_directory +'positions' + self.sheet.name.replace('/','_') + '.pickle','w')
        pickle.dump((xx,yy),f)
          
        mixing_templates=[]
        for depth in numpy.arange(sheet.parameters.min_depth,sheet.parameters.max_depth+self.parameters.depth_sampling_step,self.parameters.depth_sampling_step):
            temp = numpy.reshape(light_flux_lookup(numpy.transpose([zeros+depth,numpy.sqrt(numpy.power(x,2)  + numpy.power(y,2))])),(2*n+1,2*n+1))
            a  = temp[n,n:]
            cutof = numpy.argmax((numpy.sum(a)-numpy.cumsum(a))/numpy.sum(a) < 0.01)
            assert numpy.shape(temp[n-cutof:n+cutof+1,n-cutof:n+cutof+1]) == (2*cutof+1,2*cutof+1), str(numpy.shape(temp[n-cutof:n+cutof,n-cutof:n+cutof])) + 'vs' + str((2*cutof+1,2*cutof+1))
            mixing_templates.append((temp[n-cutof:n+cutof+1,n-cutof:n+cutof+1],cutof))

        signal_function = load_component(self.parameters.stimulating_signal)
        stimulator_signals,self.scale = signal_function(sheet,stimulator_coordinates[0],stimulator_coordinates[1],self.parameters.current_update_interval,self.parameters.stimulating_signal_parameters )

        #stimulator_signals = numpy.reshape(stimulator_signals,((2*n+1)*(2*n+1),-1))
        
        self.mixed_signals = numpy.zeros((self.sheet.pop.size,numpy.shape(stimulator_signals)[2]),dtype=numpy.float64)
        
        # find coordinates given spacing and shift by half the array size
        nearest_ix = numpy.rint(yy/self.parameters.spacing)+n
        nearest_iy = numpy.rint(xx/self.parameters.spacing)+n
        nearest_iz = numpy.rint((numpy.array(self.sheet.pop.positions[2])-sheet.parameters.min_depth)/self.parameters.depth_sampling_step)

        nearest_ix[nearest_ix<0] = 0
        nearest_iy[nearest_iy<0] = 0
        nearest_ix[nearest_ix>2*n] = 2*n
        nearest_iy[nearest_iy>2*n] = 2*n


        for i in xrange(0,self.sheet.pop.size):
            temp,cutof = mixing_templates[int(nearest_iz[i])]

            ss = stimulator_signals[max(int(nearest_ix[i]-cutof),0):int(nearest_ix[i]+cutof+1),max(int(nearest_iy[i]-cutof),0):int(nearest_iy[i]+cutof+1),:]
            if ss != numpy.array([]):
               temp = temp[max(int(cutof-nearest_ix[i]),0):max(int(2*n+1+cutof-nearest_ix[i]),0),max(int(cutof-nearest_iy[i]),0):max(int(2*n+1+cutof-nearest_iy[i]),0)]
               self.mixed_signals[i,:] = K*W*numpy.dot(temp.flatten(),numpy.reshape(ss,(len(temp.flatten()),-1)))


        lam=numpy.squeeze(numpy.max(self.mixed_signals,axis=1))
        for i in xrange(0,self.sheet.pop.size):
              self.sheet.add_neuron_annotation(i, 'Light activation magnitude(' +self.sheet.name + ',' +  str(self.scale) + ',' +  str(self.parameters.stimulating_signal_parameters.orientation.value)  + ',' +  str(self.parameters.stimulating_signal_parameters.sharpness) + ',' +  str(self.parameters.spacing) + ')', lam[i], protected=True)

        #ax = pylab.subplot(154, projection='3d')
        ax = pylab.subplot(154)
        pylab.gca().set_aspect('equal')
        pylab.title('Activation magnitude (neurons)')
        #ax.scatter(self.sheet.pop.positions[0],self.sheet.pop.positions[1],self.sheet.pop.positions[2],s=10,c=lam,cmap='gray',vmin=0)
        ax.scatter(self.sheet.pop.positions[0],self.sheet.pop.positions[1],s=10,c=lam,cmap='gray',vmin=0)
        ax = pylab.gca()
        #ax.set_zlim(ax.get_zlim()[::-1])
        
        assert numpy.shape(self.mixed_signals) == (self.sheet.pop.size,numpy.shape(stimulator_signals)[2]), "ERROR: mixed_signals doesn't have the desired size:" + str(numpy.shape(self.mixed_signals)) + " vs " +str((self.sheet.pop.size,numpy.shape(stimulator_signals)[1]))
        
        self.stimulation_duration = numpy.shape(self.mixed_signals)[1] * self.parameters.current_update_interval
        
        if shared_scs != None:
           self.scs = shared_scs
        else:
           self.scs = [self.sheet.sim.StepCurrentSource(times=[0.0], amplitudes=[0.0]) for cell in self.sheet.pop.all_cells] 
           for cell,scs in zip(self.sheet.pop.all_cells,self.scs):
               cell.inject(scs)

    def prepare_stimulation(self,duration,offset):
        assert self.stimulation_duration == duration, "stimulation_duration != duration :"  + str(self.stimulation_duration) + " " + str(duration)
        times = numpy.arange(0,self.stimulation_duration,self.parameters.current_update_interval) + offset
        times[0] = times[0] + 3*self.sheet.dt
        for i in xrange(0,len(self.scs)):
            self.scs[i].set_parameters(times=Sequence(times), amplitudes=Sequence(self.mixed_signals[i,:].flatten()),copy=False)

    def inactivate(self,offset):
        for scs in self.scs:
            scs.set_parameters(times=[offset+3*self.sheet.dt], amplitudes=[0.0],copy=False)

def ChRsystem(y,time,X,sampling_period):
          PhoC1toO1 = 1.0993e-19 * 50
          PhoC2toO2 = 7.1973e-20 * 50
          PhoC1toC2 = 1.936e-21 * 50
          PhoC2toC1 = 1.438e-20 * 50

          O1toC1 = 0.125
          O2toC2 = 0.015
          O2toS  = 0.0001
          C2toC1 = 1e-7
          StoC1  = 3e-6

          a = int(numpy.floor(time/sampling_period))
          b = time/sampling_period - a

          if a < len(X)-1:
            I = X[a]*(1-b) + b * X[a+1];
          else:
            I = 0

          O1,O2,C1,C2,S = y

          _O1 = - O1toC1 * O1                    + PhoC1toO1 * I * C1
          _O2 = - O2toC2 * O2                    + PhoC2toO2 * I * C2            - O2toS * O2

          _S  = - StoC1 * S + O2toS * O2

          _C1 = O1toC1 * O1    - PhoC1toO1 * I * C1       - PhoC1toC2 * I * C1    + C2toC1 * C2             + PhoC2toC1 * I * C2            + StoC1 * S
          _C2 = O2toC2 * O2    - C2toC1 * C2              - PhoC2toC1 * I * C2    + PhoC1toC2 * I * C1      - PhoC2toO2 * I * C2

          return (_O1,_O2,_C1,_C2,_S)


class LocalStimulatorArrayChR(LocalStimulatorArray):
      """
      Like *LocalStimulatorArray* but the signal calculated to impinge on a neuron is interpreted as light (photons/s/cm^2)
      impinging on the neuron and the signal is transformed via a model of Channelrhodopsin (courtesy of Quentin Sabatier)
      to give the final injected current. 
      
      Note that we approximate the current by ignoring the voltage dependence of the channels, as it is very expensive 
      to inject conductance in PyNN. The Channelrhodopsin has reverse potential of ~0, and we assume that our neurons 
      sits on average at -60mV to calculate the current. 
      """
      def __init__(self, sheet, parameters,shared_scs=None):
          LocalStimulatorArray.__init__(self, sheet,parameters,shared_scs)
          times = numpy.arange(0,self.stimulation_duration,self.parameters.current_update_interval)
          ax = pylab.subplot(155)
          ax.set_title('Single neuron current injection profile')
          
          ax.plot(times,self.mixed_signals[100,:],'k')
          ax.set_ylabel('photons/cm2/s', color='k')

          for i in xrange(0,len(self.scs)):
              res = odeint(ChRsystem,[0,0,0.8,0.2,0],times,args=(self.mixed_signals[i,:].flatten(),self.parameters.current_update_interval))
              self.mixed_signals[i,:] =  60 * (17.2*res[:,0] + 2.9 * res[:,1])  / 2500 ; # the 60 corresponds to the 60mV difference between ChR reverse potential of 0mV and our expected mean Vm of about 60mV. This happens to end up being in nA which is what pyNN expect for current injection.
          
          for i in xrange(0,self.sheet.pop.size):
                  self.sheet.add_neuron_annotation(i, 'Light activation magnitude ChR(' +  str(self.scale) + ',' +  str(self.parameters.stimulating_signal_parameters.orientation.value) + '_' +  str(self.parameters.stimulating_signal_parameters.sharpness) + '_' +  str(self.parameters.spacing) + ')', numpy.max(self.mixed_signals[i,:]), protected=True)

          ax2 = ax.twinx()
          ax2.plot(times,self.mixed_signals[100,:],'g')
          ax2.set_ylabel('nA', color='g')

          f = open(Global.root_directory +'mixed_signals' + self.sheet.name.replace('/','_') + '_' +  str(self.scale) + '_' +  str(self.parameters.stimulating_signal_parameters.orientation.value) + '_' +  str(self.parameters.stimulating_signal_parameters.sharpness) + '_' +  str(self.parameters.spacing) + '.pickle','w')
          pickle.dump(self.mixed_signals  ,f)
          f.close()

          pylab.savefig(Global.root_directory +'LocalStimulatorArrayTest_' + self.sheet.name.replace('/','_') + '.png')


def test_stimulating_function(sheet,coor_x,coor_y,current_update_interval,parameters):
    z = sheet.pop.all_cells.astype(int)
    vals = numpy.array([sheet.get_neuron_annotation(i,'LGNAfferentOrientation') for i in xrange(0,len(z))])
    mean_orientations = []

    px,py = sheet.vf_2_cs(sheet.pop.positions[0],sheet.pop.positions[1])

    pylab.subplot(151)
    pylab.gca().set_aspect('equal')
    pylab.title('Orientatin preference (neurons)')
    pylab.scatter(px,py,c=vals/numpy.pi,cmap='hsv')
    pylab.hold(True)
    #pylab.scatter(coor_x.flatten(),coor_y.flatten(),c='k',cmap='hsv')

    ors = scipy.interpolate.griddata(zip(px,py), vals, (coor_x, coor_y), method='nearest')

    pylab.subplot(152)
    pylab.title('Orientatin preference (stimulators)')
    pylab.gca().set_aspect('equal')
    pylab.scatter(coor_x.flatten(),coor_y.flatten(),c=ors.flatten(),cmap='hsv')
    signals = numpy.zeros((numpy.shape(coor_x)[0],numpy.shape(coor_x)[1],int(parameters.duration/current_update_interval)))
        
    for i in xrange(0,numpy.shape(coor_x)[0]):
        for j in xrange(0,numpy.shape(coor_x)[0]):
            signals[i,j,int(numpy.floor(parameters.onset_time/current_update_interval)):int(numpy.floor(parameters.offset_time/current_update_interval))] = parameters.scale.value*numpy.exp(-numpy.power(circular_dist(parameters.orientation.value,ors[i][j],numpy.pi),2)/parameters.sharpness)

    pylab.subplot(153)
    pylab.gca().set_aspect('equal')
    pylab.title('Activation magnitude (stimulators)')
    pylab.scatter(coor_x.flatten(),coor_y.flatten(),c=numpy.squeeze(numpy.mean(signals,axis=2)).flatten(),cmap='gray')
    pylab.title(str(parameters.orientation.value))
    #pylab.colorbar()
    return numpy.array(signals),parameters.scale.value

def test_stimulating_function_Naka(sheet,coor_x,coor_y,current_update_interval,parameters):
    z = sheet.pop.all_cells.astype(int)
    vals = numpy.array([sheet.get_neuron_annotation(i,'LGNAfferentOrientation') for i in xrange(0,len(z))])
    mean_orientations = []

    px,py = sheet.vf_2_cs(sheet.pop.positions[0],sheet.pop.positions[1])

    pylab.subplot(151)
    pylab.gca().set_aspect('equal')
    pylab.title('Orientatin preference (neurons)')
    pylab.scatter(px,py,c=vals/numpy.pi,cmap='hsv')
    pylab.hold(True)

    ors = scipy.interpolate.griddata(zip(px,py), vals, (coor_x, coor_y), method='nearest')

    pylab.subplot(152)
    pylab.title('Orientatin preference (stimulators)')
    pylab.gca().set_aspect('equal')
    pylab.scatter(coor_x.flatten(),coor_y.flatten(),c=ors.flatten(),cmap='hsv')
    signals = numpy.zeros((numpy.shape(coor_x)[0],numpy.shape(coor_x)[1],int(parameters.duration/current_update_interval)))
        
    # figure out the light scale 
    rate = parameters.nv_r_max * numpy.power(parameters.contrast.value,parameters.nv_exponent) / (numpy.power(parameters.contrast.value,parameters.nv_exponent) + parameters.nv_c50)
    scale = numpy.power(rate * parameters.cs_c50  / (parameters.cs_r_max - rate), 1/ parameters.cs_exponent)

    for i in xrange(0,numpy.shape(coor_x)[0]):

      
        for j in xrange(0,numpy.shape(coor_x)[0]):
            signals[i,j,int(numpy.floor(parameters.onset_time/current_update_interval)):int(numpy.floor(parameters.offset_time/current_update_interval))] = scale*numpy.exp(-numpy.power(circular_dist(parameters.orientation.value,ors[i][j],numpy.pi),2)/parameters.sharpness)

    pylab.subplot(153)
    pylab.gca().set_aspect('equal')
    pylab.title('Activation magnitude (stimulators)')
    pylab.scatter(coor_x.flatten(),coor_y.flatten(),c=numpy.squeeze(numpy.mean(signals,axis=2)).flatten(),cmap='gray')
    pylab.title(str(parameters.orientation.value))

    return numpy.array(signals),scale