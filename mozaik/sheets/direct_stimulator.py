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
import numpy as np
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
import matplotlib
from mozaik.analysis.analysis import SingleValue, AnalogSignalList
from neo.core.analogsignal import AnalogSignal as NeoAnalogSignal
import quantities as qt
from mozaik.tools.units import *
import io
from numba import jit

from builtins import zip

from mpi4py import MPI

mpi_comm = MPI.COMM_WORLD


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

    By default the direct stimulation should ensure that it is mpi-safe - this is especially crucial for
    stimulators that involve source of randomness. However, the DirectSimulators also can inspect the mpi_safe
    value of the population to which they are assigned, and if it is False they can switch to potentially
    more efficient implementation that will however not be reproducible across multi-process simulations.

    Important: the function inactivate should only temporarily inactivate the stimulator, a subsequent call to prepare_stimulation
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

    def save_to_datastore(self,data_store,stimulus):
        """
        Save direct stimulation data to the datastore, to be used for analysis and
        visualization.

        Parameters
        ----------
        data_store : DataStore
                   The data store into which to store the direct stimulation data.
        """
        pass


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
                self.np_exc = self.sheet.sim.Population(len(self.sheet.pop), native_cell_type("poisson_generator"),{'rate': 0})
                self.sheet.sim.Projection(self.np_exc, self.sheet.pop,self.sheet.sim.OneToOneConnector(),synapse_type=exc_syn,receptor_type='excitatory')
                #self.np_exc = self.sheet.sim.Population(1, native_cell_type("poisson_generator"),{'rate': 0})
                #self.sheet.sim.Projection(self.np_exc, self.sheet.pop,self.sheet.sim.AllToAllConnector(),synapse_type=exc_syn,receptor_type='excitatory')

            if (self.parameters.inh_firing_rate != 0 or self.parameters.inh_weight != 0):
                self.np_inh = self.sheet.sim.Population(len(self.sheet.pop), native_cell_type("poisson_generator"),{'rate': 0})
                self.sheet.sim.Projection(self.np_exc, self.sheet.pop,self.sheet.sim.OneToOneConnector(),synapse_type=exc_syn,receptor_type='excitatory')
                #self.np_inh = self.sheet.sim.Population(1, native_cell_type("poisson_generator"),{'rate': 0})
                #self.sheet.sim.Projection(self.np_inh, self.sheet.pop,self.sheet.sim.AllToAllConnector(),synapse_type=inh_syn,receptor_type='inhibitory')
        
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
            for i in range(len(self.np_exc)):
                if self.np_exc._mask_local[i]:
                    self.np_exc[i].set_parameters(rate=self.parameters.exc_firing_rate)
                if self.np_inh._mask_local[i]:
                    self.np_inh[i].set_parameters(rate=self.parameters.inh_firing_rate)
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
            for i in range(len(self.np_exc)):
                if self.np_exc[i]._mask_local:
                    self.np_exc[i].set_parameters(rate=0)
                if self.np_inh[i]._mask_local:
                    self.np_inh[i].set_parameters(rate=0)
            

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
    
    Currently this experiment does not work with MPI
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

        exc_syn = self.sheet.sim.StaticSynapse(weight=self.parameters.exc_weight,delay=2*self.sheet.model.parameters.min_delay)
        if (self.parameters.exc_firing_rate != 0 or self.parameters.exc_weight != 0):
            self.ssae = self.sheet.sim.Population(self.sheet.pop.size,self.sheet.sim.SpikeSourceArray())
            seeds=mozaik.get_seeds((self.sheet.pop.size,))
            self.stgene = [StGen(rng=numpy.random.RandomState(seed=seeds[i])) for i in self.to_stimulate_indexes]
            self.sheet.sim.Projection(self.ssae, self.sheet.pop,self.sheet.sim.OneToOneConnector(),synapse_type=exc_syn,receptor_type='excitatory') 

    def prepare_stimulation(self,duration,offset):

        if (self.parameters.exc_firing_rate != 0 and self.parameters.exc_weight != 0):
           for j,i in enumerate(self.to_stimulate_indexes):
                if self.ssae._mask_local[i]:
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
        self.scs.set_parameters(times=[offset+self.sheet.dt*3], amplitudes=[self.parameters.current],copy=False)
        
    def inactivate(self,offset):
        self.scs.set_parameters(times=[offset+self.sheet.dt*3], amplitudes=[0.0],copy=False)



class OpticalStimulatorArray(DirectStimulator):
    """
    This class assumes there is a regular grid of stimulators (parameters `size` and
    `spacing` control the geometry of the grid), with each stimulator stimulating
    indiscriminately the local population of neurons in the given sheet. The
    stimulations from different stimulators add up linearly.

    The temporal profile of the stimulator is given by function specified in the
    parameter `stimulating_signal`. This function receives a list of stimulator
    x coordinates and y coordinates, the update interval, and any extra
    user parameters specified in the parameter `stimulating_signal_parameters`.
    It should return a 3D numpy array of size:
    coor_x.shape[0] x coor_x.shape[1] x (stimulation_duration/update_interval)

    The function specified in `stimulating_signal` should thus look like this:

    def stimulating_signal_function(sheet, coor_x, coor_y, update_interval, parameters)

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
                     The distance between stimulators (the number of stimulators will
                     thus be (size/distance)^2)

    update_interval : float (ms)
                     The interval at which the stimulator is updated. Thus the length of
                     the stimulation is update_interval times the number of values
                     returned by the function specified in the `stimulating_signal`
                     parameter.
    
    depth_sampling_step : float (μm)
                     For optimization reasons we will assume that neurons lie at
                     discrete range of depth spaced at `depth_sampling_step`

    light_source_light_propagation_data : str
                     The path to the radial profile light dispersion data.

    stimulating_signal : str
                     The python path to a function that defines the stimulation.

    stimulating_signal_parameters : ParameterSet
                     The parameters passed to the function specified in
                     `stimulating_signal`

    transfection_proportion : float
                     Set the proportion of transfected cells in each sheet in
                     sheet_list. Must have equal length to sheet_list. The constants
                     must be in the range (0,1) - 0 means no cells, 1 means
                     all cells.

    Notes
    -----

    For now this is not mpi optimized.
    """
    
    
    required_parameters = ParameterSet({
            'size': float,
            'spacing' : float,
            'stimulating_signal' : str,
            'stimulating_signal_parameters' : ParameterSet,
            'update_interval' : float,
            'depth_sampling_step' : float,
            'light_source_light_propagation_data' : str,
            'transfection_proportion' : float,
    })
    
    def __init__(self, sheet,parameters,shared_scs=None,optimized_scs=True):
        DirectStimulator.__init__(self, sheet,parameters)

        assert math.fmod(self.parameters.size,self.parameters.spacing) < 0.000000001 , "Error the size has to be multiple of spacing!"
        assert math.fmod(self.parameters.size / self.parameters.spacing /2,2) < 0.000000001 , "Error the size and spacing have to be such that they give odd number of elements!"

        
        axis_coors = numpy.arange(0,self.parameters.size+self.parameters.spacing,self.parameters.spacing) - self.parameters.size/2.0 

        n = int(numpy.floor(len(axis_coors)/2.0))
        stimulator_coords_y, stimulator_coords_x = numpy.meshgrid(axis_coors, axis_coors)

        #let's load up disperssion data and setup interpolation
        f = open(self.parameters.light_source_light_propagation_data,'rb')
        radprofs = pickle.load(f,encoding='latin1')
        f.close()

        #light_flux_lookup =  scipy.interpolate.RegularGridInterpolator((numpy.arange(0,1080,60),numpy.linspace(0,1,354)*149.701*numpy.sqrt(2)), radprofs, method='linear',bounds_error=False,fill_value=0)
        light_flux_lookup =  scipy.interpolate.RegularGridInterpolator((np.linspace(0,1080,radprofs.shape[0]),numpy.linspace(0,1,radprofs.shape[1])*299.7*numpy.sqrt(2)), radprofs, method='linear',bounds_error=False,fill_value=0)

        # the constant translating the data in radprofs to photons/s/cm^2
        K = 2.97e26
        W = 3.9e-10

        # now let's calculate mixing weights, this will be a matrix nxm where n is 
        # the number of neurons in the population and m is the number of stimulators
        x =  stimulator_coords_x.flatten()
        y =  stimulator_coords_y.flatten()
        xx,yy = self.sheet.vf_2_cs(self.sheet.pop.positions[0],self.sheet.pop.positions[1])
        zeros = numpy.zeros(len(x))
          
        mixing_templates=[]
        for depth in numpy.arange(sheet.parameters.min_depth,sheet.parameters.max_depth+self.parameters.depth_sampling_step,self.parameters.depth_sampling_step):
            temp = numpy.reshape(light_flux_lookup(numpy.transpose([zeros+depth,numpy.sqrt(numpy.power(x,2)  + numpy.power(y,2))])),(2*n+1,2*n+1))
            a  = temp[n,n:]
            cutof = numpy.argmax((numpy.sum(a)-numpy.cumsum(a))/numpy.sum(a) < 0.01)
            assert numpy.shape(temp[n-cutof:n+cutof+1,n-cutof:n+cutof+1]) == (2*cutof+1,2*cutof+1), str(numpy.shape(temp[n-cutof:n+cutof,n-cutof:n+cutof])) + 'vs' + str((2*cutof+1,2*cutof+1))
            mixing_templates.append((temp[n-cutof:n+cutof+1,n-cutof:n+cutof+1],cutof))

        signal_function = load_component(self.parameters.stimulating_signal)
        self.stimulator_signals = signal_function(sheet,stimulator_coords_x,stimulator_coords_y,self.parameters.update_interval,self.parameters.stimulating_signal_parameters)

        self.mixed_signals_photo = numpy.zeros((self.sheet.pop.size,numpy.shape(self.stimulator_signals)[2]),dtype=numpy.float64)
        
        # find coordinates given spacing and shift by half the array size
        nearest_ix = numpy.rint(xx/self.parameters.spacing)+n
        nearest_iy = numpy.rint(yy/self.parameters.spacing)+n
        nearest_iz = numpy.rint((numpy.array(self.sheet.pop.positions[2])-sheet.parameters.min_depth)/self.parameters.depth_sampling_step)

        nearest_ix[nearest_ix<0] = 0
        nearest_iy[nearest_iy<0] = 0
        nearest_ix[nearest_ix>2*n] = 2*n
        nearest_iy[nearest_iy>2*n] = 2*n

        for i in range(0,self.sheet.pop.size):
            temp,cutof = mixing_templates[int(nearest_iz[i])]

            ss = self.stimulator_signals[max(int(nearest_ix[i]-cutof),0):int(nearest_ix[i]+cutof+1),max(int(nearest_iy[i]-cutof),0):int(nearest_iy[i]+cutof+1),:]
            if ss.size != 0:
               temp = temp[max(int(cutof-nearest_ix[i]),0):max(int(2*n+1+cutof-nearest_ix[i]),0),max(int(cutof-nearest_iy[i]),0):max(int(2*n+1+cutof-nearest_iy[i]),0)]
               self.mixed_signals_photo[i,:] = K*W*numpy.dot(temp.flatten(),numpy.reshape(ss,(len(temp.flatten()),-1)))

        self.stimulation_duration = numpy.shape(self.mixed_signals_photo)[1] * self.parameters.update_interval

        assert numpy.shape(self.mixed_signals_photo) == (self.sheet.pop.size,self.stimulator_signals.shape[2]), "ERROR: mixed_signals_photo doesn't have the desired size:" + str(self.mixed_signals_photo.shape) + " vs " +str((self.sheet.pop.size,stimulator_signals.shape[2]))

        if optimized_scs:
            self.setup_scs(shared_scs)
        else:
            self.setup_scs_old(shared_scs)

        self.stimulator_signals = self.compress_array(self.stimulator_signals)

    @staticmethod
    def compress_array(array):
        compressed = io.BytesIO()
        np.savez_compressed(compressed, array)
        return compressed

    @staticmethod
    def decompress_array(array):
        array.seek(0)
        return np.load(array)['arr_0']

    def setup_scs(self, shared_scs):
        stimulated_cell_indices = self.mixed_signals_photo.sum(axis=1)>0
        self.stimulated_cells = self.sheet.pop.all_cells[stimulated_cell_indices]
        self.mixed_signals_photo = self.mixed_signals_photo[stimulated_cell_indices]

        if self.parameters.transfection_proportion != 1:
            sel_idx = np.random.choice(range(len(self.stimulated_cells)),size=int(self.parameters.transfection_proportion*len(self.stimulated_cells)))
            self.stimulated_cells = self.stimulated_cells[sel_idx]
            self.mixed_signals_photo = self.mixed_signals_photo[sel_idx]

        if shared_scs == None:
            shared_scs = {}

        self.scs = [self.sheet.sim.StepCurrentSource(times=[0.0], amplitudes=[0.0]) if cell not in shared_scs else shared_scs[cell] for cell in self.stimulated_cells]

        for cell,scs in zip(self.stimulated_cells,self.scs):
            if cell not in shared_scs:
                cell.inject(scs)

    def setup_scs_old(self, shared_scs):
        if shared_scs != None:
            self.scs = shared_scs
        else:
            self.scs = [self.sheet.sim.StepCurrentSource(times=[0.0], amplitudes=[0.0]) for cell in self.sheet.pop.all_cells]
            for cell,scs in zip(self.sheet.pop.all_cells,self.scs):
                cell.inject(scs)


    def prepare_stimulation(self,duration,offset):
        assert self.stimulation_duration == duration, "stimulation_duration != duration :"  + str(self.stimulation_duration) + " " + str(duration)
        assert hasattr(self,"mixed_signals_current"), "Child class has to implement conversion of optical stimulation to current!"
        times = numpy.arange(0,self.stimulation_duration,self.parameters.update_interval) + offset
        times[0] = times[0] + 3*self.sheet.dt
        for i in range(0,len(self.scs)):
            self.scs[i].set_parameters(times=Sequence(times), amplitudes=Sequence(self.mixed_signals_current[i,:].flatten()),copy=False)

    def inactivate(self,offset):
        for scs in self.scs:
            scs.set_parameters(times=[offset+3*self.sheet.dt], amplitudes=[0.0],copy=False)

    def save_to_datastore(self,data_store,stimulus):
        photo_mixed_signals = self.decompress_array(self.mixed_signals_photo)
        data_store.full_datastore.add_analysis_result(
            AnalogSignalList(
                [NeoAnalogSignal(photo_mixed_signals[i, :], sampling_period=self.parameters.update_interval*qt.ms, units=qt.dimensionless) for i in range(len(self.stimulated_cells))],
                [int(ID) for ID in self.stimulated_cells],
                qt.dimensionless,
                x_axis_name="time",
                y_axis_name="optical_stimulation_photons",
                sheet_name=self.sheet.name,
                stimulus_id=str(stimulus),
            )
        )
        data_store.full_datastore.add_analysis_result(
            AnalogSignalList(
                [NeoAnalogSignal(self.mixed_signals_current[i, :], sampling_period=self.parameters.update_interval*qt.ms, units=qt.nA) for i in range(len(self.stimulated_cells))],
                [int(ID) for ID in self.stimulated_cells],
                qt.nA,
                x_axis_name="time",
                y_axis_name="optical_stimulation_current",
                sheet_name=self.sheet.name,
                stimulus_id=str(stimulus),
            )
        )
        data_store.full_datastore.add_analysis_result(
            SingleValue(
                value_name="optical_stimulation_array_compressed",
                value=self.stimulator_signals,
                value_units=qt.dimensionless,
                sheet_name=self.sheet.name,
                stimulus_id=str(stimulus),
            )
        )

    def debug_plot_stimulator_signals():
        pylab.figure(figsize=(10, 10))
        ax = pylab.subplot(111)
        ax.set_aspect("equal")
        ax.set_title("Activation magnitude (stimulators)")
        sc = ax.scatter(
            coor_x.flatten(),
            coor_y.flatten(),
            c=numpy.squeeze(numpy.mean(signals, axis=2)).flatten(),
            cmap="viridis",
        )
        pylab.colorbar(sc, ax=ax)
        pylab.savefig(
            Global.root_directory
            + "orient_stim"
            + sheet.name.replace("/", "_")
            + ".png"
        )
        pylab.clf()


@jit()
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


class OpticalStimulatorArrayChR(OpticalStimulatorArray):
    """
    Like *OpticalStimulatorArray*, but the light (photons/s/cm^2) impinging on the
    neuron is transformed via a model of Channelrhodopsin (courtesy of Quentin Sabatier)
    to give the final injected current.

    Note that we approximate the current by ignoring the voltage dependence of the
    channels, as it is very expensive to inject conductance in PyNN. The
    Channelrhodopsin has reverse potential of ~0, and we assume that our neurons
    sits on average at -60mV to calculate the current.
    """
    def __init__(self, sheet, parameters,shared_scs=None,optimized_scs=True):
        OpticalStimulatorArray.__init__(self, sheet,parameters,shared_scs,optimized_scs)
        self.times = numpy.arange(0,self.stimulation_duration,self.parameters.update_interval)
        self.mixed_signals_current = np.zeros_like(self.mixed_signals_photo)

        for i in range(0,len(self.scs)):
            res = odeint(ChRsystem,[0,0,0.2,0.8,0],self.times,args=(self.mixed_signals_photo[i,:].flatten(),self.parameters.update_interval),hmax=self.parameters.update_interval)
            # Here we assume that we don't calculate the output if the input is zero
            if optimized_scs:
                assert res[:,0:2].sum() != 0, "ODE solving failed!"
            self.mixed_signals_current[i,:] =  60 * (17.2*res[:,0] + 2.9 * res[:,1])  / 2500 ; # the 60 corresponds to the 60mV difference between ChR reverse potential of 0mV and our expected mean Vm of about 60mV. This happens to end up being in nA which is what pyNN expect for current injection.

        self.mixed_signals_photo = self.compress_array(self.mixed_signals_photo)

    def debug_plot(self):
        pylab.figure(figsize=(15,15))
        ax = pylab.subplot(121)
        pylab.gca().set_aspect('equal')
        pylab.title('Activation magnitude (neurons)')
        lum = []
        for c in self.sheet.pop.all_cells:
            idx = np.where(self.stimulated_cells == c)[0]
            lum.append(0 if len(idx) == 0 else np.max(self.mixed_signals_photo[idx[0],:]))
        sc = ax.scatter(self.sheet.pop.positions[0],self.sheet.pop.positions[1],s=10,c=lum,vmin=0)
        pylab.colorbar(sc, ax=ax)

        idx = np.argmax(self.mixed_signals_photo.sum(axis=1))
        ax = pylab.subplot(122)
        ax.set_title('Single neuron current injection profile')
        ax.plot(self.times,self.mixed_signals_photo[idx,:],'k')
        ax.set_ylabel('photons/cm2/s', color='k')

        ax2 = ax.twinx()
        ax2.plot(self.times,self.mixed_signals_current[idx,:],'g')
        ax2.set_ylabel('nA', color='g')
        pylab.savefig(Global.root_directory +'OpticalStimulatorArrayTest_' + self.sheet.name.replace('/','_') + '.png')
        pylab.clf()


def stimulating_pattern_flash(sheet, coor_x, coor_y, update_interval, parameters):
    """
    Stimulation with a static stimulation pattern, its exact form is determined
    by the supplied extra parameters. The stimulus turns on at *onset_time*, turns off
    at *offset_time*. The overall duration of the stimulus is *duration* ms.

    Parameters
    ----------
    sheet : Sheet
                The stimulated sheet

    coor_x : numpy array
                X coordinates of all electrodes

    coor_y : numpy array
                Y coordinates of all electrodes

    update_interval : float (ms)
                Timestep in which the stimulator updates

    parameters : Parameters
                Extra parameters for the stimulator functions. They must at minimum
                include the following parameters:

                duration : float (ms)
                        Overall stimulus duration

                onset_time : float (ms)
                        Time point when the stimulation turns on

                offset_time : float(ms)
                        Time point when the stimulation turns off
    """
    signals = numpy.zeros(
        (
            numpy.shape(coor_x)[0],
            numpy.shape(coor_x)[1],
            int(parameters.duration / update_interval),
        )
    )

    t_onset = int(numpy.floor(parameters.onset_time / update_interval))
    t_offset = int(numpy.floor(parameters.offset_time / update_interval))

    mask = generate_2d_stim(sheet, coor_x, coor_y, parameters)
    signals[:, :, t_onset:t_offset] = np.repeat(mask[:, :, np.newaxis], t_offset-t_onset, axis=2)

    return signals

def generate_2d_stim(sheet, coor_x, coor_y, parameters):
    """
    Generates a 2d pattern for cortical stimulation.

    Parameters
    ----------
    sheet : Sheet
                The stimulated sheet

    coor_x : numpy array
                X coordinates of all electrodes

    coor_y : numpy array
                Y coordinates of all electrodes

    parameters : Parameters
                Extra parameters for the stimulator functions. They must at minimum
                include the following parameter:

                intensity : float
                        Stimulation intensity, going from 0 to 1

    """
    if parameters.shape == "or_map":
        return or_map_mask(sheet, coor_x, coor_y, parameters)
    elif parameters.shape in ["hexagon", "circle","hexagon"]:
        return simple_shapes_binary_mask(coor_x, coor_y, parameters.shape, parameters) * parameters.intensity
    elif parameters.shape == "image":
        return image_stim(coor_x, coor_y, parameters)
    else:
        raise ValueError("Unknown shape %s for cortical stimulation!", parameters.shape)


def image_stim(coor_x, coor_y, parameters):
    """
    Generate stimulation in the pattern of a grayscale image, loaded from a .npy
    file containing a 2D numpy array, with values between 0 (black) and 1 (white).

    The mapping between the axes of the numpy array and cortical space
    is 0->X, 1->Y.

    If the image has a different aspect ratio or number of pixels as the stimulation
    array, it will be stretched to fit the array.

    Parameters
    ----------
    coor_x : numpy array
                X coordinates of all electrodes

    coor_y : numpy array
                Y coordinates of all electrodes

    parameters : ParameterSet
        intensity : float
                Stimulation intensity constant
        image_path : str
                Path to the .npy file containing the image (2D array).
    """
    for i in range(coor_x.shape[1]):
        assert np.allclose(coor_x[:, i], coor_x[:, i]), "X coordinates must be in grid!"
    for i in range(coor_y.shape[0]):
        assert np.allclose(coor_y[0, :], coor_y[i, :]), "Y coordinates must be in grid!"
    A = np.load(parameters.image_path)
    assert len(A.shape) == 2, "The image must be 2D! Instead, the image shape is: " % A.shape
    assert np.all(A >= 0) and np.all(A <= 1), "All values in the image must be in the range of (0,1)!"
    A_interp = scipy.interpolate.interp2d(
        np.linspace(coor_x[:, 0].min(), coor_x[:, 0].max(), A.shape[0]),
        np.linspace(coor_y[0, :].min(), coor_y[0, :].max(), A.shape[1]),
        A,
        fill_value=0,
    )(coor_x[:, 0], coor_y[0, :])
    return A_interp * parameters.intensity


def or_map_mask(sheet,coor_x,coor_y,parameters):
    """
    Stimulating pattern based on the cortical orientation map, where one orientation
    is selected as the primary orientation to maximally stimulate (with *intensity*
    intensity), and the stimulation intensity for the other orientations falls off as a
    Gaussian with the circular distance from the selected orientation:

    Stimulation intensity = intensity * e^(-0.5*d^2/sharpness)
    d = circular_dist(selected_orientation-or_map_orientation)

    Parameters
    ----------
    sheet : Sheet
                Sheet to retrieve neuron orientations (proxy for orientation map) from

    coor_x : numpy array
                X coordinates of all electrodes

    coor_y : numpy array
                Y coordinates of all electrodes

    parameters : ParameterSet
                    intensity : float
                            Stimulation intensity constant
                    sharpness : float
                            Variance of the Gaussian falloff
                    orientation : float
                            Selected orientation to stimulate
    """
    z = sheet.pop.all_cells.astype(int)
    vals = numpy.array([sheet.get_neuron_annotation(i,'LGNAfferentOrientation') for i in range(0,len(z))])
    px,py = sheet.vf_2_cs(sheet.pop.positions[0],sheet.pop.positions[1])
    ors = scipy.interpolate.griddata(list(zip(px,py)), vals, (coor_x, coor_y), method='nearest')

    return parameters.intensity*np.exp(-0.5*np.power(circular_dist(parameters.orientation,ors,np.pi),2)/parameters.sharpness)


def simple_shapes_binary_mask(coor_x, coor_y, shape, parameters):
    """
    Generate a stimulation pattern of one or more simple shapes of the same type, in the
    form of a binary mask. The list of coordinates, *coords* defines the number and
    centers of these shapes.

    All coordinates and lengths should be interpreted as cortical coordinates in μm!

    Currently three types of shapes are supported:

    polygon: *points* defines the coordinates of the polygon verices, compared to the
             current coordinate in *coords*. These points can be rotated by *angle*, if
             specified.
    circle:  *radius* defines the radii of circles, with *coords* centers
    hexagon: *radius* defines the radii (or edge length) of hexagons, *coords* their
             center coordinates, and *angle* their rotation (at 0 angle they are
             rotated such that the top and bottom edges are horizontal)

    Parameters
    ----------
    coor_x : numpy array
                X coordinates of all electrodes

    coor_y : numpy array
                Y coordinates of all electrodes

    shape : str
            polygon, circle or hexagon

    parameters : ParameterSet
        coords : list((x,y))
                Coordinates of the centers of the shapes to draw
        points : list((x,y))
                Polygon coordinates compared to their center (coords[i])
        radius : float
                Circle or hexagon radius
        angle : float
                Hexagon or polygon rotation
        inverted : bool
                Inverts the pattern if True. Defaults to False if not included.
    """
    known_shapes = ["polygon","circle","hexagon"]
    assert shape in known_shapes, "Shape %s not among known shapes: %s" % (shape, known_shapes)

    if "angle" not in parameters:
        parameters.angle = 0
    if "inverted" not in parameters:
        parameters.inverted = False
    if type(parameters.coords[0]) != list and type(parameters.coords[0]) != tuple:
        parameters.coords = [parameters.coords]
    if shape == "polygon":
        points = np.array(parameters.points)
        # Calculate center of mass
        parameters.coords = [(points.T[0].mean(), points.T[1].mean())]
        points = points - np.array(parameters.coords[0])
    elif shape == "hexagon":
        points = (
            np.array(
                [
                    [0, 1],
                    [np.sqrt(3) / 2, 0.5],
                    [np.sqrt(3) / 2, -0.5],
                    [0, -1],
                    [-np.sqrt(3) / 2, -0.5],
                    [-np.sqrt(3) / 2, 0.5],
                ]
            )
            * parameters.radius
        )

    mask = np.full(coor_x.shape, False)
    for x_c, y_c in parameters.coords:
        if shape == "polygon" or shape == "hexagon":
            path = matplotlib.path.Path(points)
        if shape == "circle":
            path = matplotlib.path.Path.circle(radius=parameters.radius)

        coords = np.hstack((coor_x.reshape(-1, 1), coor_y.reshape(-1, 1)))

        transform = (
            matplotlib.transforms.Affine2D()
            .translate(x_c, y_c)
            .rotate_around(x_c, y_c, parameters.angle)
        )
        mask_ = path.contains_points(coords, transform=transform, radius=-0.0001)
        mask_ = mask_.reshape(coor_x.shape[0], coor_x.shape[1])
        mask = np.logical_or(mask, mask_)

    if parameters.inverted:
        mask = np.logical_not(mask)
    return mask

def single_pixel(sheet, coor_x, coor_y, update_interval, parameters):
    """
    A simple stimulation pattern where for the entire duration a single stimulator
    pixel is active (with an intensity of 1), all others have a value of 0.

    Parameters
    ----------
    coor_x : numpy array
                X coordinates of all electrodes

    coor_y : numpy array
                Y coordinates of all electrodes

    update_interval : float (ms)
                Timestep in which the stimulator updates

    parameters : ParameterSet
        x : int
            x position of the lit up pixel.

        y : int
            y position of the lit up pixel.

        duration : float (ms)
            Overall stimulus duration

    Only used in a for testing.
    """
    x, y = parameters["x"], parameters["y"]
    assert x in coor_x and y in coor_y
    signals = np.zeros(
        (
            np.shape(coor_x)[0],
            np.shape(coor_x)[1],
            int(parameters.duration / update_interval),
        )
    )
    signals[np.where(x == coor_x)[0][0], np.where(y == coor_y)[1][0], :] = 1
    return signals
