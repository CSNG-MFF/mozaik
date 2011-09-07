# encoding: utf-8
"""
Retina/LGN model based on that developed by Jens Kremkow (CNRS-INCM/ALUF)
"""

import numpy
from MozaikLite.framework.containers import Layer
from MozaikLite.framework.space import VisualSpace, VisualRegion
from MozaikLite.framework.sheets import Sheet
import cai97
import logging
from NeuroTools import visual_logging
from NeuroTools.plotting import progress_bar
from NeuroTools.signals.spikes import SpikeList
from NeuroTools.parameters import ParameterSet
from pyNN import random
from pyNN.space import Grid2D

logger = logging.getLogger("Mozaik")

def meshgrid3D(x, y, z):
    """A slimmed-down version of http://www.scipy.org/scipy/numpy/attachment/ticket/966/meshgrid.py"""
    x = numpy.asarray(x)
    y = numpy.asarray(y)
    z = numpy.asarray(z)
    mult_fact = numpy.ones((len(x), len(y), len(z)))
    nax = numpy.newaxis
    return x[:,nax,nax]*mult_fact, \
           y[nax,:,nax]*mult_fact, \
           z[nax,nax,:]*mult_fact


class SpatioTemporalReceptiveField(object):
    
    def __init__(self, func, func_params, width, height, duration):
        """
        `func` should be a function of x, y, t, and a ParameterSet object
        `func_params` should be the ParameterSet that is passed to `func`.
        `width` (x-dimension) and `height` (y-dimension) are in degrees and
        `duration` is in ms.
        We take x=0 and y=0 at the centre of the spatial kernel
        """
        self.func = func
        self.func_params = func_params
        self.width = float(width)
        self.height = float(height)
        self.duration = float(duration)
        self.kernel = None
        self.spatial_resolution = numpy.inf
        self.temporal_resolution = numpy.inf
    
    def quantize(self, dx, dy, dt):
        """
        Return a quantized version of the receptive field. If `dx` does not
        divide exactly into the width, then the actual width will be slightly
        larger than the nominal width.
        `dx` and `dy` should be in degrees and `dt` in ms.
        """
        assert dx == dy # For now, at least
        nx = numpy.ceil(self.width/dx)
        ny = numpy.ceil(self.height/dy)
        nt = numpy.ceil(self.duration/dt)
        width = nx*dx
        height = ny*dy
        duration = nt*dt
        # x and y are the coordinates of the centre of each pixel
        # I use linspace instead of arange as arange sometimes can return inconsistent number of elements 
        # (must have something to do with rounding errors)
        #x = numpy.arange(0.0, width, dx)  + dx/2.0 - width/2.0
        #y = numpy.arange(0.0, height, dy) + dy/2.0 - height/2.0
        
        x = numpy.linspace(0.0, width-dx, width/dx)  + dx/2.0- width/2.0
        y = numpy.linspace(0.0, height-dy, height/dy) + dx/2.0 - height/2.0
        
        # t is the time at the beginning of each timestep
        t = numpy.arange(0.0, duration, dt)
        X,Y,T = meshgrid3D(y,x,t) # x,y are reversed because (x,y) <--> (j,i)
        kernel = self.func(X, Y, T, self.func_params)
        logger.debug("Created receptive field kernel: width=%gº, height=%gº, duration=%g ms, shape=%s" % 
                         (width, height, duration, kernel.shape))
        logger.debug("  before normalization: min=%g, max=%g" % 
                         (kernel.min(), kernel.max()))
        kernel = kernel/(nx*ny*nt) # normalize to make the kernel sum quasi-independent of the quantization
        import pylab
        pylab.figure()
        pylab.imshow(kernel[:,:,5],interpolation='nearest')
        pylab.colorbar()
        
        logger.debug("  after normalization: min=%g, max=%g, sum=%g" % 
                         (kernel.min(), kernel.max(), kernel.sum()))
        self.kernel = kernel
        self.spatial_resolution = dx
        self.temporal_resolution = dt

    @property
    def kernel_duration(self):
        return self.kernel.shape[2]

    def __str__(self):
        s = "Receptive field: width=%gº, height=%gº, duration=%g ms" % (self.width, self.height, self.duration)
        if self.kernel is not None:
            k = self.kernel
            h,w = k.shape[0:2]
            s += ", quantization=%s, actual width=%gº, actual_height=%gº, min=%g, max=%g." % \
                 (k.shape, w*self.spatial_resolution, h*self.spatial_resolution, k.min(), k.max())
        else:
            s += ". Not quantized."
        return s


class CellWithReceptiveField(object):
    """
    A model of the input current to an LGN relay cell, that multiplies, in space
    and time, the luminance values impinging on its receptive field by a
    spatiotemporal kernel. Spatial summation over the result of this multiplication
    at each time point gives a current in nA, that may then be injected into a
    relay cell.
    
    initialize() should be called once, before stimulus presentation
    view() should be called in a loop, once for each stimulus frame
    response_current() should be called at the end of stimulus presentation
    """
    
    def __init__(self, x, y, receptive_field, gain):
        self.x = x # position in space
        self.y = y #
        assert isinstance(receptive_field, SpatioTemporalReceptiveField)
        self.receptive_field = receptive_field
        self.gain = gain # (nA.m²/cd) could imagine making this a function of the background luminance
        self.i = 0
        #logger.debug("Created cell with receptive field centred at %gº,%gº" % (x,y))
        #logger.debug("  " + str(receptive_field))

    def initialize(self, background_luminance, stimulus_duration):
        """
        Create the array that will contain the current response, and set the
        initial values on the assumption that the system was looking at a blank
        screen of constant luminance prior to stimulus onset.
        """
        # we add some extra padding to avoid having to check for index out-of-bounds in view()
        self.response_length = numpy.ceil(stimulus_duration/self.receptive_field.temporal_resolution) + self.receptive_field.kernel_duration
        # we should initialize based on multiplying the kernel by the background activity
        # R0 = K_0.I_0 + Sum[j=1,L-1] K_j.B
        # R1 = K_0.I_1 + K_1.I_0 + Sum[j=2,L-1] K_j.B
        # the image-dependent components will be added in view(), so we need to
        # initialize with the Sum[] k_j.B components
        self.response = numpy.zeros((self.response_length,))
        L = self.receptive_field.kernel_duration
        assert L <= self.response_length
        for i in range(L):
            self.response[i] += background_luminance*self.receptive_field.kernel[:,:,i+1:L].sum()
        self.i = 0
    
    def view(self, visual_space):
        """
        Look at the visual space and update t
        Where the kernel temporal resolution is the same as the frame duration (visual space update interval):
           R_i = Sum[j=0,L-1] K_j.I_i-j
             where L is the kernel length/duration
         Where the kernel temporal resolution = (frame duration)/α (α an integer)
           R_k = Sum[j=0,L-1] K_j.I_i'
             where i' = (k-j)//α  (// indicates integer division, discarding the remainder)
         to avoid loading the entire image sequence into memory, we build up
         the response array one frame at a time.
        """
        visual_region = VisualRegion((self.x, self.y), (self.receptive_field.width, self.receptive_field.height))
        view_array = visual_space.view(visual_region, pixel_size=self.receptive_field.spatial_resolution)
        #logger.debug("view_array.shape = %s" % str(view_array.shape))
        #logger.debug("receptive_field.kernel.shape = %s" % str(self.receptive_field.kernel.shape))
        #logger.debug("response.shape = %s" % str(self.response.shape))
        if visual_space.update_interval%self.receptive_field.temporal_resolution != 0:
            errmsg = "The receptive field temporal resolution (%g ms) must be an integer multiple of the visual space update interval (%g ms)" % \
                (self.receptive_field.temporal_resolution, visual_space.update_interval)
            raise Exception(errmsg)
        update_factor = int(visual_space.update_interval/self.receptive_field.temporal_resolution)
        product = self.receptive_field.kernel*view_array[:,:,numpy.newaxis]
        time_course = product.sum(axis=0).sum(axis=0) # sum over the space axes, leaving a time signal.
        for j in range(self.i, self.i+update_factor):
            #if self.response[j:j+self.receptive_field.kernel_duration].shape != time_course.shape:
            #    logger.error("Shape mismatch: %s != %s (j=%d, len(response)=%d, update_factor=%d, time_course=%s)" % \
            #                  (self.response[j:j+self.receptive_field.kernel_duration].shape,
            #                   time_course.shape, j, len(self.response),
            #                   update_factor, time_course))

            # make sure we do not go beyond response array - this could happen if 
            # visual_space.update_interval/self.receptive_field.temporal_resolution is not integer
            self.response[j:j+self.receptive_field.kernel_duration] += time_course[:len(self.response[j:j+self.receptive_field.kernel_duration])]
        self.i += update_factor # we assume there is only ever 1 visual space used between initializations

    def response_current(self):
        """
        Multiply the response (units of luminance (cd/m²) if we assume the
        kernel values are dimensionless) by the 'gain', to produce a current in
        nA.
        ('gain' is not a good name, but I can't think of a better one).
        Returns a dictionary containing 'times' and 'amplitudes'.
        Might be better to use a NeuroTools AnalogSignal.
        """
        
        response = self.gain * self.response[:-self.receptive_field.kernel_duration] # remove the extra padding at the end
        time_points = self.receptive_field.temporal_resolution * numpy.arange(0, len(response))
        return {'times': time_points, 'amplitudes': response}


class SpatioTemporalFilterRetinaLGN(Sheet):
    """Retina/LGN model with spatiotemporal receptive field."""
    
    required_parameters = ParameterSet({
        'density': float, # neurons per degree squared
        'size' : tuple, # degrees of visual field
        'receptive_field': ParameterSet({
            'func': str,
            'func_params':ParameterSet,
            'width': float,
            'height': float,
            'spatial_resolution':float,
            'temporal_resolution':float,
            'duration': float,
            }),
        'cell': ParameterSet({
            'model': str,
            'params': ParameterSet,
            'initial_values': ParameterSet,
        }),
        'gain': float,
        'noise': ParameterSet({
            'mean': float,
            'stdev': float, # nA
        }),
    })
    
    def __init__(self, network, parameters):
        Sheet.__init__(self, network, parameters)       
        self.shape = (int(self.parameters.size[0] * numpy.sqrt(self.parameters.density)),int(self.parameters.size[1] * numpy.sqrt(self.parameters.density)))
        self.layers = {"A":  Layer([], label="A"),}
        self._built = False
        self._to_record = [] # to allow record() to be called before present_stimulus()
        self.rf_types = ('X_ON', 'X_OFF')

        sim = self.network.sim                

        # creat the grid of cells so that they are evenly spaced in the a region of size visual_field.size 
        # centered on 0,0
        grid = Grid2D(aspect_ratio=self.shape[0]/self.shape[1], 
                      dx=1/numpy.sqrt(self.parameters.density), 
                      dy=1/numpy.sqrt(self.parameters.density), 
                      x0=-self.parameters.size[0]/2,
                      y0=-self.parameters.size[0]/2)
        
        self.pops={}
        self.scs={}
        self.ncs={}
        for rf_type in self.rf_types:
            p = sim.Population(self.shape[0]*self.shape[1],
                               getattr(sim, self.parameters.cell.model),
                               self.parameters.cell.params,grid,
                               label=rf_type)
            for var, val in self.parameters.cell.initial_values.items():
                p.initialize(var, val)
            self.pops[rf_type] = p    
            self.layers["A"].add_population(p)                           

        for rf_type in self.rf_types:            
                self.scs[rf_type] = []
                self.ncs[rf_type] = []
                
                for lgn_cell in self.pops[rf_type]:
                    scs = sim.StepCurrentSource([0],[0])
                    
                    ncs = sim.NoisyCurrentSource(mean=self.parameters.noise.mean,
                                                            stdev=self.parameters.noise.stdev,
                                                            rng=random.NativeRNG())
                    self.scs[rf_type].append(scs)
                    self.ncs[rf_type].append(ncs)
                    lgn_cell.inject(scs)
                    lgn_cell.inject(ncs)
    
    def present_stimulus(self, visual_space, duration=None):
        """
        Present a visual stimulus to the model, and create the LGN output (relay)
        neurons. If the stimulus and model parameters have been used previously,
        the spike times will be loaded from the cache, and the output neurons
        will be `SpikeSourceArray`s, otherwise, they will be the cell type
        specified in `self.parameters.lgn_cell.model`.
        """
        logger.info("Presenting visual stimulus from visual space %s" % visual_space)
        visual_space.set_duration(duration) # needed for proper caching
        self.input = visual_space # needed for caching
        sim = self.network.sim
        
        logger.debug("Generating output spikes...")
        input_currents = self._calculate_input_currents(visual_space, duration)
        
        for rf_type in self.rf_types:            
                assert isinstance(input_currents[rf_type], list)        
                
                for lgn_cell, input_current,scs,ncs in zip(self.pops[rf_type], input_currents[rf_type],self.scs[rf_type],self.ncs[rf_type]):
                    assert isinstance(input_current, dict)
                    scs._set(input_current['times'],input_current['amplitudes'])
                    
                    
        # for debugging/testing
        input_current_array = numpy.zeros((self.shape[1], self.shape[0], len(visual_space.time_points(duration))))
        update_factor = int(visual_space.update_interval/self.parameters.receptive_field.temporal_resolution)
        logger.debug("input_current_array.shape = %s, update_factor = %d, p.dim = %s" % (input_current_array.shape, update_factor, self.shape))
        k = 0
        for i in range(self.shape[1]): # self.sahpe gives (x,y), so self.shape[1] is the height
            for j in range(self.shape[0]):
                # where the kernel temporal resolution is finer than the frame update interval,
                # we only keep the current values at the start of each frame
                input_current_array[i,j, :] = input_currents['X_ON'][k]['amplitudes'][::update_factor]
                k += 1 

        # if record() has already been called, setup the recording now
        self._built = True
        for variable, cells in self._to_record: # this allows record() to be called before present_stimulus()
            self.record(variable, cells)
             
        return input_current_array # for debugging/testing


    def write_neo_object(self,segment,tstop):
        try:
            spikes = get_spikes_to_dic(self.layers["A"].populations['X_ON'].getSpikes(),self.layers["A"].populations['X_ON'])
            for k in spikes.keys():
                # it assumes segment implements and add function which takes the id of a neuorn and the corresponding its SpikeTrain
                st = SpikeTrain(spikes[k],0,tstop,quantities.ms)
                st.index = k
                segment._spiketrains.append(st)
                segment.__getattr__('Retina_ON'+'_spikes').append(len(segment._spiketrains)-1)
                
            spikes = get_spikes_to_dic(self.layers["A"].populations['X_OFF'].getSpikes(),self.layers["A"].populations['X_OFF'])
            for k in spikes.keys():
                # it assumes segment implements and add function which takes the id of a neuorn and the corresponding its SpikeTrain
                st = SpikeTrain(spikes[k],0,tstop,quantities.ms)
                st.index = k
                segment._spiketrains.append(st)
                segment.__getattr__('Retina_OFF'+'_spikes').append(len(segment._spiketrains)-1)
            logging.debug("Writing spikes from Retina to neo object.")
        except NothingToWriteError, errmsg:
            logger.debug(errmsg)
        try:
            v = get_vm_to_dic(self.layers["A"].populations['X_ON'].get_v(),self.layers["A"].populations['X_ON'])
            for k in v.keys():
                ## it assumes segment implements and add function which takes the id of a neuorn and the corresponding its SpikeTrain
                st = AnalogSignal(v[k],sampling_period=self.network.sim.get_time_step()*quantities.ms)
                st.index = k
                segment._analogsignals.append(st)
                segment.__getattr__('Retina_ON'+'_vm').append(len(segment._analogsignals)-1)
            
            v = get_vm_to_dic(self.layers["A"].populations['X_OFF'].get_v(),self.layers["A"].populations['X_OFF'])
            for k in v.keys():
                ## it assumes segment implements and add function which takes the id of a neuorn and the corresponding its SpikeTrain
                st = AnalogSignal(v[k],sampling_period=self.network.sim.get_time_step()*quantities.ms)
                st.index = k
                segment._analogsignals.append(st)
                segment.__getattr__('Retina_OFF'+'_vm').append(len(segment._analogsignals)-1)                
                
            logging.debug("Writing Vm from Retina to neo object.")
        except NothingToWriteError, errmsg:
            logger.debug(errmsg)
        
        return segment

                
    def _calculate_input_currents(self, visual_space, duration):
        """
        Calculate the input currents for all cells.
        """
        assert isinstance(visual_space, VisualSpace)
        if duration is None:
            duration = visual_space.get_maximum_duration()
        
        P_rf = self.parameters.receptive_field
        rf_function = eval(P_rf.func)
        
        rf_ON = SpatioTemporalReceptiveField(rf_function,
                                             P_rf.func_params,
                                             P_rf.width, P_rf.height, P_rf.duration)
        rf_OFF = SpatioTemporalReceptiveField(lambda x,y,t,p: -1.0*rf_function(x,y,t,p),
                                              P_rf.func_params,
                                              P_rf.width, P_rf.height, P_rf.duration)
        dx = dy = P_rf.spatial_resolution
        dt = P_rf.temporal_resolution
        for rf in rf_ON, rf_OFF:
            rf.quantize(dx, dy, dt)
        rf = {'X_ON': rf_ON, 'X_OFF': rf_OFF}
        
        # create population of CellWithReceptiveFields, setting the receptive
        # field centres based on the size/location of self
        logger.debug("Creating population of `CellWithReceptiveField`s")
        input_cells = {}
        effective_visual_field_width, effective_visual_field_height = self.parameters.size
        x_values = numpy.linspace(-effective_visual_field_width/2.0, effective_visual_field_width/2.0, self.shape[0])
        y_values = numpy.linspace(-effective_visual_field_height/2.0, effective_visual_field_height/2.0, self.shape[1])
        for rf_type in self.rf_types:
            input_cells[rf_type] = []
            for y in y_values: # we iterate over y first to be consistent with the way
                for x in x_values: # PyNN iterates over Populations
                    cell = CellWithReceptiveField(x, y, rf[rf_type], self.parameters.gain)
                    cell.initialize(visual_space.background_luminance, duration)
                    input_cells[rf_type].append(cell)
        
        # if debugging, plot kernel
        #k = cell.receptive_field.kernel
        #vmin = k.min(); vmax = k.max()
        #for slice in k.transpose():
        #    visual_logging.debug(slice, vmin=vmin, vmax=vmax)
        
        # now process the frames
        logger.debug("Processing frames")
        
        t = 0
        while t < duration:
            for rf_type in self.rf_types:
                for cell in input_cells[rf_type]:
                    cell.view(visual_space)
            t = visual_space.update()
            progress_bar(t/duration)
                    
        input_currents = {}
        for rf_type in self.rf_types:
            input_currents[rf_type] = [cell.response_current() for cell in input_cells[rf_type]]
            cell0_currents = input_currents[rf_type][0]
            logger.debug("Input current values for %s cell #0: %s" % (rf_type, cell0_currents['amplitudes']))
            #visual_logging.debug(cell0_currents['amplitudes'], cell0_currents['times'],
            #                     "Time (ms)", "Current (nA)", "Input current values for %s cell #0" % rf_type)

            #for i in xrange(0,len(input_cells[rf_type][0].response_current()['amplitudes'])):
            for i in xrange(0,1):
                a = [cell.response_current()['amplitudes'][i] for cell in input_cells[rf_type]]
                #if rf_type == self.rf_types[0]:
                import pylab
                pylab.figure()
                pylab.title('AAA')
                pylab.imshow(numpy.reshape(a,(len(x_values),len(y_values))),interpolation='nearest')
                pylab.colorbar()
            
            
        return input_currents
    
    
    
    def record(self, variable, cells='all'):  
        for name,layer in self.layers.items():
            if cells == 'all' or isinstance(cells, int):
                layer.record(variable, cells)
            elif isinstance(cells, dict):
                layer.record(variable, cells[name])
            else:
                raise Exception("cells must be 'all', a dict, or an int. Actual value of %s" % str(cells))
        
    
