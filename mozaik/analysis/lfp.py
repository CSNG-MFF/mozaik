# -*- coding: utf-8 -*-
"""
Module containing lfp specific analysis.
"""
import mozaik
import numpy
import quantities as qt
from .analysis import Analysis
from mozaik.tools.mozaik_parametrized import colapse, colapse_to_dictionary, MozaikParametrized
from mozaik.analysis.data_structures import AnalogSignal, PerAreaAnalogSignalList
from mozaik.analysis.helper_functions import psth
from parameters import ParameterSet
from mozaik.storage import queries
from mozaik.tools.circ_stat import circ_mean, circular_dist
from mozaik.tools.neo_object_operations import neo_mean, neo_sum
from builtins import zip
from collections import OrderedDict
from mozaik.tools.distribution_parametrization import PyNNDistribution
from neo.core.analogsignal import AnalogSignal as NeoAnalogSignal
from scipy.signal import butter, lfilter, hilbert

logger = mozaik.getMozaikLogger()

class GeneralizedPhase(Analysis):
    """
    Apply the generalized phased analysis on all the PerAreaAnalogSignalList of the dsv
    Compute the instantaneous frequencies, phase gradient properties, wavelengths and wavespeeds

    Other parameters
    ----------------
    threshold : float
             The threshold (0.9 used in the Davis paper) used to determine whether the sign of the instantaneous frequency is significant or not
    """
    required_parameters = ParameterSet({
      'threshold': float,
    })

    def perform_analysis(self):
        units = qt.Hz
        for sheet in self.datastore.sheets():

           dsv1 = queries.param_filter_query(self.datastore, sheet_name=sheet, identifier=['PerAreaAnalogSignalList'])
           for paasl in dsv1.get_analysis_result():
               asl = paasl.asl
               # Assuming spacing between two datapoint is constant
               x_ps = paasl.x_coords[1] - paasl.x_coords[0]
               y_ps = paasl.y_coords[1] - paasl.y_coords[0]

               nx = len(asl)
               ny = len(asl[0])
               sampling_period = asl[0][0].sampling_period
               t_start = asl[0][0].t_start
               dur = asl[0][0].shape[0]

               new_t_start = t_start 

               # calculate the instantaneous frequencies for each analog signal
               painstFreqs=[]
               positive = 0
               negative = 0

               shuffledSig = numpy.random.permutation(numpy.array(asl)[:,:,:,0].reshape((nx*ny,dur))).reshape(nx, ny, dur)
               instFreqsShuffled = numpy.zeros((nx,ny,dur-1))

               # Compute the analytic signal using Hilbert transform
               analyticSig = hilbert(numpy.array(asl)[:,:,:,0])
               shuffledAnalyticSig = hilbert(shuffledSig)

               for x in range(nx):
                   row = []
                   shuff_row = []
                   for y in range(ny):

                       # This formula for calculating instantaneous frequencies of discrete signals avoids to have to unwrap the phases
                       instFreqs = numpy.angle(numpy.conjugate(analyticSig[x,y,:-1])*analyticSig[x,y,1:])/(2*numpy.pi * sampling_period) 
                       instFreqsShuffled[x,y,:] = numpy.angle(numpy.conjugate(shuffledAnalyticSig[x,y,:-1])*shuffledAnalyticSig[x,y,1:])/(2*numpy.pi * sampling_period) 
                       row.append(NeoAnalogSignal(instFreqs,t_start=new_t_start, sampling_period=sampling_period, units=units))
                       positive += numpy.sum(instFreqs > 0)
                       negative += numpy.sum(instFreqs < 0)

                   painstFreqs.append(row)
               tot = (nx) * (ny) * (dur - 1)
               if positive/tot > self.parameters.threshold:
                   sign = 1
               elif negative/tot > self.parameters.threshold:
                   sign = -1
               else:
                   sign = float("NaN") 

               if numpy.sum(instFreqsShuffled > 0)/tot > self.parameters.threshold:
                   signShuffled = 1
               elif numpy.sum(instFreqsShuffled < 0)/tot > self.parameters.threshold:
                   signShuffled = -1
               else:
                   signShuffled = float("NaN")


               dx = numpy.zeros((nx, ny, dur))
               dy= numpy.zeros((nx, ny, dur))

               dx_shuff = numpy.zeros((nx, ny, dur))
               dy_shuff = numpy.zeros((nx, ny, dur))

               # Compute the spatial gradients at each position for each time point
               for t in range(dur):
                   tmp_dx = numpy.zeros((nx, ny))
                   tmp_dx[0,:] = numpy.angle(numpy.conjugate(analyticSig[0,:,t])*analyticSig[1,:,t])/x_ps
                   tmp_dx[-1,:] = numpy.angle(numpy.conjugate(analyticSig[-2,:,t])*analyticSig[-1,:,t])/x_ps
                   tmp_dx[1:-1,:] = numpy.angle(numpy.conjugate(analyticSig[:-2,:,t])*analyticSig[2:,:,t])/(2 * x_ps)

                   dx[:,:,t] = -sign * tmp_dx

                   tmp_dy = numpy.zeros((nx, ny))
                   tmp_dy[:,1] = numpy.angle(numpy.conjugate(analyticSig[:,0,t])*analyticSig[:,1,t])/y_ps
                   tmp_dy[:,-1] = numpy.angle(numpy.conjugate(analyticSig[:,-2,t])*analyticSig[:,-1,t])/y_ps
                   tmp_dy[:,1:-1] = numpy.angle(numpy.conjugate(analyticSig[:,:-2,t])*analyticSig[:,2:,t])/(2 * y_ps)

                   dy[:,:,t] = -sign * tmp_dy


                   tmp_dx_shuff = numpy.zeros((nx, ny))
                   tmp_dx_shuff[0,:] = numpy.angle(numpy.conjugate(shuffledAnalyticSig[0,:,t])*shuffledAnalyticSig[1,:,t])/x_ps
                   tmp_dx_shuff[-1,:] = numpy.angle(numpy.conjugate(shuffledAnalyticSig[-2,:,t])*shuffledAnalyticSig[-1,:,t])/x_ps
                   tmp_dx_shuff[1:-1,:] = numpy.angle(numpy.conjugate(shuffledAnalyticSig[:-2,:,t])*shuffledAnalyticSig[2:,:,t])/(2 * x_ps)

                   dx_shuff[:,:,t] = -sign * tmp_dx_shuff

                   tmp_dy_shuff = numpy.zeros((nx, ny))
                   tmp_dy_shuff[:,1] = numpy.angle(numpy.conjugate(shuffledAnalyticSig[:,0,t])*shuffledAnalyticSig[:,1,t])/y_ps
                   tmp_dy_shuff[:,-1] = numpy.angle(numpy.conjugate(shuffledAnalyticSig[:,-2,t])*shuffledAnalyticSig[:,-1,t])/y_ps
                   tmp_dy_shuff[:,1:-1] = numpy.angle(numpy.conjugate(shuffledAnalyticSig[:,:-2,t])*shuffledAnalyticSig[:,2:,t])/(2 * y_ps)

                   dy_shuff[:,:,t] = -sign * tmp_dy_shuff
                   
               adx = []
               ady = []
               pm = []
               pd  = []
               wl  = []
               swl = []
               ws  = []
               sigwl = []

               wl_shuff = 1/(numpy.sqrt(dx_shuff **2 + dy_shuff ** 2)/(2 * numpy.pi)) 
               thresh = numpy.sort(wl_shuff.reshape((nx * ny * dur)))[int(nx * ny * dur*99/100)+1]

               # Store the gradients in the correct format for PerAreaAnalogSignalList
               for x in range(nx):
                   rdx = []
                   rdy = []
                   rpm = []
                   rpd = []
                   rwl = []
                   rswl = []
                   rws = []
                   rsigwl = []
                   for y in range(ny):
                       dx_tmp = dx[x,y,:]
                       dy_tmp = dy[x,y,:]
                       pm_tmp = numpy.sqrt(dx_tmp **2 + dy_tmp ** 2)/(2 * numpy.pi) 
                       wl_tmp = 1/pm_tmp
                       rdx.append(NeoAnalogSignal(dx_tmp,t_start=new_t_start, sampling_period=sampling_period, units=qt.um * units))
                       rdy.append(NeoAnalogSignal(dy_tmp,t_start=new_t_start, sampling_period=sampling_period, units=qt.um * units))
                       # Compute the gradient magnitudes
                       rpm.append(NeoAnalogSignal(pm_tmp,t_start=new_t_start, sampling_period=sampling_period, units=qt.um * units))
                       # Compute the gradient directions 
                       rpd.append(NeoAnalogSignal(numpy.arctan2(dx_tmp, dy_tmp),t_start=new_t_start, sampling_period=sampling_period, units=qt.rad))
                       # Compute the wavelengths
                       rwl.append(NeoAnalogSignal(wl_tmp,t_start=new_t_start, sampling_period=sampling_period, units=qt.um * units))
                       rswl.append(NeoAnalogSignal(1/(numpy.sqrt(dx_shuff[x,y,:] **2 + dy_shuff[x,y,:] ** 2)/(2 * numpy.pi)),t_start=new_t_start, sampling_period=sampling_period, units=qt.um * units))
                       # Compute the wave speeds
                       rws.append(NeoAnalogSignal(painstFreqs[x][y].magnitude[:,0]/pm_tmp[:-1],t_start=new_t_start, sampling_period=sampling_period, units=qt.um * units))
                       rsigwl.append(NeoAnalogSignal([1 if w > thresh else 0 for w in wl_tmp],t_start=new_t_start, sampling_period=sampling_period, units=qt.dimensionless))
                   adx.append(rdx)
                   ady.append(rdy)
                   pm.append(rpm)
                   pd.append(rpd)
                   wl.append(rwl)
                   swl.append(rswl)
                   ws.append(rws)
                   sigwl.append(rsigwl)


               
               self.datastore.full_datastore.add_analysis_result(
                   PerAreaAnalogSignalList(painstFreqs,paasl.x_coords,paasl.y_coords,units,
                                  stimulus_id=paasl.stimulus_id,
                                  x_axis_name=paasl.x_axis_name,
                                  y_axis_name=f'Instantaneous frequencies of ({paasl.y_axis_name})',
                                  sheet_name=sheet,
                                  tags=self.tags,
                                  analysis_algorithm=self.__class__.__name__))
               self.datastore.full_datastore.add_analysis_result(
                   PerAreaAnalogSignalList(adx,paasl.x_coords,paasl.y_coords,units,
                                  stimulus_id=paasl.stimulus_id,
                                  x_axis_name=paasl.x_axis_name,
                                  y_axis_name=f'X phase gradient of ({paasl.y_axis_name})',
                                  sheet_name=sheet,
                                  tags=self.tags,
                                  analysis_algorithm=self.__class__.__name__))
               self.datastore.full_datastore.add_analysis_result(
                   PerAreaAnalogSignalList(ady,paasl.x_coords,paasl.y_coords,units,
                                  stimulus_id=paasl.stimulus_id,
                                  x_axis_name=paasl.x_axis_name,
                                  y_axis_name=f'Y phase gradient of ({paasl.y_axis_name})',
                                  sheet_name=sheet,
                                  tags=self.tags,
                                  analysis_algorithm=self.__class__.__name__))
               self.datastore.full_datastore.add_analysis_result(
                   PerAreaAnalogSignalList(pm,paasl.x_coords,paasl.y_coords,units,
                                  stimulus_id=paasl.stimulus_id,
                                  x_axis_name=paasl.x_axis_name,
                                  y_axis_name=f'Gradient magnitude of ({paasl.y_axis_name})',
                                  sheet_name=sheet,
                                  tags=self.tags,
                                  analysis_algorithm=self.__class__.__name__))
               self.datastore.full_datastore.add_analysis_result(
                   PerAreaAnalogSignalList(pd,paasl.x_coords,paasl.y_coords,units,
                                  stimulus_id=paasl.stimulus_id,
                                  x_axis_name=paasl.x_axis_name,
                                  y_axis_name=f'Gradient direction of ({paasl.y_axis_name})',
                                  sheet_name=sheet,
                                  tags=self.tags,
                                  analysis_algorithm=self.__class__.__name__))
               self.datastore.full_datastore.add_analysis_result(
                   PerAreaAnalogSignalList(wl,paasl.x_coords,paasl.y_coords,units,
                                  stimulus_id=paasl.stimulus_id,
                                  x_axis_name=paasl.x_axis_name,
                                  y_axis_name=f'Wavelength of ({paasl.y_axis_name})',
                                  sheet_name=sheet,
                                  tags=self.tags,
                                  analysis_algorithm=self.__class__.__name__))
               self.datastore.full_datastore.add_analysis_result(
                   PerAreaAnalogSignalList(swl,paasl.x_coords,paasl.y_coords,units,
                                  stimulus_id=paasl.stimulus_id,
                                  x_axis_name=paasl.x_axis_name,
                                  y_axis_name=f'Shuffled wavelength of ({paasl.y_axis_name})',
                                  sheet_name=sheet,
                                  tags=self.tags,
                                  analysis_algorithm=self.__class__.__name__))
               self.datastore.full_datastore.add_analysis_result(
                   PerAreaAnalogSignalList(ws,paasl.x_coords,paasl.y_coords,units,
                                  stimulus_id=paasl.stimulus_id,
                                  x_axis_name=paasl.x_axis_name,
                                  y_axis_name=f'Wave speed of ({paasl.y_axis_name})',
                                  sheet_name=sheet,
                                  tags=self.tags,
                                  analysis_algorithm=self.__class__.__name__))
               self.datastore.full_datastore.add_analysis_result(
                   PerAreaAnalogSignalList(sigwl,paasl.x_coords,paasl.y_coords,units,
                                  stimulus_id=paasl.stimulus_id,
                                  x_axis_name=paasl.x_axis_name,
                                  y_axis_name=f'Significant wavelength of ({paasl.y_axis_name})',
                                  sheet_name=sheet,
                                  tags=self.tags,
                                  analysis_algorithm=self.__class__.__name__))

 

class ButterworthFiltering(Analysis):
    """
    This analysis apply a butterworth filter on all the analog signals contained in all the
    AnalogSignalList, PerNeuronPairAnalogSignalList and PerAreaAnalogSignalList of the dsv

    Other parameters
    ----------------
    order : int 
             The order of the filter

    type : str
             Whether the filter is high-pass ('high'), low-pass ('low') or band-pass ('band')

    low_frequency : float (Hz)
             The low cut-off frequency, should be set for low-pass and band-pass filters

    high_frequency : float (Hz)
             The high cut-off frequency, should be set for high-pass and band-pass filters
    """
    required_parameters = ParameterSet({
      'order' : int, 
      'type': str,
      'low_frequency': float,
      'high_frequency': float,
    })
    
    def get_parameters_filter(self, sampling_period):
        # Period corresponding to the inverse of nyquist frequency
        nyq = 2 * sampling_period.rescale(qt.s).magnitude
        if self.parameters.type == 'band' or self.parameters.type == 'high':
            high = self.parameters.high_frequency * nyq

        if self.parameters.type == 'band' or self.parameters.type == 'low':
            low = self.parameters.low_frequency * nyq

        # Compute the numerator and denominator polynomials of the IIR filter 
        if self.parameters.type == 'band':
            b, a = butter(self.parameters.order, [low, high], btype='band')
        elif self.parameters.type == 'high':
            b, a = butter(self.parameters.order, high, btype='high')
        elif self.parameters.type == 'low':
            b, a = butter(self.parameters.order, low, btype='low')

        return b, a

    def perform_analysis(self):
        for sheet in self.datastore.sheets():
            # This part of the code specific to AnalogSignalList was not tested
            dsv1 = queries.param_filter_query(self.datastore, sheet_name=sheet, identifier=['AnalogSignalList','PerNeuronPairAnalogSignalList'])

            for ads in dsv1.get_analysis_result():
                asl = ads.asl
                sampling_period = asl[0].sampling_period
                t_start = asl[0].t_start
                units = asl[0].units

                # Get the parameters of the filter
                b, a = self.get_parameters_filter(sampling_period)

                # Apply the filter on each AnalogSignal
                fasl=[]
                for asignal in asl:
                    fasl.append(NeoAnalogSignal(lfilter(b, a, asignal.magnitude[:,0]),t_start=t_start, sampling_period=sampling_period, units=units))

                self.datastore.full_datastore.add_analysis_result(
                    AnalogSignalList(fasl,ads.ids,units,
                                   stimulus_id=paasl.stimulus_id,
                                   x_axis_name=paasl.x_axis_name,
                                   y_axis_name=f'Butterworth {self.parameters.type}-pass filtered of ({ads.y_axis_name}) freq=[{self.parameters.low_frequency},{self.parameters.high_frequency}], order = {self.parameters.order}',
                                   sheet_name=sheet,
                                   tags=self.tags,
                                   analysis_algorithm=self.__class__.__name__))


            # PerAreaAnalogSignalList part
            dsv1 = queries.param_filter_query(self.datastore, sheet_name=sheet, identifier=['PerAreaAnalogSignalList'])
            for paasl in dsv1.get_analysis_result():
                asl = paasl.asl
                sampling_period = asl[0][0].sampling_period
                t_start = asl[0][0].t_start
                units = asl[0][0].units
                
                # Get the parameters of the filter
                b, a = self.get_parameters_filter(sampling_period)

                # Apply the filter on each AnalogSignal
                fasl=[] 
                for x in range(len(asl)):
                    row = []
                    for y in range(len(asl)):
                        row.append(NeoAnalogSignal(lfilter(b, a, asl[x][y].magnitude[:,0]),t_start=t_start, sampling_period=sampling_period, units=units))
                    fasl.append(row)

                self.datastore.full_datastore.add_analysis_result(
                    PerAreaAnalogSignalList(fasl,paasl.x_coords,paasl.y_coords,units,
                                   stimulus_id=paasl.stimulus_id,
                                   x_axis_name=paasl.x_axis_name,
                                   y_axis_name=f'Butterworth {self.parameters.type}-pass filtered of ({paasl.y_axis_name}) freq=[{self.parameters.low_frequency},{self.parameters.high_frequency}], order = {self.parameters.order}',
                                   sheet_name=sheet,
                                   tags=self.tags,
                                   analysis_algorithm=self.__class__.__name__))



class LFPFromSynapticCurrents(Analysis):
    """
    This analysis takes each recording in DSV that has been done in response to stimulus type 'stimulus_type'
    and calculates the LFP signal using a linear combination of excitatory and inhibitory synaptic currents as a proxy.
    For each set of equal recordings (except trial) it creates one PerAreaAnalogSignalList
    `AnalysisDataStructure` instance containing the LFP signal calculated on each sub-area of the cortical space,
    defined through the x_coords and y_coords parameters.


    Other parameters
    ----------------
    downsampling : float (ms)
               the downsampling of the analog signals from which the lfp signal is constructed

    points_distance : float (micrometers)
             The distance separating each spatial points around which the LFPs will be calculated

    cropped_length : float (micrometers)
             The length of the side of the area that will be cropped for this analysis
             Allows to avoid generating LFPs for spatial positions located too close to the border of the model

    gaussian_convolution: bool
             Whether to convolve the lfp with a gaussian kernel
    
    gaussian_sigma: float
             The standard deviation of the gaussian kernel. A value must be assigned if
             gaussian_convolution is set to True
    """
    required_parameters = ParameterSet({
      'downsampling' : float, 
      'points_distance' : float,
      'cropped_length': float,
      'gaussian_convolution': bool,
      'gaussian_sigma': float,
    })

    # Need temoral bin, coordinates, and squares
    def perform_analysis(self):

        # Create a gaussian kernel at position (x,y) with standard deviation sigma
        def gaussian_kernel(x,y,sigma):
            X,Y=numpy.meshgrid(numpy.linspace(0,x_resolution-1,x_resolution),numpy.linspace(0,y_resolution-1,y_resolution))
            return numpy.exp(-((X-x)**2+(Y-y)**2)/(2.0*sigma**2))

        for sheet in self.datastore.sheets():

            dsv1 = queries.param_filter_query(self.datastore, sheet_name=sheet)
            segs = dsv1.get_segments()
            analog_ids = segs[0].get_stored_esyn_ids()
            st = [MozaikParametrized.idd(s) for s in dsv1.get_stimuli()]

            # Get the size of the sheet
            sx = eval(dsv1.full_datastore.block.annotations['sheet_parameters'])[sheet]['sx']
            sy = eval(dsv1.full_datastore.block.annotations['sheet_parameters'])[sheet]['sy']

            # Get the size of the border
            dx = sx - self.parameters.cropped_length
            dy = sy - self.parameters.cropped_length

            dix = int(dx/2/self.parameters.points_distance)
            diy = int(dy/2/self.parameters.points_distance)
            
            interpoint_resolution = 1
            
            # Create the coordinates of the points around which the lfp will be computed
            x_axis = numpy.arange(-sx/2 + self.parameters.points_distance/2, sx/2 + self.parameters.points_distance/2, self.parameters.points_distance/interpoint_resolution)
            y_axis = numpy.arange(-sy/2 + self.parameters.points_distance/2, sy/2 + self.parameters.points_distance/2, self.parameters.points_distance/interpoint_resolution)

            # resolution on both spatial dimensions
            x_resolution = x_axis.shape[0] 
            y_resolution = y_axis.shape[0] 

            for seg, s in zip(segs, st):
                
                ase = seg.get_esyn(analog_ids[0])
                asv = seg.get_vm(analog_ids[0])
                units = (ase[0] * asv[0]).units
                t_start = ase.t_start + 6*qt.ms
                
                # temporal resolution
                t_resolution = int((s.duration-6)/self.parameters.downsampling)

                # Initialize the LFP tensor
                m = numpy.zeros((x_resolution,y_resolution,t_resolution))

                # Calculate the lfps generated by the analog signals of each neuron
                for aid in analog_ids:
                    # Get the position of the neurons
                    sid = self.datastore.get_sheet_indexes(sheet,aid)
                    x = dsv1.get_neuron_positions()[sheet][0][sid] * 1000
                    y = dsv1.get_neuron_positions()[sheet][1][sid] * 1000

                    # Get the LFP position corresponding to the position of the neuron
                    x_id = int((x + sx/2 + self.parameters.points_distance/2)/self.parameters.points_distance)
                    y_id = int((y + sy/2 + self.parameters.points_distance/2)/self.parameters.points_distance)

                    # Get the analog signals of the neuron
                    e_syn = seg.get_esyn(aid).downsample(self.parameters.downsampling)
                    time_step = e_syn.sampling_period
                    e_syn = numpy.transpose(e_syn.magnitude,(1,0))[0]
                    i_syn = numpy.transpose(seg.get_isyn(aid).downsample(self.parameters.downsampling).magnitude,(1,0))[0]
                    vm = -numpy.transpose(seg.get_vm(aid).downsample(self.parameters.downsampling).magnitude,(1,0))[0]

                    # This LFP proxy is optimal when excitation is lagged by 6ms compared to inhibition
                    idiff = int(6/time_step)
                    padding = numpy.zeros(idiff)

                    lfp = (vm * e_syn)[:-idiff] + 1.65 * (vm * i_syn)[idiff:] #Same proxy as Davis et al., 2021 

                    full_signal = lfp
                    #full_signal = numpy.concatenate((padding,lfp))
                    full_signal = numpy.expand_dims(full_signal, axis=(1,2)) 
                    full_signal = numpy.concatenate([full_signal for _ in range(interpoint_resolution)], axis=1)
                    full_signal = numpy.concatenate([full_signal for _ in range(interpoint_resolution)], axis=2)
                    full_signal = numpy.transpose(full_signal,(1,2,0))

                    # Add the lfp generated by the analog signals of the neuron to the lfp tensor
                    m[x_id * interpoint_resolution:(x_id+1) * interpoint_resolution,y_id *interpoint_resolution:(y_id+1) * interpoint_resolution,:] += full_signal
                
                # Convolve the lfps with a gaussian kernel
                if self.parameters.gaussian_convolution:
                    gauss_suffix = ""
                    m_convolved = numpy.zeros((x_resolution,y_resolution,t_resolution))
                    for x in range(m.shape[0]):
                        for y in range(m.shape[1]):
                            gauss = gaussian_kernel(x,y,self.parameters.gaussian_sigma)
                            gauss = numpy.expand_dims(gauss, axis=2)
                            m_convolved[x,y,:] = numpy.sum(m * gauss, axis = (0,1))
                    m = m_convolved
                else:
                    gauss_suffix = " without convolution"
                
                # Not possible to use a real Z-score because the PixelMovie visualization takes only positive values
                avg = numpy.mean(m)
                std = numpy.std(m)
                m = m/std

                m = m[dix:-dix,diy:-diy]
                x_axis = x_axis[dix:-dix]
                y_axis = y_axis[diy:-diy]

                # Convert the tensor to a PerAreaAnalogSignalList and add it to the datastore
                lfps = []
                for x in range(m.shape[0]):
                    row = []
                    for y in range(m.shape[1]):
                        row.append(NeoAnalogSignal(m[x,y],t_start=t_start, sampling_period=time_step, units=qt.dimensionless))
                    lfps.append(row)
                self.datastore.full_datastore.add_analysis_result(
                    PerAreaAnalogSignalList(lfps,x_axis,y_axis,lfps[0][0].units,
                                   stimulus_id=str(s),
                                   x_axis_name='time',
                                   y_axis_name='LFP'+gauss_suffix,
                                   sheet_name=sheet,
                                   tags=self.tags,
                                   analysis_algorithm=self.__class__.__name__))

