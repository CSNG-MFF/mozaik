# -*- coding: utf-8 -*-
"""
Module containing lfp specific analysis.
"""
import mozaik
import numpy
import scipy
import quantities as qt
from .analysis import Analysis
from mozaik.tools.mozaik_parametrized import colapse, colapse_to_dictionary, MozaikParametrized
from mozaik.analysis.data_structures import AnalogSignal, AnalogSignalList, PerNeuronPairAnalogSignalList, PerAreaAnalogSignalList, SingleObject, PerNeuronValue, PerAreaValue
from mozaik.analysis.helper_functions import psth
from parameters import ParameterSet
from mozaik.storage import queries
from mozaik.tools.circ_stat import circ_mean, circ_len, circular_dist
from mozaik.tools.stats import explained_variance
from mozaik.tools.neo_object_operations import neo_mean, neo_sum
from builtins import zip
from collections import OrderedDict
from mozaik.tools.distribution_parametrization import PyNNDistribution
from neo.core.analogsignal import AnalogSignal as NeoAnalogSignal
from scipy.signal import butter, lfilter, filtfilt, hilbert
from scipy.interpolate import PchipInterpolator
from skimage import measure
from PIL import Image
from scipy.interpolate import NearestNDInterpolator
import pickle

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

    def stitch_phases(self,phases, instantaneousFrequencies):
        stitch_coeff = 3
        neg_freq_fltr = instantaneousFrequencies < 0
        neg_freq_fltr[0] = 0
        neg_freq_idx = numpy.nonzero(neg_freq_fltr)[0]

        intp_idx = numpy.array([])
        not_in_first_cycle = None
        neg_freq_idx_array = numpy.array(neg_freq_idx)

        while neg_freq_idx_array.shape[0]:
            not_in_first_cycle = (neg_freq_idx_array - numpy.arange(neg_freq_idx_array[0],neg_freq_idx_array[0]+neg_freq_idx_array.shape[0])) > 0
            nzr = numpy.nonzero(not_in_first_cycle)[0]
            if len(nzr):
                intp_idx = numpy.concatenate((intp_idx,numpy.arange(neg_freq_idx_array[0],neg_freq_idx_array[0]+stitch_coeff*(nzr[0]-1) + 1)))
            else:
                intp_idx = numpy.concatenate((intp_idx,numpy.arange(neg_freq_idx_array[0],neg_freq_idx_array[0]+stitch_coeff*neg_freq_idx_array.shape[0])))
            neg_freq_idx_array = neg_freq_idx_array[not_in_first_cycle]


        intp_idx = numpy.unique(intp_idx.astype(int))
        intp_idx = intp_idx[numpy.nonzero(intp_idx < phases.shape[0])[0]]
        mask = numpy.ones(phases.shape[0], dtype=bool)
        mask[intp_idx] = False
        nintp_idx = numpy.arange(phases.shape[0])[mask]

        nintp_phase_unwrapped = numpy.unwrap(phases[nintp_idx])
        intp_phase_unwrapped = PchipInterpolator(nintp_idx,nintp_phase_unwrapped)(intp_idx)
        phase_unwrapped = numpy.zeros(phases.shape)
        phase_unwrapped[nintp_idx] = nintp_phase_unwrapped
        phase_unwrapped[intp_idx] = intp_phase_unwrapped
        new_phases = (phase_unwrapped + numpy.pi) % (2 * numpy.pi) - numpy.pi

        return new_phases

    def perform_analysis(self):
        units = qt.Hz
        units_phase = qt.rad
        for sheet in self.datastore.sheets():

           dsv1 = queries.param_filter_query(self.datastore, sheet_name=sheet, identifier=['PerAreaAnalogSignalList'])
           for paasl in dsv1.get_analysis_result():
               asl = paasl.asl
               # micrometers, assuming spacing between two datapoint is constant
               x_ps = (paasl.x_coords[1] - paasl.x_coords[0])
               y_ps = (paasl.y_coords[1] - paasl.y_coords[0])

               sampling_period = asl[0][0].sampling_period.rescale(qt.s)
               t_start = asl[0][0].t_start
               new_t_start = t_start 

               asl_arr = numpy.array(asl)[:,:,:,0]
               ny = asl_arr.shape[0]
               nx = asl_arr.shape[1]
               dur = asl_arr.shape[2]

               # calculate the instantaneous frequencies for each analog signal
               painstFreqs=[]
               phases=[]
               new_sig = [] 

               positive = 0
               negative = 0

               shuffledSig = numpy.random.permutation(asl_arr.reshape(nx*ny,dur)).reshape(ny, nx, dur)

               # Compute the analytic signal using Hilbert transform
               analyticSig = hilbert(asl_arr)
               shuffledAnalyticSig = hilbert(shuffledSig.reshape(nx*ny,dur)).reshape(ny, nx, dur)

               ## This formula for calculating instantaneous frequencies of discrete signals avoids to have to unwrap the phases
               instFreqs = numpy.angle(numpy.conjugate(analyticSig[:,:,:-1])*analyticSig[:,:,1:])/(2*numpy.pi * sampling_period) 
               instFreqsShuffled = numpy.angle(numpy.conjugate(shuffledAnalyticSig[:,:,:-1])*shuffledAnalyticSig[:,:,1:])/(2*numpy.pi * sampling_period) 
               
               # Compute sign
               sign = numpy.sign(numpy.mean(instFreqs).magnitude)
               signShuffled = numpy.sign(numpy.mean(instFreqsShuffled).magnitude)

               # Rotate according to the sign
               analyticSig = numpy.abs(analyticSig) * numpy.exp(sign * 1j * numpy.angle(analyticSig))
               shuffledAnalyticSig = numpy.abs(shuffledAnalyticSig) * numpy.exp(signShuffled * 1j * numpy.angle(shuffledAnalyticSig))

               modulus = numpy.abs(analyticSig)
               newAnalyticSig = numpy.zeros(analyticSig.shape,dtype = 'complex_')
               modulusShuffled = numpy.abs(shuffledAnalyticSig)
               newShuffledAnalyticSig = numpy.zeros(shuffledAnalyticSig.shape,dtype = 'complex_')

               # Recompute the instantenous frequencies
               for y in range(ny):
                   row_freq = []
                   row_phase = []
                   row_new_sig = []
                   for x in range(nx):
                       instFreqs = numpy.angle(numpy.conjugate(analyticSig[y,x,:-1])*analyticSig[y,x,1:])/(2*numpy.pi * sampling_period)
                       row_freq.append(NeoAnalogSignal(instFreqs,t_start=new_t_start, sampling_period=sampling_period, units=units))
                       new_phase = self.stitch_phases(numpy.angle(analyticSig[y,x,:]), instFreqs) 
                       row_phase.append(NeoAnalogSignal(new_phase,t_start=new_t_start, sampling_period=sampling_period, units=units_phase))
                       row_new_sig.append(NeoAnalogSignal(modulus[y,x] * numpy.exp(1j * new_phase),t_start=new_t_start, sampling_period=sampling_period, units=qt.dimensionless))
                       newAnalyticSig[y,x] = modulus[y,x] * numpy.exp(1j * new_phase)

                       instFreqsShuffled[y,x] = numpy.angle(numpy.conjugate(shuffledAnalyticSig[y,x,:-1])*shuffledAnalyticSig[y,x,1:])/(2*numpy.pi * sampling_period)
                       new_phase_shuffled = self.stitch_phases(numpy.angle(shuffledAnalyticSig[y,x,:]), instFreqsShuffled[y,x])
                       newShuffledAnalyticSig[y,x] = modulusShuffled[y,x] * numpy.exp(1j * new_phase_shuffled)

                   painstFreqs.append(row_freq)
                   phases.append(row_phase)
                   new_sig.append(row_new_sig)
               
               dx = numpy.zeros((ny, nx, dur))
               dy = numpy.zeros((ny, nx, dur))

               dx_shuff = numpy.zeros((ny, nx, dur))
               dy_shuff = numpy.zeros((ny, nx, dur))

               # Compute the spatial gradients at each position for each time point
               for t in range(dur):
                   tmp_dy = numpy.zeros((ny, nx))
                   tmp_dy[0,:] = numpy.angle(numpy.conjugate(newAnalyticSig[0,:,t])*newAnalyticSig[1,:,t])/y_ps
                   tmp_dy[-1,:] = numpy.angle(numpy.conjugate(newAnalyticSig[-2,:,t])*newAnalyticSig[-1,:,t])/y_ps
                   tmp_dy[1:-1,:] = numpy.angle(numpy.conjugate(newAnalyticSig[:-2,:,t])*newAnalyticSig[2:,:,t])/(2 * y_ps)

                   dy[:,:,t] = -sign * tmp_dy

                   tmp_dx = numpy.zeros((ny, nx))
                   tmp_dx[:,0] = numpy.angle(numpy.conjugate(newAnalyticSig[:,0,t])*newAnalyticSig[:,1,t])/x_ps
                   tmp_dx[:,-1] = numpy.angle(numpy.conjugate(newAnalyticSig[:,-2,t])*newAnalyticSig[:,-1,t])/x_ps
                   tmp_dx[:,1:-1] = numpy.angle(numpy.conjugate(newAnalyticSig[:,:-2,t])*newAnalyticSig[:,2:,t])/(2 * x_ps)

                   dx[:,:,t] = -sign * tmp_dx


                   tmp_dy_shuff = numpy.zeros((ny, nx))
                   tmp_dy_shuff[0,:] = numpy.angle(numpy.conjugate(newShuffledAnalyticSig[0,:,t])*newShuffledAnalyticSig[1,:,t])/y_ps
                   tmp_dy_shuff[-1,:] = numpy.angle(numpy.conjugate(newShuffledAnalyticSig[-2,:,t])*newShuffledAnalyticSig[-1,:,t])/y_ps
                   tmp_dy_shuff[1:-1,:] = numpy.angle(numpy.conjugate(newShuffledAnalyticSig[:-2,:,t])*newShuffledAnalyticSig[2:,:,t])/(2 * y_ps)

                   dy_shuff[:,:,t] = -signShuffled * tmp_dy_shuff

                   tmp_dx_shuff = numpy.zeros((ny, nx))
                   tmp_dx_shuff[:,0] = numpy.angle(numpy.conjugate(newShuffledAnalyticSig[:,0,t])*newShuffledAnalyticSig[:,1,t])/x_ps
                   tmp_dx_shuff[:,-1] = numpy.angle(numpy.conjugate(newShuffledAnalyticSig[:,-2,t])*newShuffledAnalyticSig[:,-1,t])/x_ps
                   tmp_dx_shuff[:,1:-1] = numpy.angle(numpy.conjugate(newShuffledAnalyticSig[:,:-2,t])*newShuffledAnalyticSig[:,2:,t])/(2 * x_ps)

                   dx_shuff[:,:,t] = -signShuffled * tmp_dx_shuff


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
               for y in range(ny):
                   rdx = []
                   rdy = []
                   rpm = []
                   rpd = []
                   rwl = []
                   rswl = []
                   rws = []
                   rsigwl = []
                   for x in range(nx):
                       dx_tmp = dx[y,x,:]
                       dy_tmp = dy[y,x,:]
                       pm_tmp = numpy.sqrt(dx_tmp **2 + dy_tmp ** 2)/(2 * numpy.pi) 
                       wl_tmp = 1/pm_tmp
                       rdx.append(NeoAnalogSignal(dx_tmp,t_start=new_t_start, sampling_period=sampling_period, units=qt.um))
                       rdy.append(NeoAnalogSignal(dy_tmp,t_start=new_t_start, sampling_period=sampling_period, units=qt.um))
                       # Compute the gradient magnitudes
                       rpm.append(NeoAnalogSignal(pm_tmp,t_start=new_t_start, sampling_period=sampling_period, units=qt.um))
                       # Compute the gradient directions 
                       rpd.append(NeoAnalogSignal(numpy.arctan2(dy_tmp, dx_tmp),t_start=new_t_start, sampling_period=sampling_period, units=qt.rad))
                       # Compute the wavelengths
                       rwl.append(NeoAnalogSignal(wl_tmp,t_start=new_t_start, sampling_period=sampling_period, units=qt.um))
                       rswl.append(NeoAnalogSignal(1/(numpy.sqrt(dx_shuff[y,x,:] **2 + dy_shuff[y,x,:] ** 2)/(2 * numpy.pi)),t_start=new_t_start, sampling_period=sampling_period, units=qt.um))
                       # Compute the wave speeds
                       rws.append(NeoAnalogSignal(painstFreqs[y][x].magnitude[:,0]/pm_tmp[:-1],t_start=new_t_start, sampling_period=sampling_period, units=qt.um * units))
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
                   PerAreaAnalogSignalList(new_sig,paasl.x_coords,paasl.y_coords,qt.dimensionless,
                                  stimulus_id=paasl.stimulus_id,
                                  x_axis_name=paasl.x_axis_name,
                                  y_axis_name=f'GP of ({paasl.y_axis_name})',
                                  sheet_name=sheet,
                                  tags=self.tags,
                                  analysis_algorithm=self.__class__.__name__))
               self.datastore.full_datastore.add_analysis_result(
                   PerAreaAnalogSignalList(phases,paasl.x_coords,paasl.y_coords,units_phase,
                                  stimulus_id=paasl.stimulus_id,
                                  x_axis_name=paasl.x_axis_name,
                                  y_axis_name=f'Phases of ({paasl.y_axis_name})',
                                  sheet_name=sheet,
                                  tags=self.tags,
                                  analysis_algorithm=self.__class__.__name__))
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
      'vm': bool,  # calculate for Vm?
      'cond_exc': bool,  # calculate for excitatory conductance?
      'cond_inh': bool,  # calculate for inhibitory conductance?
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

        if self.parameters.type == 'band':
            low_frequency, high_frequency = self.parameters.low_frequency, self.parameters.high_frequency 
        elif self.parameters.type == 'high':
            low_frequency, high_frequency = 'NaN', self.parameters.high_frequency 
        elif self.parameters.type == 'low':
            low_frequency, high_frequency = self.parameters.low_frequency, 'NaN' 

        for sheet in self.datastore.sheets():
            dsv = queries.param_filter_query(self.datastore, sheet_name=sheet) 

            # This part of the code specific to AnalogSignalList was not tested
            dsv1 = queries.param_filter_query(dsv, sheet_name=sheet, identifier=['AnalogSignalList','PerNeuronPairAnalogSignalList'])

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
                    fasl.append(NeoAnalogSignal(filtfilt(b, a, asignal.magnitude[:,0]),t_start=t_start, sampling_period=sampling_period, units=units))

                self.datastore.full_datastore.add_analysis_result(
                    AnalogSignalList(fasl,ads.ids,units,
                                   stimulus_id=ads.stimulus_id,
                                   x_axis_name=ads.x_axis_name,
                                   y_axis_name=f'Butterworth {self.parameters.type}-pass filtered of ({ads.y_axis_name}) freq=[{low_frequency},{high_frequency}], order = {self.parameters.order}',
                                   sheet_name=sheet,
                                   tags=self.tags,
                                   analysis_algorithm=self.__class__.__name__))


            # PerAreaAnalogSignalList part
            dsv1 = queries.param_filter_query(dsv, identifier=['PerAreaAnalogSignalList'])
            for paasl in dsv1.get_analysis_result():
                asl = paasl.asl
                sampling_period = asl[0][0].sampling_period
                t_start = asl[0][0].t_start
                units = asl[0][0].units
                
                # Get the parameters of the filter
                b, a = self.get_parameters_filter(sampling_period)

                # Apply the filter on each AnalogSignal
                fasl=[] 
                for y in range(len(asl)):
                    row = []
                    for x in range(len(asl)):
                        row.append(NeoAnalogSignal(filtfilt(b, a, asl[y][x].magnitude[:,0]),t_start=t_start, sampling_period=sampling_period, units=units))
                    fasl.append(row)

                self.datastore.full_datastore.add_analysis_result(
                    PerAreaAnalogSignalList(fasl,paasl.x_coords,paasl.y_coords,units,
                                   stimulus_id=paasl.stimulus_id,
                                   x_axis_name=paasl.x_axis_name,
                                   y_axis_name=f'Butterworth {self.parameters.type}-pass filtered of ({paasl.y_axis_name}) freq=[{low_frequency},{high_frequency}], order = {self.parameters.order}',
                                   sheet_name=sheet,
                                   tags=self.tags,
                                   analysis_algorithm=self.__class__.__name__))

            segs = dsv.get_segments()

            # Vm and Conductances part
            for seg in segs:
                st = seg.annotations['stimulus']
                for asl in seg.analogsignals:
                    sampling_period = asl.sampling_period
                    t_start = asl.t_start
                    units = asl.units

                    # Get the parameters of the filter
                    b, a = self.get_parameters_filter(sampling_period)

                    # Apply the filter on each AnalogSignal
                    fasl=[]
                    for asignal in asl.T:
                        fasl.append(NeoAnalogSignal(filtfilt(b, a, asignal.magnitude),t_start=t_start, sampling_period=sampling_period, units=units))
                    
                    if (asl.name =='v' or asl.name == 'V_m') and self.parameters.vm:
                        vm_ids = seg.get_stored_vm_ids()
                        self.datastore.full_datastore.add_analysis_result(
                            AnalogSignalList(fasl,vm_ids,units,
                                           stimulus_id=st,
                                           x_axis_name='time',
                                           y_axis_name=f'Butterworth {self.parameters.type}-pass filtered of Vm freq=[{low_frequency},{high_frequency}], order = {self.parameters.order}',
                                           sheet_name=sheet,
                                           tags=self.tags,
                                           analysis_algorithm=self.__class__.__name__))
                    elif (asl.name =='gsyn_exc'  or asl.name == 'g_exc') and self.parameters.cond_exc:
                        esyn_ids = seg.get_stored_esyn_ids()
                        self.datastore.full_datastore.add_analysis_result(
                            AnalogSignalList(fasl,esyn_ids,units,
                                           stimulus_id=st,
                                           x_axis_name='time',
                                           y_axis_name=f'Butterworth {self.parameters.type}-pass filtered of ECond freq=[{low_frequency},{high_frequency}], order = {self.parameters.order}',
                                           sheet_name=sheet,
                                           tags=self.tags,
                                           analysis_algorithm=self.__class__.__name__))


                        self.datastore.full_datastore.add_analysis_result(
                            AnalogSignalList(fasl,isyn_ids,units,
                                           stimulus_id=st,
                                           x_axis_name='time',
                                           y_axis_name=f'Butterworth {self.parameters.type}-pass filtered of ICond freq=[{low_frequency},{high_frequency}], order = {self.parameters.order}',
                                           sheet_name=sheet,
                                           tags=self.tags,
                                           analysis_algorithm=self.__class__.__name__))



class LFPFromPSTH(Analysis):
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
            X,Y=numpy.meshgrid(numpy.linspace(0,y_resolution-1,y_resolution),numpy.linspace(0,x_resolution-1,x_resolution))
            return numpy.exp(-((X-x)**2+(Y-y)**2)/(2.0*sigma**2))

        for sheet in self.datastore.sheets():

            dsv1 = queries.param_filter_query(self.datastore, sheet_name=sheet,analysis_algorithm='PSTH')
            psths = dsv1.get_analysis_result()
            analog_ids = psths[0].ids
            st = [MozaikParametrized.idd(s) for s in self.datastore.get_stimuli()]

            # Get the size of the sheet
            sx = eval(dsv1.full_datastore.block.annotations['sheet_parameters'])[sheet]['sx']
            sy = eval(dsv1.full_datastore.block.annotations['sheet_parameters'])[sheet]['sy']

            # Get the size of the border
            dx = sx - self.parameters.cropped_length
            dy = sy - self.parameters.cropped_length

            dix = int(dx/2/self.parameters.points_distance)
            diy = int(dy/2/self.parameters.points_distance)

            if dix == 0 and diy == 0:
                cropped_suffix = ""
            else:
                cropped_suffix = " cropped"

            interpoint_resolution = 1

            # Create the coordinates of the points around which the lfp will be computed
            x_axis = numpy.arange(-sx/2 + self.parameters.points_distance/2, sx/2 + self.parameters.points_distance/2, self.parameters.points_distance/interpoint_resolution)
            y_axis = numpy.arange(-sy/2 + self.parameters.points_distance/2, sy/2 + self.parameters.points_distance/2, self.parameters.points_distance/interpoint_resolution)

            # resolution on both spatial dimensions
            x_resolution = x_axis.shape[0]
            y_resolution = y_axis.shape[0]

            for psth, s in zip(psths, st):
                t_start = psth.get_asl_by_id(analog_ids[0]).t_start
                time_step = psth.get_asl_by_id(analog_ids[0]).sampling_period

                # temporal resolution
                t_resolution = psth.get_asl_by_id(analog_ids[0]).shape[0] 

                # Initialize the LFP tensor
                m = numpy.zeros((x_resolution,y_resolution,t_resolution))

                # Calculate the lfps generated by the analog signals of each neuron
                for aid in analog_ids:
                    # Get the position of the neurons
                    sid = self.datastore.get_sheet_indexes(sheet,aid)
                    x = dsv1.get_neuron_positions()[sheet][0][sid] * 1000
                    y = dsv1.get_neuron_positions()[sheet][1][sid] * 1000

                    # Get the LFP position corresponding to the position of the neuron
                    x_id = int((x + sx/2)/self.parameters.points_distance)
                    y_id = int((y + sy/2)/self.parameters.points_distance)

                    lfp = numpy.reshape(psth.get_asl_by_id(aid),-1)

                    full_signal = lfp
                    full_signal = numpy.expand_dims(full_signal, axis=(1,2))
                    full_signal = numpy.concatenate([full_signal for _ in range(interpoint_resolution)], axis=1)
                    full_signal = numpy.concatenate([full_signal for _ in range(interpoint_resolution)], axis=2)
                    full_signal = numpy.transpose(full_signal,(1,2,0))

                    m[y_id * interpoint_resolution:(y_id+1) * interpoint_resolution,x_id *interpoint_resolution:(x_id+1) * interpoint_resolution,:] += full_signal



                # Convolve the lfps with a gaussian kernel
                if self.parameters.gaussian_convolution:
                    gauss_suffix = ""
                    m_convolved = numpy.zeros((y_resolution,x_resolution,t_resolution))
                    for y in range(m.shape[0]):
                        for x in range(m.shape[1]):
                            gauss = gaussian_kernel(x,y,self.parameters.gaussian_sigma)
                            gauss = numpy.expand_dims(gauss, axis=2)
                            m_convolved[y,x,:] = numpy.sum(m * gauss, axis = (0,1))
                    m = m_convolved
                else:
                    gauss_suffix = " without convolution"

                # Cropping
                if dix and diy:
                    m = m[diy:-diy,dix:-dix]
                    x_axis_cropped = x_axis[dix:-dix]
                    y_axis_cropped = y_axis[diy:-diy]
                else:
                    x_axis_cropped = x_axis
                    y_axis_cropped = y_axis

                # Normalization of the signal 
                avg = numpy.mean(m)
                std = numpy.std(m)
                m = (m - avg)/std

                # Convert the tensor to a PerAreaAnalogSignalList and add it to the datastore
                lfps = []
                for y in range(m.shape[0]):
                    row = []
                    for x in range(m.shape[1]):
                        row.append(NeoAnalogSignal(m[y,x],t_start=t_start, sampling_period=time_step, units=qt.dimensionless))
                    lfps.append(row)
                self.datastore.full_datastore.add_analysis_result(
                    PerAreaAnalogSignalList(lfps,x_axis_cropped,y_axis_cropped,lfps[0][0].units,
                                   stimulus_id=str(s),
                                   x_axis_name='time',
                                   y_axis_name='PSTH LFP' + cropped_suffix + gauss_suffix,
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
            X,Y=numpy.meshgrid(numpy.linspace(0,y_resolution-1,y_resolution),numpy.linspace(0,x_resolution-1,x_resolution))
            return numpy.exp(-((X-x)**2+(Y-y)**2)/(2.0*sigma**2))

        for sheet in self.datastore.sheets():

            dsv1 = queries.param_filter_query(self.datastore, sheet_name=sheet)
            segs = dsv1.get_segments()
            analog_ids = segs[0].get_stored_esyn_ids()
            st = [MozaikParametrized.idd(s) for s in dsv1.get_stimuli()]

            # Get the size of the sheet
            sx = eval(dsv1.full_datastore.block.annotations['sheet_parameters'])[sheet]['sx']
            sy = eval(dsv1.full_datastore.block.annotations['sheet_parameters'])[sheet]['sy']

            #Get the reversal potentials
            e_rev_E = eval(dsv1.full_datastore.block.annotations['sheet_parameters'])[sheet]['cell']['params']['e_rev_E']
            e_rev_I = eval(dsv1.full_datastore.block.annotations['sheet_parameters'])[sheet]['cell']['params']['e_rev_I']

            # Get the size of the border
            dx = sx - self.parameters.cropped_length
            dy = sy - self.parameters.cropped_length

            dix = int(dx/2/self.parameters.points_distance)
            diy = int(dy/2/self.parameters.points_distance)

            if dix == 0 and diy == 0:
                cropped_suffix = ""
            else:
                cropped_suffix = " cropped"

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
                    x_id = int((x + sx/2)/self.parameters.points_distance)
                    y_id = int((y + sy/2)/self.parameters.points_distance)

                    # Get the analog signals of the neuron
                    e_syn = seg.get_esyn(aid).downsample(self.parameters.downsampling)
                    time_step = e_syn.sampling_period
                    e_syn = numpy.transpose(e_syn.magnitude,(1,0))[0]
                    i_syn = numpy.transpose(seg.get_isyn(aid).downsample(self.parameters.downsampling).magnitude,(1,0))[0]
                    vm = numpy.transpose(seg.get_vm(aid).downsample(self.parameters.downsampling).magnitude,(1,0))[0]

                    # This LFP proxy is optimal when excitation is lagged by 6ms compared to inhibition
                    idiff = int(6/time_step)

                    lfp = ((e_rev_E - vm) * e_syn)[:-idiff] - 1.65 * ((e_rev_I - vm) * i_syn)[idiff:] #Same proxy as Davis et al., 2021 

                    full_signal = lfp
                    full_signal = numpy.expand_dims(full_signal, axis=(1,2)) 
                    full_signal = numpy.concatenate([full_signal for _ in range(interpoint_resolution)], axis=1)
                    full_signal = numpy.concatenate([full_signal for _ in range(interpoint_resolution)], axis=2)
                    full_signal = numpy.transpose(full_signal,(1,2,0))

                    m[y_id * interpoint_resolution:(y_id+1) * interpoint_resolution,x_id * interpoint_resolution:(x_id+1) * interpoint_resolution,:] += full_signal


                
                # Convolve the lfps with a gaussian kernel
                if self.parameters.gaussian_convolution:
                    gauss_suffix = ""
                    m_convolved = numpy.zeros((y_resolution,x_resolution,t_resolution))
                    for y in range(m.shape[0]):
                        for x in range(m.shape[1]):
                            gauss = gaussian_kernel(x,y,self.parameters.gaussian_sigma)
                            gauss = numpy.expand_dims(gauss, axis=2)
                            m_convolved[y,x,:] = numpy.sum(m * gauss, axis = (0,1))
                    m = m_convolved
                else:
                    gauss_suffix = " without convolution"
                
                # Cropping
                if dix and diy:
                    m = m[diy:-diy,dix:-dix]
                    x_axis_cropped = x_axis[dix:-dix]
                    y_axis_cropped = y_axis[diy:-diy]
                else:
                    x_axis_cropped = x_axis
                    y_axis_cropped = y_axis

                # Normalization of the signal 
                avg = numpy.mean(m)
                std = numpy.std(m)
                m = (m - avg)/std

                # Convert the tensor to a PerAreaAnalogSignalList and add it to the datastore
                lfps = []
                for y in range(m.shape[0]):
                    row = []
                    for x in range(m.shape[1]):
                        row.append(NeoAnalogSignal(m[y,x],t_start=t_start, sampling_period=time_step, units=qt.dimensionless))
                    lfps.append(row)
                self.datastore.full_datastore.add_analysis_result(
                    PerAreaAnalogSignalList(lfps,x_axis_cropped,y_axis_cropped,lfps[0][0].units,
                                   stimulus_id=str(s),
                                   x_axis_name='time',
                                   y_axis_name='LFP' + cropped_suffix + gauss_suffix,
                                   sheet_name=sheet,
                                   tags=self.tags,
                                   analysis_algorithm=self.__class__.__name__))

class FindSourceWave(Analysis):
    required_parameters = ParameterSet({
        'evaluation_phase': float, # rad. The phase of the signal for which we evaluate the crossings
        'tolerance': float, # rad. The tolerance for detecting waves
        'value_name': str,
    })

    def perform_analysis(self):

        def divergence(x,y,fx,fy):
            x = x[:,numpy.newaxis]
            dfx = numpy.zeros(fx.shape)
            dfx[:,1:-1] = (fx[:,2:] - fx[:,:-2])/(x[2:] - x[:-2]) 
            dfx[:,0] = (fx[:,1] - fx[:,0])/(x[1] - x[0]) 
            dfx[:,-1] = (fx[:,-1] - fx[:,-2])/(x[-1] - x[-2]) 

            y = y[:,numpy.newaxis,numpy.newaxis]
            dfy = numpy.zeros(fy.shape)
            dfy[1:-1,:] = (fy[2:,:] - fy[:-2,:])/(y[2:] - y[:-2]) 
            dfy[0,:] = (fy[1,:] - fy[0,:])/(y[1] - y[0]) 
            dfy[-1,:] = (fy[-1,:] - fy[-2,:])/(y[-1] - y[-2]) 

            return dfx + dfy


        for sheet in self.datastore.sheets():
            dsv = queries.param_filter_query(self.datastore, sheet_name=sheet, identifier=['PerAreaAnalogSignalList'])
            signal = queries.param_filter_query(dsv, y_axis_name = f'GP of ({self.parameters.value_name})', ads_unique=True).get_analysis_result()[0]
            gradient_x = queries.param_filter_query(dsv, y_axis_name = f'X phase gradient of ({self.parameters.value_name})', ads_unique=True).get_analysis_result()[0]
            gradient_y = queries.param_filter_query(dsv, y_axis_name = f'Y phase gradient of ({self.parameters.value_name})', ads_unique=True).get_analysis_result()[0]
            
            asl = signal.asl
            gradient_x_asl = numpy.array(gradient_x.asl)[:,:,:,0]
            gradient_y_asl = numpy.array(gradient_y.asl)[:,:,:,0]

            sampling_period = asl[0][0].sampling_period
            t_start = asl[0][0].t_start
            new_t_start = t_start

            asl_arr = numpy.array(asl)[:,:,:,0]
            ny = asl_arr.shape[0]
            nx = asl_arr.shape[1]
            dur = asl_arr.shape[2]
            
            spatial_sum = numpy.sum(asl_arr,axis=(0,1))/(nx*ny)
            circular_dist_v = numpy.vectorize(circular_dist)
            phase_diff = circular_dist_v(numpy.angle(spatial_sum),self.parameters.evaluation_phase,2*numpy.pi)
            sign_diff = numpy.sign(phase_diff[1:] - phase_diff[:-1])
            ts = numpy.nonzero(sign_diff[1:] - sign_diff[:-1] == 2)[0] + 1
            ts_tol = ts[phase_diff[ts] < self.parameters.tolerance]

            X, Y = numpy.meshgrid(numpy.arange(nx),numpy.arange(ny))

            div_array = numpy.zeros((ny,nx,ts_tol.shape[0]))
            source_points = numpy.zeros((2,ts_tol.shape[0]))
            div_array = divergence(numpy.arange(nx),numpy.arange(ny),gradient_x_asl[:,:,ts_tol],gradient_y_asl[:,:,ts_tol])
            for i in range(len(ts_tol)):
                argmax = numpy.argmax(div_array[:,:,i])
                source_points[0,i] = argmax//nx
                source_points[1,i] = argmax%nx

            self.datastore.full_datastore.add_analysis_result(
                SingleObject(ts_tol*sampling_period+t_start, sampling_period.units,
                               stimulus_id=signal.stimulus_id,
                               object_name=f'Evaluation points of ({self.parameters.value_name})',
                               sheet_name=sheet,
                               tags=self.tags,
                               analysis_algorithm=self.__class__.__name__))

            self.datastore.full_datastore.add_analysis_result(
                SingleObject(ts_tol, qt.Dimensionless,
                               stimulus_id=signal.stimulus_id,
                               object_name=f'Evaluation points indices of ({self.parameters.value_name})',
                               sheet_name=sheet,
                               tags=self.tags,
                               analysis_algorithm=self.__class__.__name__))

            self.datastore.full_datastore.add_analysis_result(
                SingleObject(source_points, qt.Dimensionless,
                               stimulus_id=signal.stimulus_id,
                               object_name=f'Source points of ({self.parameters.value_name})',
                               sheet_name=sheet,
                               tags=self.tags,
                               analysis_algorithm=self.__class__.__name__))


class FindStartPropagation(Analysis):
    """
    This analysis takes each PerAreaAnalogSignalList in the dsv, and create a list object containing
    the time points that are selected to analyze the start of the spatial propagation of the activity.

    The following criteria apply to the selected time points:
    1) The maximum local activity within this time point should exceed `activity_threshold`, as a 
    fraction of the maximum activity of the signal.
    2) For this time point, the spatial bins for which the activity exceeds a fraction (defined as
    `spatial_bins_threshold`) of the maximum local activity should be adjacent to one another
    3) For this time point, the number of spatial bins for which the activity exceeds a fraction 
    4) There was no selected time point within the last `time_spacing` time points
    (defined as `spatial_bins_threshold`) should be lower than `max_area_activity`.
    5) The following m time points should also validate 2), with m defnined by the parameter
    `n_ts_adjacent`
    6) The mean activity across the n following time point should be higher than
    `activity_threshold_next', with n defined by the parameter `n_ts_activity` 



    Other parameters
    ----------------
    """
    required_parameters = ParameterSet({
        'activity_threshold': float, # The threshold of the local maximum value of activity for a time point to be selected, normalized by the maximal acitivity of the signal  
        'spatial_bins_threshold': float, # 
        'max_area_activity': int, # 
        'time_spacing': int, # 
        'n_ts_adjacent': int, # 
        'activity_threshold_next': float, # 
        'n_ts_activity': int, # 
    })

    def perform_analysis(self):
        for sheet in self.datastore.sheets():
            dsv1 = queries.param_filter_query(self.datastore, sheet_name=sheet, identifier=['PerAreaAnalogSignalList'])
            for paasl in dsv1.get_analysis_result():
                asl = paasl.asl
                sampling_period = asl[0][0].sampling_period
                t_start = asl[0][0].t_start
                name = paasl.y_axis_name
                signal = numpy.array(asl)[:,:,:,0] 
                duration = signal.shape[2]
                min_ = numpy.min(signal)
                signal = numpy.squeeze(signal - min_)
                max_ = numpy.max(signal)
                
                ts = numpy.arange(duration)[:]

                # Only time points with maximal local activity higher than threshold are selected
                ts = ts[numpy.max(signal[:,:,ts],axis=(0,1)) > max_ * self.parameters.activity_threshold]
                
                ts_tol = []
                for t in ts:
                    s = signal[:,:,t]

                    # Check if any of the n_ts_adjacent time points contains multiple clusters of activity
                    non_adjacent = False
                    for dt in range(1,self.parameters.n_ts_adjacent + 1):
                        mask = signal[:,:,t+dt] > numpy.max(signal[:,:,t+dt]) * self.parameters.spatial_bins_threshold
                        _, num_labels = measure.label(mask, connectivity=2, return_num=True)
                        if num_labels > 1:
                            non_adjacent = True

                    # Compute the number of spatial bins at time t which activity exceeds spatial_bins_threshold, and check whether they form multiple clusters
                    mask = s > numpy.max(s) * self.parameters.spatial_bins_threshold
                    _, num_labels = measure.label(mask, connectivity=2, return_num=True)

                    # Check whether the spacing with the last selected time point is sufficient 
                    if (len(ts_tol)==0 or t - ts_tol[-1] > self.parameters.time_spacing):
                        sufficient_spacing = True
                    else:
                        sufficient_spacing = False

                    # Check whether the activity in the following time points is sufficiently high
                    if numpy.mean([numpy.max(signal[:,:,t+dt])for dt in numpy.arange(1,self.parameters.n_ts_activity)])/max_ > self.parameters.activity_threshold_next:
                        active_following_time_points = True
                    else: 
                        active_following_time_points = False


                    if num_labels == 1 and not non_adjacent and numpy.sum(mask) < self.parameters.max_area_activity and sufficient_spacing and active_following_time_points:
                        ts_tol.append(t)    

                self.datastore.full_datastore.add_analysis_result(
                    SingleObject(ts_tol*sampling_period+t_start, sampling_period.units,
                                   stimulus_id=paasl.stimulus_id,
                                   object_name=f'Start propagation time points of ({name})',
                                   sheet_name=sheet,
                                   tags=self.tags,
                                   analysis_algorithm=self.__class__.__name__))

                self.datastore.full_datastore.add_analysis_result(
                    SingleObject(ts_tol, qt.Dimensionless,
                                   stimulus_id=paasl.stimulus_id,
                                   object_name=f'Start propagation time points indices of ({name})',
                                   sheet_name=sheet,
                                   tags=self.tags,
                                   analysis_algorithm=self.__class__.__name__))



class SignalPropagationOrientationBias(Analysis):
    required_parameters = ParameterSet({
    'map_file': str,
    'cropped_length': float,
    'timesteps': int, 
    'spatial_bins_threshold': float, 
    })

    def generate_surrogates(self, crop_coefficient, bins):
        """
        Generate surrogates based on an orientation map.

        Parameters
        ----------

        crop_coefficient : float
             Determines the size of the map that is cropped 

        bins: int
             The number of spatial bins in each axis
        Output
        --------

        surrogates: list
             List containing every surrogated represented as 2-D arrays of shape (bins,bins)
        """

        f = open(self.parameters.map_file, 'rb')
        or_map = pickle.load(f, encoding="latin1")
        f.close()
    
        cutout_size = int(or_map.shape[0] / crop_coefficient)
        sz = or_map.shape[0]
    
        surrogates = []
        x,y = sz/2-cutout_size/2,sz/2-cutout_size/2
    
        numpy.random.seed(1001)
        for angle in [0,90,180,270]:
            for flip in [True,False]:
                I = Image.fromarray(or_map*255)
                I = I.rotate(angle,fillcolor=255)
                if flip:
                    I = I.transpose(method=Image.FLIP_LEFT_RIGHT)
                I = I.crop((x,y,x+cutout_size,y+cutout_size))
                a = numpy.asarray(I)/255
                if len(numpy.where(a == 1)[0]) == 0:
                    coords_xa = numpy.arange(a.shape[1])
                    coords_ya = numpy.arange(a.shape[0])
                    Ya, Xa = numpy.meshgrid(coords_ya, coords_xa)
                    ND = NearestNDInterpolator(list(zip(Xa.flatten(), Ya.flatten())),a.flatten())
                    coords_xb = numpy.linspace(0,a.shape[1]-1,orientation_matrix.shape[1])
                    coords_yb = numpy.linspace(0,a.shape[0]-1,orientation_matrix.shape[0])
                    Yb, Xb = numpy.meshgrid(coords_yb, coords_xb)
                    b = ND((Yb,Xb)) * numpy.pi
                    if flip or angle!=0:
                        surrogates.append(b)
                    else:
                        same = b
        return surrogates,same

    def propagation_signal(signal,signal_conv,orientation_matrix,surrogates,ts_tol):
        min_ = numpy.min(signal)
        signal = signal - min_
        max_ = numpy.max(signal)
        
        min_conv = numpy.min(signal_conv)
        signal_conv = signal_nconv - min_conv
        max_conv = numpy.max(signal_conv)
        
        orrs_means=[]
        surrs_means=[]
        orrs_lens = []
        surrs_lens = []
        orrs_means_conv=[]
        surrs_means_conv=[]
        surrs_lens_conv = []
        orrs_lens_conv = []
        base_means=[]
        base_lens = []
    
        ones = numpy.ones(orientation_matrix.shape)
    
        circular_dist_v = numpy.vectorize(circular_dist)
    
        for t_start in ts_tol:
            orrs_mean = []
            orrs_len = []
            surrs_mean = []
            surrs_len = []
            orrs_mean_conv = []
            orrs_len_conv = []
            surrs_mean_conv = []
            surrs_len_conv = []
            base_mean = []
            base_len = []
    
            sig0 = numpy.squeeze(signal[:,:,t_start])
            sig_conv0 = numpy.squeeze(signal_conv[:,:,t_start])
            orrs_mean_t0 = circ_mean(orientation_matrix,sig0,high=numpy.pi)[0]
            orrs_len_t0 = circ_len(orientation_matrix,sig0,high=numpy.pi)
            surrs_mean_t0 = [circ_mean(r,sig0,high=numpy.pi)[0] for r in surrogates]
            surrs_len_t0 = [circ_len(r,sig,high=numpy.pi) for r in surrogates]
            orrs_mean_conv_t0 = circ_mean(orientation_matrix,sig_conv0,high=numpy.pi)[0]
            orrs_len_conv_t0 = circ_len(orientation_matrix,sig_conv0,high=numpy.pi)
            surrs_mean_conv_t0 = [circ_mean(r,sig_conv0,high=numpy.pi)[0] for r in surrogates]
            surrs_len_conv_t0 = [circ_len(r,sig_conv0,high=numpy.pi) for r in surrogates]
            s_zone = sig_conv0 > numpy.max(sig_conv0) * self.parameters.spatial_bins_threshold 

            for t in range(self.parameters.timesteps):
                sig = numpy.squeeze(signal[:,:,t_start + t])
                sig_conv = numpy.squeeze(signal_conv[:,:,t_start + t])
    
                orrs_mean.append(circ_mean(orientation_matrix[~s_zone],sig[~s_zone],high=numpy.pi)[0])
                orrs_len.append(circ_len(orientation_matrix[~s_zone],sig[~s_zone],high=numpy.pi))
                surrs_mean.append([circ_mean(r[~s_zone],sig[~s_zone],high=numpy.pi)[0] for r in surrogates])
                surrs_len.append([circ_len(r[~s_zone],sig[~s_zone],high=numpy.pi) for r in surrogates])
                orrs_mean_conv.append(circ_mean(orientation_matrix[~s_zone],sig_conv[~s_zone],high=numpy.pi)[0])
                orrs_len_conv.append(circ_len(orientation_matrix[~s_zone],sig_conv[~s_zone],high=numpy.pi))
                surrs_mean_conv.append([circ_mean(r[~s_zone],sig_conv[~s_zone],high=numpy.pi)[0] for r in surrogates])
                surrs_len_conv.append([circ_len(r[~s_zone],sig_conv[~s_zone],high=numpy.pi) for r in surrogates])
    
            orrs_mean = numpy.array(orrs_mean)
            diffs = numpy.abs(circular_dist(orrs_mean,orrs_mean_t0,numpy.pi))
            orrs_means.append(diffs)
            orrs_lens.append(orrs_len)
            surrs_mean=numpy.array(surrs_mean)
            surrs_mean_t0=numpy.array(surrs_mean_t0)
            diffs_surr = numpy.abs(circular_dist_v(surrs_mean,surrs_mean_t0,numpy.pi))
            orrs_mean_conv = numpy.array(orrs_mean_conv)
            diffs_conv = numpy.abs(circular_dist(orrs_mean_conv,orrs_mean_conv_t0,numpy.pi))
            orrs_means_conv.append(diffs_conv)
            orrs_lens_conv.append(orrs_len_conv)
            surrs_mean_conv=numpy.array(surrs_mean_conv)
            surrs_mean_conv_t0=numpy.array(surrs_mean_conv_t0)
            diffs_surr_conv = numpy.abs(circular_dist_v(surrs_mean_conv,surrs_mean_conv_t0,numpy.pi))
            base_mean = numpy.array(base_mean)
            diffs_base = numpy.abs(circular_dist(base_mean,base_mean_t0,numpy.pi))
            surrs_avg = numpy.mean(diffs_surr, axis = 1)
            surrs_avg_conv = numpy.mean(diffs_surr_conv,axis = 1)
            surrs_means.append(surrs_avg)
            surrs_means_conv.append(surrs_avg_conv)
    
            surrs_len=numpy.array(surrs_len)
            surrs_len_conv=numpy.array(surrs_len_conv)
            surrs_avg = numpy.mean(surrs_len, axis = 1)
            surrs_avg_conv = numpy.mean(surrs_len_conv,axis = 1)
            surrs_lens.append(surrs_avg)
            surrs_lens_conv.append(surrs_avg_conv)
    
        return orrs_means, surrs_means, orrs_means_conv, surrs_means_conv, orrs_lens, surrs_lens, orrs_lens_conv, surrs_lens_conv

    def perform_analysis(self):
        for sheet in self.datastore.sheets():
           dsv = queries.param_filter_query(self.datastore, sheet_name=sheet) 
           sx = eval(dsv.full_datastore.block.annotations['sheet_parameters'])[sheet]['sx']
           map_stretch = eval(dsv.full_datastore.block.annotations['sheet_parameters'])['V1_Exc_L4']['AfferentConnection']['or_map_stretch']
           tag_cropped = f'Cropping: {self.parameters.cropped_length}'
           tag_ncropped = f'Cropping: {sx}'
           tags_cropped = self.tags.copy().append(tag_cropped)
           tags_ncropped = self.tags.copy().append(tag_ncropped)
           dsv_cropped = queries.tag_based_query(dsv, tags_cropped)
           dsv_ncropped = queries.tag_based_query(dsv, tags_ncropped)
           paasl_cropped = queries.param_filter_query(dsv_cropped, sheet_name=sheet, y_axis_name='LFP without convolution').get_analysis_result()[0]
           paasl = queries.param_filter_query(dsv_ncropped, sheet_name=sheet, y_axis_name='LFP without convolution').get_analysis_result()[0]
           paasl_conv_cropped = queries.param_filter_query(dsv_cropped, sheet_name=sheet, y_axis_name='LFP').get_analysis_result()[0]
           paasl_conv = queries.param_filter_query(dsv_ncropped, sheet_name=sheet, y_axis_name='LFP').get_analysis_result()[0]

           ts_tol = queries.param_filter_query(dsv, y_axis_name='Start propagation time points indices of (LFP)',ads_unique=True).get_analysis_result()[0].object

           signal_cropped = numpy.squeeze(numpy.array(paasl_cropped.asl))
           signal = numpy.squeeze(numpy.array(paasl.asl))
           signal_conv_cropped = numpy.squeeze(numpy.array(paasl_conv_cropped.asl))
           signal_conv = numpy.squeeze(numpy.array(paasl[0]))
           ny, ny, duration = signal.shape
           ny_cropped = signal.shape[0]

           min_cropped = numpy.min(signal_cropped)
           signal_cropped = signal_cropped - min_cropped
           max_cropped = numpy.max(signal)
       
           min_conv_cropped = numpy.min(signal_conv_cropped)
           signal_conv_cropped = signal_nconv_cropped - min_conv_cropped
           max_conv_cropped = numpy.max(signal_conv_cropped)
       
           sampling_period = paasl[0][0].sampling_period
           t_start = paasl[0][0].t_start

           crop_coefficient = map_stretch 
           crop_coefficient_cropped = map_stretch * sx / self.parameters.cropped_length  

           surrogates, orientation_matrix = self.generate_surrogates(crop_coefficient,ny)
           surrogates_cropped, orientation_matrix_cropped = self.generate_surrogates(crop_coefficient_cropped,ny_cropped)

           orrs_means, surrs_means, orrs_means_conv, surrs_means_conv, orrs_lens, surrs_lens, orrs_lens_conv, surrs_lens_conv = propagation_signal(signal,signal_conv,orientation_matrix,surrogates,ts_tol)
           orrs_means_cropped, surrs_means_cropped, orrs_means_conv_cropped, surrs_means_conv_cropped, orrs_lens_cropped, surrs_lens_cropped, orrs_lens_conv_cropped, surrs_lens_conv_cropped = propagation_signal(signal,signal_conv,orientation_matrix_cropped,surrogates_cropped,ts_tol)

           self.datastore.full_datastore.add_analysis_result(
               PerAreaValue(orientation_matrix,paasl.x_coords,paasl.y_coords,qt.rad,
                              stimulus_id=paasl.stimulus_id,
                              value_name='Orientation label of '+paasl.y_axis_name,
                              sheet_name=sheet,
                              tags=tags_ncropped,
                              analysis_algorithm=self.__class__.__name__))
           self.datastore.full_datastore.add_analysis_result(
               PerAreaValue(orientation_matrix_cropped,paasl_cropped.x_coords,paasl_cropped.y_coords,qt.rad,
                              stimulus_id=paasl_cropped.stimulus_id,
                              value_name='Orientation label of '+paasl_cropped.y_axis_name,
                              sheet_name=sheet,
                              tags=tags_cropped,
                              analysis_algorithm=self.__class__.__name__))
           self.datastore.full_datastore.add_analysis_result(
               SingleObject(orrs_means, qt.rad,
                              stimulus_id=paasl.stimulus_id,
                              object_name='Orientation differences of '+paasl.y_axis_name,
                              sheet_name=sheet,
                              tags=self.tags_ncropped,
                              analysis_algorithm=self.__class__.__name__))
           self.datastore.full_datastore.add_analysis_result(
               SingleObject(surrs_means, qt.rad,
                              stimulus_id=paasl.stimulus_id,
                              object_name='Surrogates orientation differences of '+paasl.y_axis_name,
                              sheet_name=sheet,
                              tags=self.tags_ncropped,
                              analysis_algorithm=self.__class__.__name__))
           self.datastore.full_datastore.add_analysis_result(
               SingleObject(orrs_lens, qt.Dimensionless,
                              stimulus_id=paasl.stimulus_id,
                              object_name='Orientation selectivity of '+paasl.y_axis_name,
                              sheet_name=sheet,
                              tags=self.tags_ncropped,
                              analysis_algorithm=self.__class__.__name__))
           self.datastore.full_datastore.add_analysis_result(
               SingleObject(surrs_lens, qt.Dimensionless,
                              stimulus_id=paasl.stimulus_id,
                              object_name='Surrogates orientation selectivity of '+paasl.y_axis_name,
                              sheet_name=sheet,
                              tags=self.tags_ncropped,
                              analysis_algorithm=self.__class__.__name__))
           self.datastore.full_datastore.add_analysis_result(
               SingleObject(orrs_means_conv, qt.rad,
                              stimulus_id=paasl_conv.stimulus_id,
                              object_name='Orientation differences of '+paasl_conv.y_axis_name,
                              sheet_name=sheet,
                              tags=self.tags_ncropped,
                              analysis_algorithm=self.__class__.__name__))
           self.datastore.full_datastore.add_analysis_result(
               SingleObject(surrs_means_conv, qt.rad,
                              stimulus_id=paasl_conv.stimulus_id,
                              object_name='Surrogates orientation differences of '+paasl_conv.y_axis_name,
                              sheet_name=sheet,
                              tags=self.tags_ncropped,
                              analysis_algorithm=self.__class__.__name__))
           self.datastore.full_datastore.add_analysis_result(
               SingleObject(orrs_lens_conv, qt.Dimensionless,
                              stimulus_id=paasl_conv.stimulus_id,
                              object_name='Orientation selectivity of '+paasl_conv.y_axis_name,
                              sheet_name=sheet,
                              tags=self.tags_ncropped,
                              analysis_algorithm=self.__class__.__name__))
           self.datastore.full_datastore.add_analysis_result(
               SingleObject(surrs_lens_conv, qt.Dimensionless,
                              stimulus_id=paasl_conv.stimulus_id,
                              object_name='Surrogates orientation selectivity of '+paasl_conv.y_axis_name,
                              sheet_name=sheet,     
                              tags=self.tags_ncropped,      
                              analysis_algorithm=self.__class__.__name__))
           self.datastore.full_datastore.add_analysis_result(
               SingleObject(orrs_means_cropped, qt.rad,
                              stimulus_id=paasl_cropped.stimulus_id,
                              object_name='Orientation differences of '+paasl_cropped.y_axis_name,
                              sheet_name=sheet,
                              tags=self.tags_cropped,
                              analysis_algorithm=self.__class__.__name__))
           self.datastore.full_datastore.add_analysis_result(
               SingleObject(surrs_means_cropped, qt.rad,
                              stimulus_id=paasl_cropped.stimulus_id,
                              object_name='Surrogates orientation differences of '+paasl_cropped.y_axis_name,
                              sheet_name=sheet,
                              tags=self.tags_cropped,
                              analysis_algorithm=self.__class__.__name__))
           self.datastore.full_datastore.add_analysis_result(
               SingleObject(orrs_lens_cropped, qt.Dimensionless,
                              stimulus_id=paasl_cropped.stimulus_id,
                              object_name='Orientation selectivity of '+paasl_cropped.y_axis_name,
                              sheet_name=sheet,
                              tags=self.tags_cropped,
                              analysis_algorithm=self.__class__.__name__))
           self.datastore.full_datastore.add_analysis_result(
               SingleObject(surrs_lens_cropped, qt.Dimensionless,
                              stimulus_id=paasl_cropped.stimulus_id,
                              object_name='Surrogates orientation selectivity of '+paasl_cropped.y_axis_name,
                              sheet_name=sheet,
                              tags=self.tags_cropped,
                              analysis_algorithm=self.__class__.__name__))
           self.datastore.full_datastore.add_analysis_result(
               SingleObject(orrs_means_conv_cropped, qt.rad,
                              stimulus_id=paasl_conv_cropped.stimulus_id,
                              object_name='Orientation differences of '+paasl_conv_cropped.y_axis_name,
                              sheet_name=sheet,
                              tags=self.tags_cropped,
                              analysis_algorithm=self.__class__.__name__))
           self.datastore.full_datastore.add_analysis_result(
               SingleObject(surrs_means_conv_cropped, qt.rad,
                              stimulus_id=paasl_conv_cropped.stimulus_id,
                              object_name='Surrogates orientation differences of '+paasl_conv_cropped.y_axis_name,
                              sheet_name=sheet,
                              tags=self.tags_cropped,
                              analysis_algorithm=self.__class__.__name__))
           self.datastore.full_datastore.add_analysis_result(
               SingleObject(orrs_lens_conv_cropped, qt.Dimensionless,
                              stimulus_id=paasl_conv_cropped.stimulus_id,
                              object_name='Orientation selectivity of '+paasl_conv_cropped.y_axis_name,
                              sheet_name=sheet,
                              tags=self.tags_cropped,
                              analysis_algorithm=self.__class__.__name__))
           self.datastore.full_datastore.add_analysis_result(
               SingleObject(surrs_lens_conv_cropped, qt.Dimensionless,
                              stimulus_id=paasl_conv_cropped.stimulus_id,
                              object_name='Surrogates orientation selectivity of '+paasl_conv_cropped.y_axis_name,
                              sheet_name=sheet,     
                              tags=self.tags_cropped,      
                              analysis_algorithm=self.__class__.__name__))


class AreaOrientationLabeling(Analysis):
    required_parameters = ParameterSet({
    'map_file': str,
    'cropped_length': float,
    })

    def perform_analysis(self):
        f = open(self.parameters.map_file, 'rb')
        or_map = pickle.load(f, encoding="latin1")
        f.close()

        for sheet in self.datastore.sheets():
            paasl = queries.param_filter_query(self.datastore,identifier='PerAreaAnalogSignalList',sheet_name=sheet,ads_unique=True).get_analysis_result()[0]
            sx = eval(self.datastore.full_datastore.block.annotations['sheet_parameters'])[sheet]['sx']
            map_stretch = eval(self.datastore.full_datastore.block.annotations['model_parameters'])['sheets']['l4_cortex_exc']['AfferentConnection']['or_map_stretch']

            cutout_size = int(or_map.shape[0] / map_stretch * self.parameters.cropped_length / sx)

            x,y = or_map.shape[1]/2-cutout_size/2,or_map.shape[0]/2-cutout_size/2
            I = Image.fromarray(or_map*255)
            I = I.crop((x,y,x+cutout_size,y+cutout_size))
            a = numpy.asarray(I)/255
            coords_xa = numpy.arange(a.shape[1]) * self.parameters.cropped_length/(a.shape[1] - 1)  - self.parameters.cropped_length/2
            coords_ya = numpy.arange(a.shape[0]) * self.parameters.cropped_length/(a.shape[0] - 1)  - self.parameters.cropped_length/2
            Ya, Xa = numpy.meshgrid(coords_ya, coords_xa)
            ND = NearestNDInterpolator(list(zip(Xa.flatten(), Ya.flatten())),a.flatten())
            #coords_xb = numpy.linspace(0,a.shape[1]-1,orientation_matrix.shape[1])
            #coords_yb = numpy.linspace(0,a.shape[0]-1,orientation_matrix.shape[0])
            Yb, Xb = numpy.meshgrid(paasl.y_coords, paasl.x_coords)
            b = ND((Yb,Xb)) * numpy.pi

            self.datastore.full_datastore.add_analysis_result(
               PerAreaValue(b,qt.rad,paasl.x_coords,paasl.y_coords,
                              stimulus_id=None,
                              value_name='Orientation label, cropped size = '+str(self.parameters.cropped_length),
                              sheet_name=sheet,
                              tags=self.tags,
                              analysis_algorithm=self.__class__.__name__))

class NauhausAnalysis(Analysis):
    required_parameters = ParameterSet({
    'cropped_length': float,
    'points_distance': float,
    'electrodes_spacing': float,
    'n_neurons': int,
    'duration_analysis': float,
    })



    def perform_analysis(self):
        exp_fit = lambda x,a,b,c: a + b * numpy.exp(x/c)
        errfunc = lambda p, x, y: numpy.linalg.norm(exp_fit_v(x,*p) - y)
        exp_fit_v = numpy.vectorize(exp_fit)


        def exponential_fit(distance,amplitude,n_attempts=10):
            evar_max = 0
            p1 = [0,1,1]

            for _ in range(n_attempts):
                p0 = [numpy.max(amplitude),(numpy.random.rand()-0.5)*20,(numpy.random.rand()-0.5)*10]

                try:
                    try:
                        p1_, success = scipy.optimize.curve_fit(exp_fit_v,distance,amplitude,p0=p0)
                        evar = explained_variance(amplitude,exp_fit_v(distf,*p1_))
                    except ZeroDivisionError:
                        p1_ = [0,1,1]
                        evar = 0
                    if evar > evar_max:
                        p1 = p1_
                        evar_max = evar

                except RuntimeError:
                    return None, 0
            
            residual_amplitudes = amp - exp_fit_v(dist/1000,*p1)

            return residual_amplitudes, evar_max
        

        for sheet in self.datastore.sheets():
            sx = eval(self.datastore.full_datastore.block.annotations['sheet_parameters'])[sheet]['sx']
            magnification = eval(self.datastore.full_datastore.block.annotations['sheet_parameters'])[sheet]['magnification_factor']
            map_stretch = eval(self.datastore.full_datastore.block.annotations['model_parameters'])['sheets']['l4_cortex_exc']['AfferentConnection']['or_map_stretch']

            dsv = queries.param_filter_query(self.datastore, sheet_name=sheet,st_direct_stimulation_name=None,st_name='InternalStimulus')
            seg = queries.param_filter_query(dsv).get_segments()[0]
            spikes_ids = seg.get_stored_spike_train_ids()
            spikes =  seg.get_spiketrain(spikes_ids)
            idx = self.datastore.get_sheet_indexes(sheet_name=sheet, neuron_ids=spikes_ids)

            xs = self.datastore.get_neuron_positions()[sheet][0][idx]
            ys = self.datastore.get_neuron_positions()[sheet][1][idx]

            orientation_values =self.datastore.get_analysis_result(identifier='PerNeuronValue', value_name=['LGNAfferentOrientation', 'ORMapOrientation'], sheet_name=[sheet])[0].get_value_by_id(spikes_ids)
            orientation_matrix = queries.param_filter_query(self.datastore,sheet_name=sheet,identifier='PerAreaValue',value_name='Orientation label, cropped size = '+str(int(sx))).get_analysis_result()[0].values
            paasl = queries.param_filter_query(dsv,identifier='PerAreaAnalogSignalList',y_axis_name='Butterworth band-pass filtered of (LFP) freq=[5,100], order = 4').get_analysis_result()[0]
            x_coords, y_coords = numpy.meshgrid(paasl.x_coords,paasl.y_coords)

            sig = numpy.squeeze(numpy.array(paasl.asl))
            speeds = []
            nx,ny,duration = sig.shape

            iso_amps = []
            mid_amps = []
            ortho_amps = []

            numpy.random.seed(1001)
            idx_chosen = numpy.random.choice(numpy.arange(len(spikes)),self.parameters.n_neurons,replace=False)
            ids_chosen = self.datastore.get_sheet_ids(sheet_name=sheet, indexes=idx_chosen)
            ids_output = []
            ids_speeds = []

            # Computing the coordinates of the electrodes on the grid
            e_points_spacing = int(self.parameters.electrodes_spacing//self.parameters.points_distance)        
            nex = nx//e_points_spacing + 1
            edx = int((nex * e_points_spacing  - nx)/2)
            ney = ny//e_points_spacing + 1
            edy = int((ney * e_points_spacing  - nx)/2)    
            idex_coords, idey_coords = numpy.meshgrid(numpy.arange(edx,nx,e_points_spacing),numpy.arange(edy,ny,e_points_spacing))

            # Looping over `n_neurons` randomly selected neurons
            for i,(spt,x,y,orr,idd) in enumerate(zip(numpy.array(spikes,dtype=object)[idx_chosen],numpy.array(xs)[idx_chosen],numpy.array(ys)[idx_chosen],numpy.array(orientation_values)[idx_chosen],ids_chosen)):
                x_cs = x * magnification
                y_cs = y * magnification

                orr_idx = int(orr * 8 // numpy.pi)    

                # Compute spike-triggered LFP only for seed neurons located in the cropped portion of the model
                if x_cs > - self.parameters.cropped_length/2 and x_cs < self.parameters.cropped_length/2 and y_cs > - self.parameters.cropped_length/2 and y_cs < self.parameters.cropped_length/2:
                    ceiled_spt = numpy.ceil(spt.magnitude).astype(int)
                    ceiled_spt = ceiled_spt[ceiled_spt<duration-self.parameters.duration_analysis]
                    stlfps = []
                    if len(ceiled_spt) >5:
                        for t in ceiled_spt:
                            stlfps.append(sig[edx::e_points_spacing,edy::e_points_spacing,t:t+self.parameters.duration_analysis])
                            
                        stlfps = numpy.array(stlfps)
                        stlfp = numpy.mean(stlfps,axis=0) 
                        idx, idy = int((x_cs+sx/2)/self.parameters.points_distance),int((y_cs+sx/2)/self.parameters.points_distance)
                        t_peak = numpy.argmax(stlfp,axis=2) + 1
                        dist = numpy.sqrt((idey_coords-idy)**2 + (idex_coords-idx)**2) * self.parameters.points_distance
                        speed = dist/t_peak
                        speeds.append(numpy.mean(speed))
                        ids_speeds.append(idd)
                        distf = dist.flatten()/magnification
                        t_peakf = t_peak.flatten()

                        amp = numpy.max(stlfp,axis=2)
                        ampf = amp.flatten()

                        residual_amplitudes, evar = exponential_fit(distf,ampf) 
                        if evar > 0.5:
                            orr_diff = circular_dist(orientation_matrix,orr,numpy.pi)
                            mask_iso = orr_diff < numpy.pi/6
                            mask_mid = numpy.logical_and(orr_diff < numpy.pi/3, numpy.pi/6 < orr_diff)
                            mask_ortho = numpy.pi/3 < orr_diff  

                            iso_amps.append(numpy.mean(residual_amplitudes[mask_iso[edx::e_points_spacing,edy::e_points_spacing]]))
                            mid_amps.append(numpy.mean(residual_amplitudes[mask_mid[edx::e_points_spacing,edy::e_points_spacing]]))
                            ortho_amps.append(numpy.mean(residual_amplitudes[mask_ortho[edx::e_points_spacing,edy::e_points_spacing]]))
                            ids_output.append(idd)

            self.datastore.full_datastore.add_analysis_result(
                PerNeuronValue(iso_amps,ids_output,qt.Dimensionless,
                               stimulus_id=paasl.stimulus_id,
                               value_name='Residual amplitudes iso-oriented spike-triggered LFP',
                               sheet_name=sheet,
                               tags=self.tags,
                               analysis_algorithm=self.__class__.__name__,
                               period=None))
            self.datastore.full_datastore.add_analysis_result(
                PerNeuronValue(mid_amps,ids_output,qt.Dimensionless, 
                               stimulus_id=paasl.stimulus_id,
                               value_name='Residual amplitudes mid-oriented spike-triggered LFP',
                               sheet_name=sheet,
                               tags=self.tags,
                               analysis_algorithm=self.__class__.__name__,
                               period=None))
            self.datastore.full_datastore.add_analysis_result(
                PerNeuronValue(ortho_amps,ids_output,qt.Dimensionless, 
                               stimulus_id=paasl.stimulus_id,
                               value_name='Residual amplitudes orthogonal spike-triggered LFP',
                               sheet_name=sheet,
                               tags=self.tags,
                               analysis_algorithm=self.__class__.__name__,
                               period=None))
            self.datastore.full_datastore.add_analysis_result(
                PerNeuronValue(speeds,ids_speeds,qt.um/qt.ms,
                               stimulus_id=paasl.stimulus_id,
                               value_name='Propagation speed spike-triggered LFP',
                               sheet_name=sheet,
                               tags=self.tags,
                               analysis_algorithm=self.__class__.__name__,
                               period=None))


