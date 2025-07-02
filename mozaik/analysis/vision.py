# -*- coding: utf-8 -*-
"""
Module containing vision specific analysis.
"""
import mozaik
import numpy
import scipy
import quantities as qt
from .analysis import Analysis
from mozaik.tools.mozaik_parametrized import colapse, colapse_to_dictionary, MozaikParametrized
from mozaik.analysis.data_structures import PerNeuronValue
from mozaik.analysis.helper_functions import psth
from parameters import ParameterSet
from mozaik.storage import queries
from mozaik.tools.circ_stat import circ_mean, circular_dist
from mozaik.tools.neo_object_operations import neo_mean, neo_sum
from builtins import zip
from collections import OrderedDict
import pandas

logger = mozaik.getMozaikLogger()

class ModulationRatio(Analysis):
    """
    This analysis calculates the modulation ration (as the F1/F0) for all
    neurons in the data using all available responses recorded to the
    FullfieldDriftingSinusoidalGrating stimuli. This method also requires
    that 'orientation preference' has already been calculated for all the 
    neurons.
    
    The `ModulationRatio` takes all responses recorded to the FullfieldDriftingSinusoidalGrating and
    calculates their PSTH.  Then it collapses this list of PSTHs into groups, each containing PSTH associated
    with the same FullfieldDriftingSinusoidalGrating stimulus with the  exception of the orientation. 
    For each such group it then goes through each recorded neuron and selects the closest 
    presented orientation to the orientation peference of the given neuron, and using the PSTH associated 
    with this selected orientation it calculates the modulation ratio for that neuron. This way for each
    group it will calculate modulation ratios for all recorded neurons, and will store them in datastore
    using the PerNeuronValue data structure.
    """

    def perform_analysis(self):
        for sheet in self.datastore.sheets():
            # Load up spike trains for the right sheet and the corresponding
            # stimuli, and transform spike trains into psth
            dsv = queries.param_filter_query(self.datastore,identifier='AnalogSignalList',sheet_name=sheet,analysis_algorithm='PSTH',st_name='FullfieldDriftingSinusoidalGrating')
            assert queries.equal_ads(dsv,except_params=['stimulus_id']) , "It seems PSTH computed in different ways are present in datastore, ModulationRatio can accept only one"
            psths = dsv.get_analysis_result()
            st = [MozaikParametrized.idd(p.stimulus_id) for p in psths]
            # average across trials
            psths, stids = colapse(psths,st,parameter_list=['trial'],func=neo_mean,allow_non_identical_objects=True)

            # retrieve the computed orientation preferences
            pnvs = self.datastore.get_analysis_result(identifier='PerNeuronValue',
                                                      sheet_name=sheet,
                                                      value_name='orientation preference')
            if len(pnvs) != 1:
                logger.error("ERROR: Expected only one PerNeuronValue per sheet "
                             "with value_name 'orientation preference' in datastore, got: "
                             + str(len(pnvs)))
                return None
        
            or_pref = pnvs[0]
            # find closest orientation of grating to a given orientation preference of a neuron
            # first find all the different presented stimuli:
            ps = OrderedDict()
            for s in st:
                ps[MozaikParametrized.idd(s).orientation] = True
            ps = list(ps.keys())

            # now find the closest presented orientations
            closest_presented_orientation = []
            for i in range(0, len(or_pref.values)):
                circ_d = 100000
                idx = 0
                for j in range(0, len(ps)):
                    if circ_d > circular_dist(or_pref.values[i], ps[j], numpy.pi):
                        circ_d = circular_dist(or_pref.values[i], ps[j], numpy.pi)
                        idx = j
                closest_presented_orientation.append(ps[idx])

            closest_presented_orientation = numpy.array(closest_presented_orientation)

            # collapse along orientation - we will calculate MR for each
            # parameter combination other than orientation
            d = colapse_to_dictionary(psths, stids, "orientation")
            for (st, vl) in d.items():
                # here we will store the modulation ratios, one per each neuron
                modulation_ratio = []
                f0 = []
                f1 = []
                f1_phases = []
                ids = []
                frequency = MozaikParametrized.idd(st).temporal_frequency * MozaikParametrized.idd(st).getParams()['temporal_frequency'].units
                for (orr, ppsth) in zip(vl[0], vl[1]):
                    for j in numpy.nonzero(orr == closest_presented_orientation)[0]:
                        if or_pref.ids[j] in ppsth.ids:
                            a = or_pref.ids[j]
                            mr,F0,F1,F1_phase = self._calculate_MR(ppsth.get_asl_by_id(or_pref.ids[j]).flatten(),frequency)
                            modulation_ratio.append(mr)
                            f0.append(F0)
                            f1.append(F1)
                            f1_phases.append(F1_phase)
                            ids.append(or_pref.ids[j])
                            
                logger.debug('Adding PerNeuronValue:' + str(sheet))
                self.datastore.full_datastore.add_analysis_result(
                    PerNeuronValue(modulation_ratio,
                                   ids,
                                   qt.dimensionless,
                                   value_name='Modulation ratio' + '(' + psths[0].x_axis_name + ')',
                                   sheet_name=sheet,
                                   tags=self.tags,
                                   period=None,
                                   analysis_algorithm=self.__class__.__name__,
                                   stimulus_id=str(st)))

                self.datastore.full_datastore.add_analysis_result(
                    PerNeuronValue(f0,
                                   ids,
                                   qt.dimensionless,
                                   value_name='F0' + '(' + psths[0].x_axis_name + ')',
                                   sheet_name=sheet,
                                   tags=self.tags,
                                   period=None,
                                   analysis_algorithm=self.__class__.__name__,
                                   stimulus_id=str(st)))
                
                self.datastore.full_datastore.add_analysis_result(
                    PerNeuronValue(f1,
                                   ids,
                                   qt.dimensionless,
                                   value_name='F1' + '(' + psths[0].x_axis_name + ')',
                                   sheet_name=sheet,
                                   tags=self.tags,
                                   period=None,
                                   analysis_algorithm=self.__class__.__name__,
                                   stimulus_id=str(st)))

                self.datastore.full_datastore.add_analysis_result(
                    PerNeuronValue(f1_phases,
                                   ids,
                                   qt.rad,
                                   value_name='F1 phase' + '(' + psths[0].x_axis_name + ')',
                                   sheet_name=sheet,
                                   tags=self.tags,
                                   period=2*numpy.pi,
                                   analysis_algorithm=self.__class__.__name__,
                                   stimulus_id=str(st)))

                #import pylab
                #pylab.figure()
                #pylab.hist(modulation_ratio)

    def _calculate_MR(self,signal, frequency):
        """
        Calculates MR at frequency 1/period for each of the signals in the signal_list

        Returns an array of MRs on per each signal in signal_list
        """
        duration = signal.t_stop - signal.t_start
        period = 1/frequency
        period = period.rescale(signal.t_start.units)
        cycles = duration / period
        first_har = int(round(float(cycles.magnitude)))

        fft = numpy.fft.fft(signal)

        if abs(fft[0]) != 0:
            return 2*abs(fft[first_har])/abs(fft[0]),abs(fft[0])/len(signal),2*abs(fft[first_har])/len(signal),numpy.angle(fft[first_har])
        else:
            logger.info("MR: ARGH: " + str(fft[0]) +"  " +  str(numpy.mean(signal)))
            return 10,abs(fft[0])/len(signal),2*abs(fft[first_har])/len(signal),numpy.angle(fft[first_har])

class ModulationRatioSpecificOrientation(Analysis):
    """
    This analysis is similar to the ModulationRatio analysis, but instead
    of retrieving the measured orientation preference of the neurons, it
    assumes all neurons have an orientation selectivity similar to the
    `orientation` parameter.
    It takes all PSTHs contained in the DSV, that must be associated to a single type of stimulus 
    that doesn't necessarily have to be a FullfieldSinusoidalDriftingGrating. Then it collapses this list of PSTHs
    into groups, each containing PSTH associated with the same stimulus with the 
    exception of the orientation. For each such group it then goes through each recorded neuron 
    and selects the closest presented orientation to the orientation parameter, and using the PSTH associated
    with this selected orientation it calculates the modulation ratio for that neuron. This way for each
    group it will calculate modulation ratios for all recorded neurons, and will store them in datastore
    using the PerNeuronValue data structure.
    """
    required_parameters = ParameterSet({
        'orientation' : float,
    })

    def perform_analysis(self):
        for sheet in self.datastore.sheets():
            # Load up spike trains for the right sheet and the corresponding
            # stimuli, and transform spike trains into psth
            dsv = queries.param_filter_query(self.datastore,identifier='AnalogSignalList',sheet_name=sheet,analysis_algorithm='PSTH',st_orientation=self.parameters.orientation)
            assert queries.equal_ads(dsv,except_params=['stimulus_id']) , "It seems PSTH computed in different ways are present in datastore, ModulationRatio can accept only one"
            psths = dsv.get_analysis_result()
            st = [MozaikParametrized.idd(p.stimulus_id) for p in psths]
            assert queries.ads_with_equal_stimuli(dsv,params=['orientation']), "The stimuli in the DSV should have the same properties"
            psths = dsv.get_analysis_result()
            st = [MozaikParametrized.idd(p.stimulus_id) for p in psths]
            # average across trials
            psths, stids = colapse(psths,st,parameter_list=['trial'],func=neo_mean,allow_non_identical_objects=True)
            for (psth, stid) in zip(psths, stids):
                # here we will store the modulation ratios, one per each neuron
                modulation_ratio = []
                f0 = []
                f1 = []
                f1_phases = []
                frequency = MozaikParametrized.idd(stid).temporal_frequency * MozaikParametrized.idd(stid).getParams()['temporal_frequency'].units
                ids = psth.ids
                for idd in ids:
                    mr,F0,F1,F1_phase = self._calculate_MR(psth.get_asl_by_id(idd).flatten(),frequency)
                    modulation_ratio.append(mr)
                    f0.append(F0)
                    f1.append(F1)
                    f1_phases.append(F1_phase)

                logger.debug('Adding PerNeuronValue:' + str(sheet))
                self.datastore.full_datastore.add_analysis_result(
                    PerNeuronValue(modulation_ratio,
                                   ids,
                                   qt.dimensionless,
                                   value_name='Modulation ratio orientation '+ str(self.parameters.orientation) + '(' + psths[0].x_axis_name + ')',
                                   sheet_name=sheet,
                                   tags=self.tags,
                                   period=None,
                                   analysis_algorithm=self.__class__.__name__,
                                   stimulus_id=str(stid)))

                self.datastore.full_datastore.add_analysis_result(
                    PerNeuronValue(f0,
                                   ids,
                                   qt.dimensionless,
                                   value_name='F0 orientation '+ str(self.parameters.orientation) + '(' + psths[0].x_axis_name + ')',
                                   sheet_name=sheet,
                                   tags=self.tags,
                                   period=None,
                                   analysis_algorithm=self.__class__.__name__,
                                   stimulus_id=str(stid)))

                self.datastore.full_datastore.add_analysis_result(
                    PerNeuronValue(f1,
                                   ids,
                                   qt.dimensionless,
                                   value_name='F1 orientation '+ str(self.parameters.orientation) + '(' + psths[0].x_axis_name + ')',
                                   sheet_name=sheet,
                                   tags=self.tags,
                                   period=None,
                                   analysis_algorithm=self.__class__.__name__,
                                   stimulus_id=str(stid)))

                self.datastore.full_datastore.add_analysis_result(
                    PerNeuronValue(f1_phases,
                                   ids,
                                   qt.rad,
                                   value_name='F1 phase orientation '+ str(self.parameters.orientation) + '(' + psths[0].x_axis_name + ')',
                                   sheet_name=sheet,
                                   tags=self.tags,
                                   period=2*numpy.pi,
                                   analysis_algorithm=self.__class__.__name__,
                                   stimulus_id=str(stid)))


    def _calculate_MR(self,signal, frequency):
        """
        Calculates MR at frequency 1/period for each of the signals in the signal_list

        Returns an array of MRs on per each signal in signal_list
        """
        duration = signal.t_stop - signal.t_start
        period = 1/frequency
        period = period.rescale(signal.t_start.units)
        cycles = duration / period
        first_har = int(round(float(cycles.magnitude)))

        fft = numpy.fft.fft(signal)

        if abs(fft[0]) != 0:
            return 2*abs(fft[first_har])/abs(fft[0]),abs(fft[0])/len(signal),2*abs(fft[first_har])/len(signal),numpy.angle(fft[first_har])
        else:
            logger.info("MR: ARGH: " + str(fft[0]) +"  " +  str(numpy.mean(signal)))
            return 10,abs(fft[0])/len(signal),2*abs(fft[first_har])/len(signal),numpy.angle(fft[first_har])

class Analog_F0andF1(Analysis):
      """
      Calculates the DC and first harmonic of trial averaged vm and conductancesfor each neuron.

      It also calculates the DC and F1 to any AnalogSignalList present in the datastore (not trial averaged this time).

      The data_store has to contain responses or AnalogSignalLists to the same stimulus type, and the stymulus type has to have
      <temporal_frequency> parameter which is used as the first harmonic frequency.
      
      It stores them in PerNeuronValue datastructures (one for exc. one for inh. conductances, and one for each AnalogSignalList).
      
      Notes
      -----
      
      Only neurons for which the corresponding signals were measured will be included in the PerNeuronValue data structures.

      """

      def perform_analysis(self):
            
            for sheet in self.datastore.sheets():
                dsv = queries.param_filter_query(self.datastore, sheet_name=sheet)
                if len(dsv.get_segments()) != 0:
                  assert queries.equal_stimulus_type(self.datastore) , "Data store has to contain only recordings to the same stimulus type"
                  st = self.datastore.get_stimuli()[0]
                  assert 'temporal_frequency' in MozaikParametrized.idd(st).getParams(), "The stimulus has to have parameter temporal_frequency which is used as first harmonic"

                  segs1, stids = colapse(dsv.get_segments(),dsv.get_stimuli(),parameter_list=['trial'],allow_non_identical_objects=True)
                  for segs,st in zip(segs1, stids):
                      first_analog_signal = segs[0].get_esyn(segs[0].get_stored_esyn_ids()[0])
                      duration = first_analog_signal.t_stop - first_analog_signal.t_start
                      frequency = MozaikParametrized.idd(st).temporal_frequency * MozaikParametrized.idd(st).getParams()['temporal_frequency'].units
                      period = 1/frequency
                      period = period.rescale(first_analog_signal.t_start.units)
                      cycles = duration / period
                      first_har = int(round(cycles))
                      
                      e_f0 = [abs(numpy.fft.fft(numpy.mean([seg.get_esyn(idd) for seg in segs],axis=0).flatten())[0]/len(segs[0].get_esyn(idd))) for idd in segs[0].get_stored_esyn_ids()]
                      i_f0 = [abs(numpy.fft.fft(numpy.mean([seg.get_isyn(idd) for seg in segs],axis=0).flatten())[0]/len(segs[0].get_isyn(idd))) for idd in segs[0].get_stored_isyn_ids()]
                      v_f0 = [abs(numpy.fft.fft(numpy.mean([seg.get_vm(idd) for seg in segs],axis=0).flatten())[0]/len(segs[0].get_vm(idd))) for idd in segs[0].get_stored_vm_ids()]
                      e_f1 = [2*abs(numpy.fft.fft(numpy.mean([seg.get_esyn(idd) for seg in segs],axis=0).flatten())[first_har]/len(segs[0].get_esyn(idd))) for idd in segs[0].get_stored_esyn_ids()]
                      i_f1 = [2*abs(numpy.fft.fft(numpy.mean([seg.get_isyn(idd) for seg in segs],axis=0).flatten())[first_har]/len(segs[0].get_isyn(idd))) for idd in segs[0].get_stored_isyn_ids()]
                      v_f1 = [2*abs(numpy.fft.fft(numpy.mean([seg.get_vm(idd) for seg in segs],axis=0).flatten())[first_har]/len(segs[0].get_vm(idd))) for idd in segs[0].get_stored_vm_ids()]
                      
                      cond_units = segs[0].get_esyn(segs[0].get_stored_esyn_ids()[0]).units
                      vm_units = segs[0].get_vm(segs[0].get_stored_esyn_ids()[0]).units
                      
                      self.datastore.full_datastore.add_analysis_result(PerNeuronValue(e_f0,segs[0].get_stored_esyn_ids(),cond_units,value_name = 'F0_Exc_Cond',sheet_name=sheet,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(st)))        
                      self.datastore.full_datastore.add_analysis_result(PerNeuronValue(i_f0,segs[0].get_stored_isyn_ids(),cond_units,value_name = 'F0_Inh_Cond',sheet_name=sheet,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(st)))        
                      self.datastore.full_datastore.add_analysis_result(PerNeuronValue(v_f0,segs[0].get_stored_vm_ids(),vm_units,value_name = 'F0_Vm',sheet_name=sheet,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(st)))        
                      self.datastore.full_datastore.add_analysis_result(PerNeuronValue(e_f1,segs[0].get_stored_esyn_ids(),cond_units,value_name = 'F1_Exc_Cond',sheet_name=sheet,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(st)))        
                      self.datastore.full_datastore.add_analysis_result(PerNeuronValue(i_f1,segs[0].get_stored_isyn_ids(),cond_units,value_name = 'F1_Inh_Cond',sheet_name=sheet,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(st)))        
                      self.datastore.full_datastore.add_analysis_result(PerNeuronValue(v_f1,segs[0].get_stored_vm_ids(),vm_units,value_name = 'F1_Vm',sheet_name=sheet,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(st)))        

                # AnalogSignalList part 
                dsv = queries.param_filter_query(dsv, sheet_name=sheet,name='AnalogSignalList')
                for asl in dsv.get_analysis_result():
                    assert 'temporal_frequency' in MozaikParametrized.idd(asl.stimulus_id).getParams(), "The stimulus has to have parameter temporal_frequency which is used as first harmonic"

                    signals = asl.asl
                    first_analog_signal = signals[0]
                    duration = first_analog_signal.t_stop - first_analog_signal.t_start
                    frequency = MozaikParametrized.idd(asl.stimulus_id).temporal_frequency * MozaikParametrized.idd(asl.stimulus_id).getParams()['temporal_frequency'].units
                    period = 1/frequency
                    period = period.rescale(first_analog_signal.t_start.units)
                    cycles = duration / period
                    first_har = int(round(cycles))

                    f0 = [abs(numpy.fft.fft(signal.flatten())[0])/len(signal) for signal in signals]
                    f1 = [2*abs(numpy.fft.fft(signal.flatten())[first_har])/len(signal) for signal in signals]
                    f1_phase = [numpy.angle(numpy.fft.fft(signal.flatten())[first_har]) for signal in signals]

                    self.datastore.full_datastore.add_analysis_result(PerNeuronValue(f0,asl.ids,asl.y_axis_units,value_name = 'F0('+ asl.y_axis_name + ')',sheet_name=sheet,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=asl.stimulus_id))                            
                    self.datastore.full_datastore.add_analysis_result(PerNeuronValue(f1,asl.ids,asl.y_axis_units,value_name = 'F1('+ asl.y_axis_name + ')',sheet_name=sheet,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=asl.stimulus_id))                                                
                    self.datastore.full_datastore.add_analysis_result(PerNeuronValue(f1_phase,asl.ids,asl.y_axis_units,value_name = 'F1 phase('+ asl.y_axis_name + ')',sheet_name=sheet,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=asl.stimulus_id))

class LocalHomogeneityIndex(Analysis):      
    """
    Calculates Local Homogeneity Index (LHI) for each neuron, of each PerNeuronValue that is present in the datastore, according to:
    Nauhaus, I., Benucci, A., Carandini, M., & Ringach, D. (2008). Neuronal selectivity and local map structure in visual cortex. Neuron, 57(5), 673–679. 
    """
    required_parameters = ParameterSet({
        'sigma' : float,
    }) 
    def perform_analysis(self):
        sigma = self.parameters.sigma
        for sheet in self.datastore.sheets():
            positions = self.datastore.get_neuron_positions()[sheet]
            for pnv in queries.param_filter_query(self.datastore,sheet_name=sheet,identifier='PerNeuronValue').get_analysis_result():
                lhis = []
                for x in pnv.ids:
                    idx = self.datastore.get_sheet_indexes(sheet,x)
                    sx = positions[0][idx]
                    sy = positions[1][idx]
                    lhi_current=[0,0]
                    for y in pnv.ids:
                        idx = self.datastore.get_sheet_indexes(sheet,y)
                        tx = positions[0][idx]
                        ty = positions[1][idx]
                        lhi_current[0]+=numpy.exp(-((sx-tx)*(sx-tx)+(sy-ty)*(sy-ty))/(2*sigma*sigma))*numpy.cos(2*pnv.get_value_by_id(y))
                        lhi_current[1]+=numpy.exp(-((sx-tx)*(sx-tx)+(sy-ty)*(sy-ty))/(2*sigma*sigma))*numpy.sin(2*pnv.get_value_by_id(y))
                    lhis.append(numpy.sqrt(lhi_current[0]*lhi_current[0] + lhi_current[1]*lhi_current[1])/(2*numpy.pi*sigma*sigma))
                
                self.datastore.full_datastore.add_analysis_result(
                    PerNeuronValue(lhis,
                                   pnv.ids,
                                   qt.dimensionless,
                                   value_name='LocalHomogeneityIndex' + '(' + str(self.parameters.sigma) + ':' + pnv.value_name + ')',
                                   sheet_name=sheet,
                                   tags=self.tags,
                                   period=None,
                                   analysis_algorithm=self.__class__.__name__))

class SizeTuningAnalysis(Analysis):
      """
      Calculates the size tuning properties   
      """   
        
      required_parameters = ParameterSet({
          'neurons': list,  # list of neurons for which to compute this (normally this analysis will only makes sense for neurons for which the sine grating disk stimulus has been optimally oriented)
          'sheet_name' : str
      })      
        
      
      def perform_analysis(self):
                dsv = queries.param_filter_query(self.datastore,identifier='PerNeuronValue',sheet_name=self.parameters.sheet_name,st_name='DriftingSinusoidalGratingDisk')
                
                if len(dsv.get_analysis_result()) == 0: return
                assert queries.ads_with_equal_stimulus_type(dsv)
                assert queries.equal_ads(dsv,except_params=['stimulus_id'])
                self.pnvs = dsv.get_analysis_result()
                units_fr = self.pnvs[0].value_units
                
                # get stimuli
                self.st = [MozaikParametrized.idd(s.stimulus_id) for s in self.pnvs]
                
                
                # transform the pnvs into a dictionary of tuning curves according along the 'radius' parameter
                # also make sure they are ordered according to the first pnv's idds 
                
                self.tc_dict = colapse_to_dictionary([z.get_value_by_id(self.parameters.neurons) for z in self.pnvs],self.st,"radius")
                for k in self.tc_dict.keys():
                        crf_sizes = []
                        supp_sizes= []
                        sis = []
                        max_responses=[]
                        csis = []
                        
                        # we will do the calculation neuron by neuron
                        for i in range(0,len(self.parameters.neurons)):
                            
                            rads = self.tc_dict[k][0]
                            values = numpy.array([a[i] for a in self.tc_dict[k][1]])
                            
                            # sort them based on radiuses
                            rads , values = zip(*sorted(zip(rads,values)))
                                                        
                            max_response = numpy.max(values)
                            crf_index  = numpy.argmax(values)
                            crf_size = rads[crf_index]
                            
                            if crf_index < len(values)-1:
                                supp_index = crf_index+numpy.argmin(values[crf_index+1:])+1
                            else:
                                supp_index = len(values)-1
                            supp_size = rads[supp_index]                                

                            if supp_index < len(values)-1:
                                cs_index = supp_index+numpy.argmax(values[supp_index+1:])+1
                            else:
                                cs_index = len(values)-1

                            
                            if values[crf_index] != 0:
                                si = (values[crf_index]-values[supp_index])/values[crf_index]
                            else:
                                si = 0

                            if values[cs_index] != 0:
                                csi = (values[cs_index]-values[supp_index])/values[crf_index]
                            else:
                                csi = 0

                            crf_sizes.append(crf_size)
                            supp_sizes.append(supp_size)
                            sis.append(si)
                            max_responses.append(max_response)
                            csis.append(csi)
                            
                            
                        self.datastore.full_datastore.add_analysis_result(PerNeuronValue(max_responses,self.parameters.neurons,units_fr,value_name = 'Max. response of ' + self.pnvs[0].value_name ,sheet_name=self.parameters.sheet_name,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(k)))
                        self.datastore.full_datastore.add_analysis_result(PerNeuronValue(crf_sizes,self.parameters.neurons,self.st[0].getParams()["radius"].units,value_name = 'Max. facilitation radius of ' + self.pnvs[0].value_name ,sheet_name=self.parameters.sheet_name,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(k)))
                        self.datastore.full_datastore.add_analysis_result(PerNeuronValue(supp_sizes,self.parameters.neurons,self.st[0].getParams()["radius"].units,value_name = 'Max. suppressive radius of ' + self.pnvs[0].value_name ,sheet_name=self.parameters.sheet_name,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(k)))
                        self.datastore.full_datastore.add_analysis_result(PerNeuronValue(sis,self.parameters.neurons,None,value_name = 'Suppression index of ' + self.pnvs[0].value_name ,sheet_name=self.parameters.sheet_name,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(k)))
                        self.datastore.full_datastore.add_analysis_result(PerNeuronValue(csis,self.parameters.neurons,None,value_name = 'Counter-suppression index of ' + self.pnvs[0].value_name ,sheet_name=self.parameters.sheet_name,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(k)))
                        
class SizeTuningAnalysisFit(Analysis):
      """
      Calculates the size tuning properties   
      """

      required_parameters = ParameterSet({
          'neurons': list,  # list of neurons for which to compute this (normally this analysis will only makes sense for neurons for which the sine grating disk stimulus has been optimally oriented)
          'sheet_name' : str
      })

      def _fitgaussian(self,X,Y):
          from scipy.special import erf

          fitfunc = lambda p,x:  p[0]*erf(x/p[1])**2 - p[2] *erf(x/(p[1] + p[3]))**2 + p[4] *erf(x/(p[1]+ p[3]+p[5]))**2 + p[6]
          fitfunc_st = lambda p,x:  p[0]*erf(x/p[1])**2 - p[2] *erf(x/(p[1] + p[3]))**2 + p[4]
          errfunc = lambda p, x, y: numpy.linalg.norm(fitfunc(p,x) - y) # Distance to the target function
          errfunc_st = lambda p, x, y: numpy.linalg.norm(fitfunc_st(p,x) - y) # Distance to the target function

          err = []
          res = []
          p0 = [8.0, 0.43, 8.0, 0.18, 3.0 ,1.4,numpy.min(Y)] # Initial guess for the parameters

          err_st = []
          res_st = []
          p0_st = [8.0, 0.43, 8.0, 0.18,numpy.min(Y)] # Initial guess for the parameters

          for i in range(2,30):
           for j in range(5,22):
              p0_st[1] = i/30.0
              p0_st[3] = j/20.0
              r = scipy.optimize.fmin_tnc(errfunc_st, numpy.array(p0_st), args=(numpy.array(X),numpy.array(Y)),disp=0,bounds=[(0,None),(0,None),(0,None),(0,None),(0,None)],approx_grad=True)
              res_st.append(r)
              err_st.append(errfunc_st(r[0],numpy.array(X),numpy.array(Y)))

          res_st=res_st[numpy.argmin(err_st)]
          p0[0:4] = res_st[0][0:-1]
          p0[-1] = res_st[0][-1]
          res = []
          for j in range(5,33):
            for k in range(1,15):
                p0[3] = j/30.0
                p0[5] = k/6.0
                r = scipy.optimize.fmin_tnc(errfunc, numpy.array(p0), args=(numpy.array(X),numpy.array(Y)),disp=0,bounds=[(p0[0]*9/10,p0[0]*10/9),(p0[1]*9/10,p0[1]*10/9),(0,None),(0,None),(0,None),(0,None),(0,None)],approx_grad=True)
                res.append(r)
                err.append(errfunc(r[0],numpy.array(X),numpy.array(Y)))

          x = numpy.linspace(0,X[-1],100)
          res=res[numpy.argmin(err)]
          print(f'{Y} {res} {p0} 222222222', flush=True)
          if numpy.linalg.norm(Y-numpy.mean(Y),2) != 0:
                err = numpy.linalg.norm(fitfunc(res[0],X)-Y,2)/numpy.linalg.norm(Y-numpy.mean(Y),2)
          else:
                err = 0
          return fitfunc(res[0],x), err

      def perform_analysis(self):
                dsv = queries.param_filter_query(self.datastore,identifier='PerNeuronValue',sheet_name=self.parameters.sheet_name,st_name='DriftingSinusoidalGratingDisk',analysis_algorithm='TrialAveragedFiringRate', value_name='Firing rate')

                if len(dsv.get_analysis_result()) == 0: return
                assert queries.ads_with_equal_stimulus_type(dsv)
                assert queries.equal_ads(dsv,except_params=['stimulus_id'])
                self.pnvs = dsv.get_analysis_result()

                radii = list(set([MozaikParametrized.idd(s).radius for s in queries.param_filter_query(self.datastore, st_name='DriftingSinusoidalGratingDisk').get_stimuli()]))
                max_radii = max(radii)
                units_fr = self.pnvs[0].value_units


                # get stimuli
                self.st = [MozaikParametrized.idd(s.stimulus_id) for s in self.pnvs]


                # transform the pnvs into a dictionary of tuning curves according along the 'radius' parameter
                # also make sure they are ordered according to the first pnv's idds 

                self.tc_dict = colapse_to_dictionary([z.get_value_by_id(self.parameters.neurons) for z in self.pnvs],self.st,"radius")
                for k in self.tc_dict.keys():
                        crf_sizes = []
                        supp_sizes= []
                        sis = []
                        max_responses=[]
                        csis = []
                        errs = []

                        # we will do the calculation neuron by neuron
                        for i in range(0,len(self.parameters.neurons)):

                            rads = self.tc_dict[k][0]
                            values = numpy.array([a[i] for a in self.tc_dict[k][1]])

                            # sort them based on radiuses
                            rads , values = zip(*sorted(zip(rads,values)))

                            values, err = self._fitgaussian(rads,values)
                            errs.append(err)

                            rads = numpy.linspace(0,max_radii,100)

                            max_response = numpy.max(values)
                            crf_index  = numpy.argmax(values)
                            crf_size = rads[crf_index]

                            if crf_index < len(values)-1:
                                supp_index = crf_index+numpy.argmin(values[crf_index+1:])+1
                            else:
                                supp_index = len(values)-1
                            supp_size = rads[supp_index]

                            if supp_index < len(values)-1:
                                cs_index = supp_index+numpy.argmax(values[supp_index+1:])+1
                            else:
                                cs_index = len(values)-1


                            if values[crf_index] != 0:
                                si = (values[crf_index]-values[supp_index])/values[crf_index]
                            else:
                                si = 0

                            if values[cs_index] != 0:
                                csi = (values[cs_index]-values[supp_index])/values[crf_index]
                            else:
                                csi = 0

                            crf_sizes.append(crf_size)
                            supp_sizes.append(supp_size)
                            sis.append(si)
                            max_responses.append(max_response)
                            csis.append(csi)


                        self.datastore.full_datastore.add_analysis_result(PerNeuronValue(max_responses,self.parameters.neurons,units_fr,value_name = 'Max. response of ' + self.pnvs[0].value_name ,sheet_name=self.parameters.sheet_name,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(k)))
                        self.datastore.full_datastore.add_analysis_result(PerNeuronValue(crf_sizes,self.parameters.neurons,self.st[0].getParams()["radius"].units,value_name = 'Max. facilitation radius of ' + self.pnvs[0].value_name ,sheet_name=self.parameters.sheet_name,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(k)))
                        self.datastore.full_datastore.add_analysis_result(PerNeuronValue(supp_sizes,self.parameters.neurons,self.st[0].getParams()["radius"].units,value_name = 'Max. suppressive radius of ' + self.pnvs[0].value_name ,sheet_name=self.parameters.sheet_name,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(k)))
                        self.datastore.full_datastore.add_analysis_result(PerNeuronValue(sis,self.parameters.neurons,None,value_name = 'Suppression index of ' + self.pnvs[0].value_name ,sheet_name=self.parameters.sheet_name,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(k)))
                        self.datastore.full_datastore.add_analysis_result(PerNeuronValue(csis,self.parameters.neurons,None,value_name = 'Counter-suppression index of ' + self.pnvs[0].value_name ,sheet_name=self.parameters.sheet_name,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(k)))
                        self.datastore.full_datastore.add_analysis_result(PerNeuronValue(errs,self.parameters.neurons,None,value_name = 'Size tuning error of fitting of ' + self.pnvs[0].value_name ,sheet_name=self.parameters.sheet_name,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(k)))

class SizeTuningRingAnalysisFit(Analysis):
      """
      Calculates the size tuning properties
      """

      required_parameters = ParameterSet({
          'neurons': list,  # list of neurons for which to compute this (normally this analysis will only makes sense for neurons for which the sine grating disk stimulus has been optimally oriented)
          'sheet_name' : str
      })

      def _fitgaussian(self,X,Y,outer_radius):
          from scipy.integrate import quad
          erf = lambda z,sig: numpy.exp(-(z/sig)**2)
            
          def fitfunc(p,xs):
                return numpy.array([p[0]*quad(erf,x,outer_radius,args=(p[1]))[0]**2/(1+p[2] *quad(erf,x,outer_radius,args=(p[3]))[0]**2) + p[4] for x in xs])
          
          errfunc = lambda p, x, y: numpy.linalg.norm(fitfunc(p,x) - y) # Distance to the target function

          err = []
          res = []
          p0 = [8.0, 0.43, 8.0, 0.5,numpy.min(Y)] # Initial guess for the parameters

          for i in range(2,30):
           for j in range(5,22):
              p0[1] = i/30.0
              p0[3] = j/20.0
              r = scipy.optimize.fmin_tnc(errfunc, numpy.array(p0), args=(numpy.array(X),numpy.array(Y)),disp=0,bounds=[(0,None),(0,None),(0,None),(0,None),(0,None)],approx_grad=True)
              res.append(r)
              err.append(errfunc(r[0],numpy.array(X),numpy.array(Y)))

          x = numpy.linspace(0,X[-1],100)
          res=res[numpy.argmin(err)]
          if numpy.linalg.norm(Y-numpy.mean(Y),2) != 0:
                err = numpy.linalg.norm(fitfunc(res[0],X)-Y,2)/numpy.linalg.norm(Y-numpy.mean(Y),2)
          else:
                err = 0
          print('err: '+ str(err*100), flush=True)
          return fitfunc(res[0],x), err

      def size_tuning_measures_ring(self,rads,values):
          max_response = numpy.max(values)
          peak_index  = numpy.argmax(values[:-1]-values[1:] > 0)
          peak_size = 2* rads[peak_index]
          if values[peak_index] != 0:
              csi = (values[peak_index]-values[0])/values[peak_index]
          else:
              csi = 0
          return [max_response,peak_size,csi]


      def perform_analysis(self):
          dsv = queries.param_filter_query(self.datastore,identifier='PerNeuronValue',sheet_name=self.parameters.sheet_name,st_name='DriftingSinusoidalGratingRing',analysis_algorithm='TrialAveragedFiringRate', value_name='Firing rate')

          if len(dsv.get_analysis_result()) == 0: return
          assert queries.ads_with_equal_stimulus_type(dsv)
          assert queries.equal_ads(dsv,except_params=['stimulus_id'])
          self.pnvs = dsv.get_analysis_result()

          radii = list(set([MozaikParametrized.idd(s).inner_aperture_radius for s in queries.param_filter_query(self.datastore, st_name='DriftingSinusoidalGratingRing').get_stimuli()]))
          max_radius = max(radii)
          units_fr = self.pnvs[0].value_units

          # get stimuli
          self.st = [MozaikParametrized.idd(s.stimulus_id) for s in self.pnvs]

          self.tc_dict = colapse_to_dictionary([z.get_value_by_id(self.parameters.neurons) for z in self.pnvs],self.st,"inner_aperture_radius")
          for k in self.tc_dict.keys():
              peak_sizes = []
              csis = []
              errs = []
              max_responses = []

                # we will do the calculation neuron by neuron
              for i in range(0,len(self.parameters.neurons)):

                  rads = self.tc_dict[k][0]
                  values = numpy.array([a[i] for a in self.tc_dict[k][1]])

                  # sort them based on radiuses
                  rads , values = zip(*sorted(zip(rads,values)))

                  values, err = self._fitgaussian(rads,values,max_radius)
                  errs.append(err)

                  rads = numpy.linspace(0,max_radius,100)

                  max_response, peak_size, csi = self.size_tuning_measures_ring(rads,values)
                  peak_sizes.append(peak_size)
                  max_responses.append(max_response)
                  csis.append(csi)


              self.datastore.full_datastore.add_analysis_result(PerNeuronValue(max_responses,self.parameters.neurons,units_fr,value_name = 'Max. response of ' + self.pnvs[0].value_name ,sheet_name=self.parameters.sheet_name,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(k)))
              self.datastore.full_datastore.add_analysis_result(PerNeuronValue(peak_sizes,self.parameters.neurons,self.st[0].getParams()["inner_aperture_radius"].units,value_name = 'Max. response radius of ' + self.pnvs[0].value_name ,sheet_name=self.parameters.sheet_name,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(k)))
              self.datastore.full_datastore.add_analysis_result(PerNeuronValue(csis,self.parameters.neurons,None,value_name = 'Center-suppression index of ' + self.pnvs[0].value_name ,sheet_name=self.parameters.sheet_name,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(k)))
              self.datastore.full_datastore.add_analysis_result(PerNeuronValue(errs,self.parameters.neurons,None,value_name = 'Size tuning error of fitting of ' + self.pnvs[0].value_name ,sheet_name=self.parameters.sheet_name,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(k)))

class OCTCTuningAnalysis(Analysis):
      """
      Calculates the Orientation Contrast tuning properties.
      """   
        
      required_parameters = ParameterSet({
          'neurons': list,  # list of neurons for which to compute this (normally this analysis will only makes sense for neurons for which the sine grating disk stimulus has been optimally oriented)
          'sheet_name' : str
      })      
        
      
      def perform_analysis(self):
                dsv = queries.param_filter_query(self.datastore,identifier='PerNeuronValue',sheet_name=self.parameters.sheet_name,st_name='DriftingSinusoidalGratingCenterSurroundStimulus')
                
                if len(dsv.get_analysis_result()) == 0: return
                assert queries.ads_with_equal_stimulus_type(dsv)
                assert queries.equal_ads(dsv,except_params=['stimulus_id'])
                self.pnvs = dsv.get_analysis_result()
                
                # get stimuli
                self.st = [MozaikParametrized.idd(s.stimulus_id) for s in self.pnvs]
                
                
                # transform the pnvs into a dictionary of tuning curves according along the 'surround_orientation' parameter
                # also make sure they are ordered according to the first pnv's idds 
                
                self.tc_dict = colapse_to_dictionary([z.get_value_by_id(self.parameters.neurons) for z in self.pnvs],self.st,"surround_orientation")
                for k in self.tc_dict.keys():
                        sis = []
                        surround_tuning=[]
                        
                        # we will do the calculation neuron by neuron
                        for i in range(0,len(self.parameters.neurons)):
                            
                            ors = self.tc_dict[k][0]
                            values = numpy.array([a[i] for a in self.tc_dict[k][1]])
                            d=OrderedDict()
                            for o,v in zip(ors,values):
                                d[o] = v
                            sis.append(d[0] / d[numpy.pi/2])
                            
                            
                        self.datastore.full_datastore.add_analysis_result(PerNeuronValue(sis,self.parameters.neurons,None,value_name = 'Suppression index of ' + self.pnvs[0].value_name ,sheet_name=self.parameters.sheet_name,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(k)))

class F1CorrectedStandardDeviation(Analysis):
      """
      Calculates the standard deviation of an AnalogSignalList after removing its F1 component.
      The F1 component had to be previous calculated with the Analog_F0andF1 analysis
      The standard deviation is computed as the average of the standard deviation of every instance
      of a sliding time window of duration `time_window`. To compute the STD over the whole stimulus
      presentation, set `time_window` to 0.
      """

      required_parameters = ParameterSet({
          'y_axis_name': str,  # Name of the y_axis of the AnalogSignalList to compute the standard deviation on
          'parameters_sort' : list, # The names of the varying parameters in the dsv. ADS in each sub-dsv will be sorted against these parameters to ensure their order is the same
          'time_window': float,  # the length (in ms) of the time window over wich the standard deviations will be computed. If 0, no time_window is used

      })
      def perform_analysis(self):
          for sheet in self.datastore.sheets():
              dsv = queries.param_filter_query(self.datastore,sheet_name=sheet)

              # Get the F1 component amplitudes of the signals
              dsv_f1 = queries.param_filter_query(dsv,value_name=f'F1({self.parameters.y_axis_name})')
              for parameter in self.parameters.parameters_sort:
                  dsv_f1.sort_analysis_results(parameter)
              f1s =  dsv_f1.get_analysis_result()

              # Get the F1 component phases of the signals
              dsv_f1_phase = queries.param_filter_query(dsv,value_name=f'F1 phase({self.parameters.y_axis_name})')
              for parameter in self.parameters.parameters_sort:
                  dsv_f1_phase.sort_analysis_results(parameter)
              f1_phases =  dsv_f1_phase.get_analysis_result()

              # Get the signal on which to compute the STD
              dsv_signal = queries.param_filter_query(dsv,y_axis_name=self.parameters.y_axis_name)
              for parameter in self.parameters.parameters_sort:
                  dsv_signal.sort_analysis_results(parameter)
              signals =  dsv_signal.get_analysis_result()

              for f1, f1_phase, signal in zip(f1s,f1_phases,signals):
                  # Wouldn't work if ids are the same but not in the correct order.
                  stimulus = signal.stimulus_id
                  ids = signal.ids

                  first_analog_signal = signal.asl[0]
                  length = len(first_analog_signal)
                  duration = first_analog_signal.t_stop - first_analog_signal.t_start
                  frequency = MozaikParametrized.idd(f1.stimulus_id).temporal_frequency * MozaikParametrized.idd(f1.stimulus_id).getParams()['temporal_frequency'].units
                  period = 1/frequency
                  period = period.rescale(first_analog_signal.t_start.units)
                  cycles = duration / period
                  first_har = int(round(cycles))
                  index_window = int(first_analog_signal.sampling_rate * self.parameters.time_window)
                  stds = []
                  # For each neuron
                  for idd in ids:
                      # Reconstruct the F1 component based on its amplitude and phase
                      reconstructed_component = f1.get_value_by_id(idd) * numpy.exp(1j * f1_phase.get_value_by_id(idd))/2*length
                      reconstructed_fft = numpy.zeros(length, dtype=complex)
                      reconstructed_fft[first_har] = reconstructed_component
                      reconstructed_fft[-first_har] = numpy.conj(reconstructed_component)
                      reconstructed_signal = numpy.fft.ifft(reconstructed_fft)

                      # Substract the reconstructed component to the signal and compute the STD
                      if self.parameters.time_window != 0:
                          # If time_window argument is great than 0, compute a rolling STD over the time window
                          stds.append(numpy.mean(pandas.Series(signal.get_asl_by_id(idd).magnitude.flatten() - reconstructed_signal).rolling(index_window).std(ddof=0)))
                      else:
                          stds.append(numpy.std(signal.get_asl_by_id(idd).magnitude.flatten() - reconstructed_signal))

                  self.datastore.full_datastore.add_analysis_result(PerNeuronValue(stds,signal.ids,signal.y_axis_units,value_name = 'F1 corrected STD ('+ signal.y_axis_name + ')',sheet_name=sheet,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=signal.stimulus_id))

class LinearityIndices(Analysis):
      """
      Calculates the Non-linearity index and Linearity index as in Cagnol et al. 2025
      in respect to a PerNeuronValue with name corresponding to the `value_name` argument
      """

      required_parameters = ParameterSet({
          'value_name': str,  # Name of the PerNeuronValue to compute the linearity indices on
      })

      def perform_analysis(self):
          inner_radius =  numpy.sort(list(set([MozaikParametrized.idd(s).inner_aperture_radius for s in queries.param_filter_query(self.datastore,st_name='DriftingSinusoidalGratingRing').get_stimuli()])))
          radius =  numpy.sort(list(set([MozaikParametrized.idd(s).radius for s in queries.param_filter_query(self.datastore,st_name='DriftingSinusoidalGratingDisk').get_stimuli()])))

          for sheet in self.datastore.sheets():
              dsv_disk = queries.param_filter_query(self.datastore, value_name=self.parameters.value_name, st_name='DriftingSinusoidalGratingDisk', st_direct_stimulation_name=None, sheet_name=sheet)
              dsv_ring = queries.param_filter_query(self.datastore, value_name=self.parameters.value_name, st_name='DriftingSinusoidalGratingRing', st_direct_stimulation_name=None, sheet_name=sheet)
              ids = dsv_disk.get_analysis_result()[0].ids
              
              idd = []
              nlis = []
              lis = []
              sts = []
               
              # Fore each neuron
              for idd in ids:
                  disk = []
                  ring = []
                  disk_sd = []
                  ring_sd = []

                  # For each radius of disk stimuli, compute the mean value of the PNV across trials
                  for r in radius:
                      dsv = queries.param_filter_query(dsv_disk, st_radius = r)
                      stimuli = [pnv.stimulus_id for pnv in dsv.get_analysis_result()]
                      pnvs1, stids = colapse(dsv.get_analysis_result(),stimuli,parameter_list=['trial'],allow_non_identical_objects=True)
                      for pnvs in pnvs1:
                          disk.append(numpy.mean([pnv.get_value_by_id(idd) for pnv in pnvs]))
                      
                      # Save the disk stimuli with trial=None in a stimuli list which will be used when adding LIs to the datastore
                      if len(sts) != len(radius):
                          sts.append(stids[0])
    
                  # For each inner radius of ring stimuli, compute the mean value of the PNV across trials
                  for r in inner_radius:
                      dsv = queries.param_filter_query(dsv_ring, st_inner_aperture_radius = r)
                      stimuli = [pnv.stimulus_id for pnv in dsv.get_analysis_result()]
                      pnvs1, stids = colapse(dsv.get_analysis_result(),stimuli,parameter_list=['trial'],allow_non_identical_objects=True)
                      for pnvs in pnvs1:
                          ring.append(numpy.mean([pnv.get_value_by_id(idd) for pnv in pnvs]))
    
                  disk = numpy.array(disk)
                  ring = numpy.array(ring)

                  # Compute the non-linearity index
                  peak_dep =  max(disk-ring[-1])
                  if peak_dep > 0:
                      sumVm = max(disk + ring - 2*ring[-1])
                      normsum= (sumVm-peak_dep)/peak_dep
                      nli = normsum
                  else:
                      nli = numpy.nan
                  nlis.append(nli)

                  # Compute the linearity indices at each radius value
                  lis.append(100*((disk[-1]-ring[-1] - (disk + ring - 2*ring[-1]))/(disk[-1]-ring[-1])))
             
              lis = numpy.array(lis)
            
              # Save the NLIs
              self.datastore.full_datastore.add_analysis_result(PerNeuronValue(list(nlis), ids,None,value_name = 'NLI ' + self.parameters.value_name,sheet_name=sheet,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=None))

              # For each disk stimulus radius, save the corresponding LI. The fact that disks are used and not rings is arbitrary
              for i in range(len(sts)):
                  self.datastore.full_datastore.add_analysis_result(PerNeuronValue(list(lis[:,i]), ids,None,value_name = 'LI ' + self.parameters.value_name,sheet_name=sheet,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(sts[i])))
