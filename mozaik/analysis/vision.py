# -*- coding: utf-8 -*-
"""
Module containing vision specific analysis.
"""
import mozaik
import numpy
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
            psths, stids = colapse(psths,st,parameter_list=['trial'],func=neo_sum,allow_non_identical_objects=True)

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
                ids = []
                frequency = MozaikParametrized.idd(st).temporal_frequency * MozaikParametrized.idd(st).getParams()['temporal_frequency'].units
                for (orr, ppsth) in zip(vl[0], vl[1]):
                    for j in numpy.nonzero(orr == closest_presented_orientation)[0]:
                        if or_pref.ids[j] in ppsth.ids:
                            a = or_pref.ids[j]
                            mr,F0,F1 = self._calculate_MR(ppsth.get_asl_by_id(or_pref.ids[j]).flatten(),frequency)
                            modulation_ratio.append(mr)
                            f0.append(F0)
                            f1.append(F1)
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


                import pylab
                pylab.figure()
                pylab.hist(modulation_ratio)

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
            return 2*abs(fft[first_har])/abs(fft[0]),abs(fft[0]),2*abs(fft[first_har]),
        else:
            logger.info("MR: ARGH: " + str(fft[0]) +"  " +  str(numpy.mean(signal)))
            return 10,abs(fft[0]),2*abs(fft[first_har]),

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
                    logger.info(str(first_analog_signal.t_start))
                    logger.info(str(first_analog_signal.t_stop))
                    duration = first_analog_signal.t_stop - first_analog_signal.t_start
                    frequency = MozaikParametrized.idd(asl.stimulus_id).temporal_frequency * MozaikParametrized.idd(asl.stimulus_id).getParams()['temporal_frequency'].units
                    period = 1/frequency
                    period = period.rescale(first_analog_signal.t_start.units)
                    cycles = duration / period
                    first_har = int(round(cycles))

                    f0 = [abs(numpy.fft.fft(signal.flatten())[0])/len(signal) for signal in signals]
                    f1 = [2*abs(numpy.fft.fft(signal.flatten())[first_har])/len(signal) for signal in signals]
                    
                    self.datastore.full_datastore.add_analysis_result(PerNeuronValue(f0,asl.ids,asl.y_axis_units,value_name = 'F0('+ asl.y_axis_name + ')',sheet_name=sheet,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=asl.stimulus_id))                            
                    self.datastore.full_datastore.add_analysis_result(PerNeuronValue(f1,asl.ids,asl.y_axis_units,value_name = 'F1('+ asl.y_axis_name + ')',sheet_name=sheet,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=asl.stimulus_id))                                                


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
                            
                            
                        self.datastore.full_datastore.add_analysis_result(PerNeuronValue(max_responses,self.parameters.neurons,self.st[0].getParams()["radius"].units,value_name = 'Max. response of ' + self.pnvs[0].value_name ,sheet_name=self.parameters.sheet_name,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(k)))
                        self.datastore.full_datastore.add_analysis_result(PerNeuronValue(crf_sizes,self.parameters.neurons,self.st[0].getParams()["radius"].units,value_name = 'Max. facilitation radius of ' + self.pnvs[0].value_name ,sheet_name=self.parameters.sheet_name,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(k)))
                        self.datastore.full_datastore.add_analysis_result(PerNeuronValue(supp_sizes,self.parameters.neurons,self.st[0].getParams()["radius"].units,value_name = 'Max. suppressive radius of ' + self.pnvs[0].value_name ,sheet_name=self.parameters.sheet_name,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(k)))
                        self.datastore.full_datastore.add_analysis_result(PerNeuronValue(sis,self.parameters.neurons,None,value_name = 'Suppression index of ' + self.pnvs[0].value_name ,sheet_name=self.parameters.sheet_name,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(k)))
                        self.datastore.full_datastore.add_analysis_result(PerNeuronValue(csis,self.parameters.neurons,None,value_name = 'Counter-suppression index of ' + self.pnvs[0].value_name ,sheet_name=self.parameters.sheet_name,tags=self.tags,period=None,analysis_algorithm=self.__class__.__name__,stimulus_id=str(k)))
                        

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

