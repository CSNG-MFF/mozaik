#from MozaikLite.tools.misc import segments_to_dict_of_SpikeList,segments_to_dict_of_AnalogSignalList
from MozaikLite.stimuli.stimulus_generator import colapse
from MozaikLite.analysis.analysis_data_structures import TuningCurve, NeurotoolsData, ConductanceSignalList , AnalogSignalList
from MozaikLite.analysis.analysis_helper_functions import time_histogram_across_trials
from MozaikLite.framework.interfaces import MozaikLiteParametrizeObject
from NeuroTools.parameters import ParameterSet
from MozaikLite.storage.queries import select_stimuli_type_query,select_result_sheet_query, partition_by_stimulus_paramter_query

from NeuroTools import signals
import pylab
import quantities
import numpy 

class Analysis(MozaikLiteParametrizeObject):
    
    def __init__(self,datastore,parameters,tags=[]):
        """
        Analysis encapsulates analysis algorithms
        The interface is extremely simple: it only requires the definition of analyis function
        which when called performs the actualy analysis
        
        It is assumed that it retrieves its own data from DataStore that is suplied in the self.datastore
        parameter. Also it is assumed to include self.tags as the tags for all AnalysisDataStructure that
        it creates. See description of self.tags in AnalysisDataStructure
        
        """
        MozaikLiteParametrizeObject.__init__(self,parameters)
        self.datastore = datastore
        self.tags = tags

    def analyse(self):
        raise NotImplementedError
        pass
        

class AveragedOrientationTuning(Analysis):
    
      def analyse(self):
            print 'Starting OrientationTuning analysis'
            dsv = select_stimuli_type_query(self.datastore,'FullfieldDriftingSinusoidalGrating')

            for sheet in dsv.sheets():
                dsv1 = select_result_sheet_query(dsv,sheet)
                sp = dsv1.get_spike_lists()
                st = dsv1.get_stimuli()
                # transform spike trains due to stimuly to mean_rates
                mean_rates = [numpy.array(s.mean_rates())  for s in sp]
                # collapse against all parameters other then orientation
                (mean_rates,s) = colapse(mean_rates,st,parameter_indexes=[7])
                # take a sum of each 
                def _mean(a):
                    l = len(a)
                    return sum(a)/l
                mean_rates = [_mean(a) for a in mean_rates]  
                self.datastore.full_datastore.add_analysis_result(TuningCurve(mean_rates,s,8,sheet,tags=self.tags),sheet_name=sheet)

class GSTA(Analysis):
      """
      Computes conductance spike triggered average
      
      Note that it does not assume that spikes are aligned with the conductance sampling rate
      and will pick the bin in which the given spike falls (within the conductance sampling rate binning)
      as the center of the conductance vector that is included in the STA
      """
      
      required_parameters = ParameterSet({
        'length': float,  # length (in ms time) how long before and after spike to compute the GSTA
                          # it will be rounded down to fit the sampling frequency
        'neurons' : list, #the list of neuron indexes for which to compute the 
      })

      
      def analyse(self):
            print 'Starting Spike Triggered Analysis of Conductances'
            
            dsv = self.datastore
            for sheet in dsv.sheets():
                dsv1 = select_result_sheet_query(dsv,sheet)
                sp = dsv1.get_spike_lists()
                st = dsv1.get_stimuli()
                g_e = dsv1.get_gsyn_e_lists()
                g_i = dsv1.get_gsyn_i_lists()

                asl_e = []
                asl_i = []
                for n in self.parameters.neurons:
                    asl_e.append(self.do_gsta(g_e,sp,n))
                    asl_i.append(self.do_gsta(g_i,sp,n))
                self.datastore.full_datastore.add_analysis_result(ConductanceSignalList(asl_e,asl_i,sheet,self.parameters.neurons,tags=self.tags),sheet_name=sheet)
                
                
      def do_gsta(self,analog_signal,sp,n):
          dt = analog_signal[0][n].dt
          gstal = int(self.parameters.length/dt)
          gsta = numpy.zeros(2*gstal+1,)
          count = 0
          for (ans,spike) in zip(analog_signal,sp):
              for time in spike[n].spike_times:
                  if time > ans[n].t_start  and time < ans[n].t_stop:
                     idx = int((time - ans[n].t_start)/dt)
                     if idx - gstal > 0 and (idx + gstal+1) <= len(ans[n]):
                        gsta = gsta +  ans[n].signal[idx-gstal:idx+gstal+1]
                        count +=1
          if count == 0:
             count = 1
          return signals.AnalogSignal(gsta/count,dt=dt,t_start=-gstal*dt,t_stop=(gstal+1)*dt)          
           
          
          
class Precision(Analysis):
      """
      Computes the precision as the autocorrelation between the PSTH of different trials
      """
      
      required_parameters = ParameterSet({
        'neurons' : list, #the list of neuron indexes for which to compute the 
        'bin_length' : float, #(ms) the size of bin to construct the PSTH from
      })

      
      def analyse(self):
            print 'Starting Precision Analysis'
            dsv = self.datastore
            for sheet in dsv.sheets():
                dsv1 = select_result_sheet_query(dsv,sheet)
                dsvs = partition_by_stimulus_paramter_query(dsv1,7)
                
                for dsv in dsvs:
                    sl = dsv.get_spike_lists()
                    t_start = sl[0].t_start
                    t_stop = sl[0].t_stop
                    duration = t_stop-t_start
                    
                    hist = time_histogram_across_trials(sl,self.parameters.bin_length)
                    al = []
                    for n in self.parameters.neurons:
                        ac = numpy.correlate(hist[n], hist[n], mode='full')
                        if numpy.sum(numpy.power(hist[n],2)) != 0:
                            ac = ac / numpy.sum(numpy.power(hist[n],2))
                       
                        al.append(signals.AnalogSignal(ac,dt=self.parameters.bin_length*quantities.ms,t_start=-duration,t_stop=duration-self.parameters.bin_length))
                    print 'Adding AnalogSignalList', sheet
                    self.datastore.full_datastore.add_analysis_result(AnalogSignalList(al,sheet,self.parameters.neurons,tags=self.tags),sheet_name=sheet)    
                        
