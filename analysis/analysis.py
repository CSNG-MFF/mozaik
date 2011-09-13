from MozaikLite.tools.misc import segments_to_dict_of_SpikeList,segments_to_dict_of_AnalogSignalList
from analysis_helper_functions import colapse
from MozaikLite.analysis.analysis_data_structures import TuningCurve, NeurotoolsData

import pylab
import numpy 

class Analysis(object):
    
    def __init__(self,datastore):
        self.datastore = datastore

    def analyse(self):
        pass

class AveragedOrientationTuning(Analysis):
    
      def analyse(self):
            print 'Starting OrientationTuning analysis'
            segments = self.datastore.get_recordings('FullfieldDriftingSinusoidalGrating',[])
            data = segments_to_dict_of_SpikeList(segments)
            
            for sheet in data:
                (sp,st) = data[sheet]
                # transform spikes trains due to stimuly to mean_rates
                mean_rates = [numpy.array(s.mean_rates())  for s in sp]
                # collapse against all parameters other then orientation
                (mean_rates,s) = colapse(mean_rates,st,parameter_indexes=[])
                # take a sum of each 
                def _mean(a):
                    l = len(a)
                    return sum(a)/l
                mean_rates = [_mean(a) for a in mean_rates]  
                self.datastore.add_analysis_result(TuningCurve(mean_rates,s,7,sheet),sheet_name=sheet)
            
class Neurotools(Analysis):
      """
      Turn the data into Neurotools data structures that can than be visualized 
      via numerous Neurotools analysis tools
      """
      def __init__(self,datastore):
          Analysis.__init__(self,datastore)
          
      def analyse(self):
          print 'Starting Neurotools analysis'
          # get all recordings
          segments = self.datastore.get_recordings(None,[])
          spike_data_dict = segments_to_dict_of_SpikeList(segments)
          (vm_data_dict,g_syn_e_data_dict,g_syn_i_data_dict) = segments_to_dict_of_AnalogSignalList(segments)
          
          for sheet in vm_data_dict.keys():
              self.datastore.add_analysis_result(NeurotoolsData(spike_data_dict[sheet],vm_data_dict[sheet],g_syn_e_data_dict[sheet],g_syn_i_data_dict[sheet]),sheet_name=sheet)
            

