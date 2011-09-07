from MozaikLite.tools.misc import segments_to_dict_of_SpikeList,segments_to_dict_of_AnalogSignalList
from analysis_helper_functions import colapse
from MozaikLite.visualization.plotting import CyclicTuningCurve
import pylab
import numpy 

class Analysis(object):
    
    def __init__(self,datastore):
        self.datastore = datastore
    
    def analyse(self):
        pass


        
class OrientationTuning(Analysis):
      def analyse(self):
            print 'Starting OrientationTuning analysis'
            segments = self.datastore.get_recordings('FullfieldDriftingSinusoidalGrating',[])
            data = segments_to_dict_of_SpikeList(segments)
            
            for sheet in data:
                print 'Sheet:' + sheet
                
                (sp,st) = data[sheet]
                # transform spikes trains due to stimuly to mean_rates
                mean_rates = [numpy.array(s.mean_rates())  for s in sp]
                # collapse against all parameters other then orientation
                (mean_rates,s) = colapse(mean_rates,st,parameter_indexes=[0,1,2,3,4,5,6,8,9])
                # take a sum of each 
                def _mean(a):
                    l = len(a)
                    return sum(a)/l
                mean_rates = [_mean(a) for a in mean_rates]  
                CyclicTuningCurve(mean_rates,s,7).plot(ylabel='Mean response rate')

class Neurotools(Analysis):
      """
      Turn the data in one big Neurotools SpikeList that can than be visualized 
      via numerous Neurotools analysis tools
      """
      def __init__(self,datastore):
          Analysis.__init__(self,datastore)
          
      def analyse(self):
          # get all recordings
          segments = self.datastore.get_recordings(None,[])
          self.spike_data_dict = segments_to_dict_of_SpikeList(segments)
          self.vm_data_dict = segments_to_dict_of_AnalogSignalList(segments)
          
            

class RasterPlot(Neurotools):
      
      def analyse(self): 
          print 'Starting RasterPlot analysis'
          Neurotools.analyse(self)
          
          for sheet in self.spike_data_dict:
              for sp,st in zip(self.spike_data_dict[sheet][0],self.spike_data_dict[sheet][1]):
                  sp.raster_plot()
                  pylab.title(sheet+ ': ' + str(st))
                  print sheet + ' mean rate is:' + numpy.str(numpy.mean(numpy.array(sp.mean_rates())))

class VmPlot(Neurotools):
      
      def analyse(self): 
          print 'Starting VmPlot analysis'
          Neurotools.analyse(self)
          
          for sheet in self.vm_data_dict:
              for vm,st in zip(self.vm_data_dict[sheet][0],self.vm_data_dict[sheet][1]):
                  vm[-1].plot()
                  pylab.title(sheet+ ': ' + str(st))
