import numpy
from mozaik.analysis.data_structures import SingleValue
from parameters import ParameterSet
from mozaik.storage import queries
import mozaik
from mozaik.analysis.analysis import Analysis
from mozaik.storage import queries

import logging
logger = logging.getLogger("mozaik")

class SpontaneousActivityLength(Analysis):
      """
      This anlysis detrmines how long does spontaneous activity stays stable. 
      
      It does so by calculating the lenght of period over which the population activity stays non zero, and excluding the points for which either population mean of neuron-to-neuron correlations or mean
      CV reach over a fixed threhsold. For these points it sets the length to 0.
      
      This anlysis assumes that PSTH, CV, and neuron-to-neuron correlation analysis has already been ran on the model, and that the PSTH has been computed with 5ms bin.
      Also PopulationMean analysis over all the above has been ran as well.
      
      NOTE: we should probably give the PSTH bin as parameter.
      """
      def perform_analysis(self):

          for sheet in self.datastore.sheets():
              dsv_psth = queries.param_filter_query(self.datastore,analysis_algorithm="PopulationMean",y_axis_name='Mean(psth (bin=5.0))',sheet_name=sheet)
              dsv_cv = queries.param_filter_query(self.datastore,analysis_algorithm="PopulationMean",value_name='Mean(CV of ISI squared)',sheet_name=sheet,identifier="SingleValue")
              dsv_corr = queries.param_filter_query(self.datastore,analysis_algorithm="PopulationMean",value_name='Mean(Correlation coefficient(psth (bin=5.0)))',sheet_name=sheet,identifier="SingleValue")
            
              assert len(dsv_cv.get_analysis_result()) == 1, "Error: SpontaneousActivityLength accepts only datastore that holds one  SingleValue analysis data structure with value_name: Mean(CV of ISI squared). It contains: %d" % len(dsv_cv.get_analysis_result())
              assert len(dsv_corr.get_analysis_result()) ==1, "Error: SpontaneousActivityLength accepts only datastore that holds one  SingleValue analysis data structure with value_name: 'Mean(Correlation coefficient(psth (bin=5.0))).It contains: %d" % len(dsv_cv.get_analysis_result())
              assert len(dsv_psth.get_analysis_result()) ==1, "Error: SpontaneousActivityLength accepts only datastore that holds one  AnalogSignal analysis data structure with value_name: 'Mean(psth (bin=5.0)).It contains: %d" % len(dsv_cv.get_analysis_result())
            
              if dsv_cv.get_analysis_result()[0].value >= 0.95 and dsv_corr.get_analysis_result()[0].value <= 0.05:
                    i = numpy.nonzero(dsv_psth.get_analysis_result()[0].analog_signal)[0][-1]
                    logger.warning(i)
                    l = dsv_psth.get_analysis_result()[0].analog_signal.times[i]
              else:      
                    l = 0
              logger.warning(dsv_psth.get_analysis_result()[0].analog_signal)
              logger.warning(dsv_cv.get_analysis_result()[0].value)
              logger.warning(dsv_corr.get_analysis_result()[0].value)
              logger.warning(l)
                
              self.datastore.full_datastore.add_analysis_result(SingleValue(value=l,value_name = 'Spontaneous activity length',sheet_name=sheet,tags=self.tags,analysis_algorithm=self.__class__.__name__))        
