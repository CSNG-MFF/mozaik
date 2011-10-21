import MozaikLite.storage.queries as queries
from MozaikLite.visualization.plotting import Plotting,RasterPlot,GSynPlot,VmPlot,ConductanceSignalListPlot,AnalogSignalListPlot
from NeuroTools.parameters import ParameterSet, ParameterDist
import matplotlib.gridspec as gridspec
from MozaikLite.storage.queries import select_stimuli_type_query,select_result_sheet_query, partition_by_stimulus_paramter_query

class Figure2(Plotting):
      required_parameters = ParameterSet({
            'sheet_name' : str,  #the name of the sheet for which to plot
      })
      
      def subplot(self,subplotspec):
          gs = gridspec.GridSpecFromSubplotSpec(12, 8, subplot_spec=subplotspec,hspace=1.0,wspace=1.0)  
          
          dsv = queries.select_stimuli_type_query(self.datastore,'FullfieldDriftingSinusoidalGrating',['*','*','*','*',90.0,'*','*','*',0.0,'*','*'])
          RasterPlot(dsv,ParameterSet({'sheet_name' : self.parameters.sheet_name, 'trial_averaged_histogram' : True, 'neurons' : [0]})).subplot(gs[0:4,:5])
          GSynPlot(dsv,ParameterSet({'sheet_name' : self.parameters.sheet_name,'neuron' : 0})).subplot(gs[4:8,:5])
          VmPlot(dsv,ParameterSet({'sheet_name' : self.parameters.sheet_name,'neuron' : 0})).subplot(gs[8:12,:5])          
          
          ConductanceSignalListPlot(queries.TagBasedQuery(ParameterSet({'tags' : ['GSTA1'] })).query(self.datastore),ParameterSet({'sheet_name' : 'V1_Exc'})).subplot(gs[2:5,6:])  
          
          AnalogSignalListPlot(dsv,ParameterSet({'sheet_name' : self.parameters.sheet_name,'ylabel' : 'AC (norm)'})).subplot(gs[7:10,6:])
          
