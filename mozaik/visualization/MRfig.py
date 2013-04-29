import pylab
import numpy

import mozaik.storage.queries as queries
import matplotlib.gridspec as gridspec
from mozaik.visualization.plotting import Plotting
from mozaik.visualization.plotting_helper_functions import *
from NeuroTools.parameters import ParameterSet

class MRfig(Plotting):
      required_parameters = ParameterSet({
            'SimpleSheetName' : str,  #the name of the sheet for which to plot
            'ComplexSheetName' : str, # which neuron to show
      })

      def plot(self):
          self.fig = pylab.figure(facecolor='w', **self.fig_param)
          gs = gridspec.GridSpec(1, 1)
          gs.update(left=0.07, right=0.97, top=0.9, bottom=0.1)
          gs = gs[0,0]
        
          dsv_simple = self.datastore.get_analysis_result(identifier='PerNeuronValue',sheet_name=self.parameters.SimpleSheetName,analysis_algorithm='ModulationRatio')
          dsv_complex = self.datastore.get_analysis_result(identifier='PerNeuronValue',sheet_name=self.parameters.ComplexSheetName,analysis_algorithm='ModulationRatio')
          print len(dsv_simple)
          assert len(dsv_simple) == 1
          assert len(dsv_complex) == 1

          dsv_simple = dsv_simple[0]
          dsv_complex = dsv_complex[0]

          gs = gridspec.GridSpecFromSubplotSpec(3, 1,subplot_spec=gs)
          ax = pylab.subplot(gs[0,0])
          ax.hist(dsv_simple.values,bins=numpy.arange(0,2.2,0.2),color='k')
          pylab.ylim(0,300)
          disable_xticks(ax)
          remove_x_tick_labels()
          remove_y_tick_labels()
          pylab.ylabel('Layer 4',fontsize=15)
          ax = pylab.subplot(gs[1,0])
          ax.hist(dsv_complex.values,bins=numpy.arange(0,2.2,0.2),color='w')
          pylab.ylim(0,300)
          disable_xticks(ax)
          remove_x_tick_labels()
          remove_y_tick_labels()
          pylab.ylabel('Layer 2/3',fontsize=15)
          ax = pylab.subplot(gs[2,0])
          ax.hist([dsv_simple.values,dsv_complex.values],bins=numpy.arange(0,2.2,0.2),histtype='barstacked',color=['k','w'])
          pylab.ylim(0,300)
          pylab.ylabel('Pooled',fontsize=15)
          three_tick_axis(ax.xaxis)
          remove_y_tick_labels()
          pylab.xlabel('Modulation ratio',fontsize=15)
          for label in ax.get_xticklabels() + ax.get_yticklabels(): 
              label.set_fontsize(15) 
          if self.plot_file_name:
                        pylab.savefig(self.plot_file_name)
        
