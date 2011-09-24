# the visualization for mozaik objects
# it is based on the matplotlib and the GridSpec object
# Each plot should implement two plotting functions:  plot and subplot

# The much more important one is subplot that accepts a SubplotSpec object (see matplotlib doc) as input which will 
# tell it where to plot. It can in turn create another SubplotSpec within
# the given SubplotSpec and call other plot commands to plot within specific subregions
# of the SubplotSpec. This allows natural way of nesting plots.

# the plot function can either not be defined in which case it defaults to the Plotting.plot 
# which simply creates a figure and calls subplot with SuplotSpec spanning the whole figure.
# Alternatively one can define the plot function if one wants to add some additional decorations
# if one know the figures is plotted on its own, that would otherwise prevent flexible use in
# nesting via the subplot 


import pylab
import matplotlib.gridspec as gridspec
import numpy
from MozaikLite.framework.interfaces import MozaikLiteParametrizeObject
from MozaikLite.stimuli.stimulus_generator import parse_stimuls_id,load_from_string
from NeuroTools.parameters import ParameterSet, ParameterDist


class Plotting(MozaikLiteParametrizeObject):
    
    def  __init__(self,datastore,parameters):
         MozaikLiteParametrizeObject.__init__(self,parameters)
         self.datastore = datastore
    
    def subplot(self,subplotspec):
        pass
    
    def plot(self):
        pylab.figure()
        self.subplot(gridspec.GridSpec(1, 1)[0,0])

class PlotTuningCurve(Plotting):
    """
    values - contain a list of lists of values, the outer list corresponding
    to stimuli the inner to neurons.
    
    stimuli_ids - contain liest of stimuli ids corresponding to the values
    
    parameter_index - corresponds to the parameter that should be plotted as 
                    - a tuning curve
                    
    neurons - which        """

    required_parameters = ParameterSet({
	  'tuning_curve_name' : str,  #the name of the tuning curve
      'neuron': int, # which neuron to plot
      'sheet_name' : str, # from which layer to plot the tuning curve
      'ylabel': str, # ylabel to write on the graph
	})

    
    def  __init__(self,datastore,parameters):
        Plotting.__init__(self,datastore,parameters)
        self.tuning_curves = self.datastore.get_analysis_result(parameters.tuning_curve_name,sheet_name=parameters.sheet_name)
    
    def subplot(self,subplotspec):
        ax = pylab.subplot(subplotspec)
        for tc in self.tuning_curves:
            tc = tc.to_dictonary_of_tc_parametrization()
            n = self.parameters.neuron
            for k in  tc:
                (a,b) = tc[k]
                par,val = zip(*sorted(zip(b,a[:,n])))
                pylab.plot(par,val,label=fromat_stimulus_id(parse_stimuls_id(k)))
            pylab.title('Orientation tuning curve, Neuron: %d' % n)
            pylab.ylabel(self.parameters.ylabel)
            pylab.legend()
            
class CyclicTuningCurvePlot(PlotTuningCurve):
    """
    Tuning curve over cyclic domain
    """
    def subplot(self,subplotspec):
        n = self.parameters.neuron
        ax = pylab.subplot(subplotspec)
        for tc in self.tuning_curves:
            tc = tc.to_dictonary_of_tc_parametrization()
            for k in  tc:
                (a,b) = tc[k]
                par,val = zip(*sorted(zip(b,a[:,n])))
                # make the tuning curve to wrap around  
                par = list(par)
                val = list(val)
                par.append(par[0])
                val.append(val[0])
                pylab.plot(numpy.arange(len(val)),val,label=fromat_stimulus_id(parse_stimuls_id(k)))
                pylab.hold('on')
            pylab.ylabel(self.parameters.ylabel)
            pylab.ylabel('Orientation')
            pylab.xticks(numpy.arange(len(val)),["%.2f"% float(a) for a in par])
            pylab.title('Orientation tuning curve, Neuron: %d' % n)
        pylab.legend()
        
        
def fromat_stimulus_id(stimulus_id):
    string = ''
    for p in stimulus_id.parameters:
        if p != '*' and p != 'x':
            string = string + ' ' + str(p)
    return string



class NeurotoolsPlot(Plotting):
    
    required_parameters = ParameterSet({
      'sheet_name' : str,  #the name of the sheet for which to plot
	})

    def  __init__(self,datastore,parameters):
        Plotting.__init__(self,datastore,parameters)
        ar = self.datastore.get_analysis_result('NeurotoolsData',sheet_name = parameters.sheet_name)    
        if len(ar) > 1:
           print 'ERROR: There should not be more than one NeuroTools analysis datastructure in storage currently!!!!'
           return 

        ar = ar[0]    
        self.vm_data_dict = ar.vm_data_dict
        self.g_syn_e_data_dict = ar.g_syn_e_data_dict
        self.g_syn_i_data_dict = ar.g_syn_i_data_dict
        self.spike_data_dict = ar.spike_data_dict

class RasterPlot(NeurotoolsPlot):
      def subplot(self,subplotspec): 
          gs = gridspec.GridSpecFromSubplotSpec(1, len(self.spike_data_dict[1]), subplot_spec=subplotspec)    
          for sp,st,idx in zip(self.spike_data_dict[0],self.spike_data_dict[1],numpy.arange(0,len(self.spike_data_dict[1]),1)):
              ax = pylab.subplot(gs[0,idx])
              sp.raster_plot(display=ax)
              #print sheet + ' mean rate is:' + numpy.str(numpy.mean(numpy.array(sp.mean_rates())))
              if idx == 0:
                pylab.ylabel('Neuron #')
              else:
                pylab.ylabel('')


class VmPlot(NeurotoolsPlot):
      def subplot(self,subplotspec):           
          gs = gridspec.GridSpecFromSubplotSpec(1, len(self.vm_data_dict[1]), subplot_spec=subplotspec)    
          for vm,st,idx in zip(self.vm_data_dict[0],self.vm_data_dict[1],numpy.arange(0,len(self.vm_data_dict[1]),1)):
              ax = pylab.subplot(gs[0,idx])
              vm[-1].plot(display=ax,ylabel='Vm')
              if idx == 0:
                pylab.ylabel('Vm')
              else:
                pylab.ylabel('')

class GSynPlot(NeurotoolsPlot):
      def subplot(self,subplotspec): 
          gs = gridspec.GridSpecFromSubplotSpec(1, len(self.g_syn_e_data_dict[1]), subplot_spec=subplotspec)  
          for gsyn_e,gsyn_i,st,idx in zip(self.g_syn_e_data_dict[0],self.g_syn_i_data_dict[0],self.g_syn_i_data_dict[1],numpy.arange(0,len(self.g_syn_e_data_dict[1]),1)):
              ax = pylab.subplot(gs[0,idx])
              gsyn_e[-1].plot(display=ax,kwargs={'color':'r','label':'exc'})
              gsyn_i[-1].plot(display=ax,kwargs={'color':'b','label':'inh'})
              if idx == 0:
                pylab.ylabel('g_syn')
              else:
                pylab.ylabel('')
          pylab.legend()  
          
class OverviewPlot(NeurotoolsPlot):
      required_parameters = ParameterSet({
            'sheet_name' : str,  #the name of the sheet for which to plot
      })
      
      def subplot(self,subplotspec):
          gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=subplotspec)  
          RasterPlot(self.datastore,ParameterSet({'sheet_name' : self.parameters.sheet_name})).subplot(gs[0,0])
          GSynPlot(self.datastore,ParameterSet({'sheet_name' : self.parameters.sheet_name})).subplot(gs[1,0])
          VmPlot(self.datastore,ParameterSet({'sheet_name' : self.parameters.sheet_name})).subplot(gs[2,0])          
          
