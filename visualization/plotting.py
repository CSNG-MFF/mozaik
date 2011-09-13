# the visualization for mozaik objects
# it is based on the matplotlib 
# one important concept the visualization code should follow is that it should not
# itself call figure() or subplot() commands, instead assume that they were already
# called before it. This allows for a simple encapsulation of the individual figures 
# into more complex matplotlib figures.

import pylab
import numpy
from MozaikLite.framework.interfaces import MozaikLiteParametrizeObject
from MozaikLite.stimuli.stimulus_generator import parse_stimuls_id,load_from_string
from NeuroTools.parameters import ParameterSet, ParameterDist

class Plotting(MozaikLiteParametrizeObject):
    
    def  __init__(self,datastore,parameters):
         MozaikLiteParametrizeObject.__init__(self,parameters)
         self.datastore = datastore
    
    def plot(self):
        pass

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
    
    def plot(self):
        for tc in self.tuning_curves:
            tc = tc.to_dictonary_of_tc_parametrization()
            n = self.parameters.neuron
            pylab.figure()
            for k in  tc:
                (a,b) = tc[k]
                par,val = zip(*sorted(zip(b,a[:,n])))
                pylab.plot(par,val,fromat_stimulus_id(parse_stimuls_id(k)),label=k)
            pylab.title('Orientation tuning curve, Neuron: %d' % n)
            pylab.ylabel(self.parameters.ylabel)
            pylab.legend()
            
class CyclicTuningCurvePlot(PlotTuningCurve):
    """
    Tuning curve over cyclic domain
    """
    def plot(self):
        n = self.parameters.neuron
        pylab.figure()
        
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
                pylab.plot(numpy.arange(len(val)),val,fromat_stimulus_id(parse_stimuls_id(k)))
            pylab.ylabel(self.parameters.ylabel)
            pylab.xticks(numpy.arange(len(val)),["%.2f"% float(a) for a in par])
            pylab.title('Orientation tuning curve, Neuron: %d' % n)
      
        
def fromat_stimulus_id(stimulus_id):
    string = ''
    for p in stimulus_id.parameters:
        if p != '*' and p != 'x':
            string = string + ' ' + str(p)
    return string



class NeurotoolsPlot(Plotting):
    
    required_parameters = ParameterSet({
	  'data_name' : str,  #the name of the tuning curve
      'sheet_name' : str,  #the name of the sheet for which to plot
	})

    def  __init__(self,datastore,parameters):
        Plotting.__init__(self,datastore,parameters)
        ar = self.datastore.get_analysis_result(parameters.data_name,sheet_name = parameters.sheet_name)    
        if len(ar) > 1:
           print 'ERROR: There should not be more than one NeuroTools analysis datastructure in storage currently!!!!'
           return 

        ar = ar[0]    
        self.vm_data_dict = ar.vm_data_dict
        self.g_syn_e_data_dict = ar.g_syn_e_data_dict
        self.g_syn_i_data_dict = ar.g_syn_i_data_dict
        self.spike_data_dict = ar.spike_data_dict

class RasterPlot(NeurotoolsPlot):
      def plot(self): 
          print 'Starting RasterPlot analysis'
          for sp,st in zip(self.spike_data_dict[0],self.spike_data_dict[1]):
              sp.raster_plot()
              pylab.title(sheet+ ': ' + str(st))
              print sheet + ' mean rate is:' + numpy.str(numpy.mean(numpy.array(sp.mean_rates())))

class VmPlot(NeurotoolsPlot):
      def plot(self):           
          print 'Starting VmPlot analysis'
          for vm,st in zip(self.vm_data_dict[0],self.vm_data_dict[1]):
              vm[-1].plot(ylabel='Vm')
              pylab.title(sheet+ ': ' + str(st))

class GSynPlot(NeurotoolsPlot):
      def plot(self): 
          print 'Starting GSynPlot analysis'
          for gsyn_e,gsyn_i,st in zip(self.g_syn_e_data_dict[0],self.g_syn_i_data_dict[0],self.vm_data_dict[1]):
              pylab.figure()
              f=pylab.subplot(111)
              gsyn_e[-1].plot(display=f,kwargs={'color':'r','label':'exc'})
              gsyn_i[-1].plot(display=f,kwargs={'color':'b','label':'inh'})
              pylab.ylabel('g_syn')
              pylab.legend()
              pylab.title(sheet+ ': ' + str(st))

