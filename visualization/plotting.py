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
from MozaikLite.storage.queries import select_stimuli_type_query,select_result_sheet_query, partition_by_stimulus_paramter_query
from MozaikLite.visualization.plotting_helper_functions import *

class Plotting(MozaikLiteParametrizeObject):
    
    def  __init__(self,datastore,parameters):
         MozaikLiteParametrizeObject.__init__(self,parameters)
         self.datastore = datastore
    
    def subplot(self,subplotspec):
        raise NotImplementedError
        pass
    
    def plot(self):
        pylab.figure(facecolor='w')
        gs = gridspec.GridSpec(1, 1)
        gs.update(left=0.05, right=0.95, top=0.95, bottom=0.05)
        self.subplot(gs[0,0])

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


class LinePlot(Plotting):          
      """
      Many plots plot many identical plots in a row for multiple data in a list they get
      This is a smaller helper class that mitigates some of the code repetition in such cases
      
      Note that the inherited class has to implement!:
      _subplot(self,idx,ax) 
      
      which plots the individual plot. The idx is index in whatever datastructure list we are plotting ans 
      axis is the axis that has to be used for plotting.
      """ 
      def  __init__(self,datastore,parameters):
           Plotting.__init__(self,datastore,parameters)    
           self.length = None
      
      def subplot(self,subplotspec): 
          if not self.length:
             print 'Error, class that derives from LinePlot has to specify the length parameter'
             return

          gs = gridspec.GridSpecFromSubplotSpec(1, self.length, subplot_spec=subplotspec)  
          for idx in xrange(0,self.length):
              self._subplot(idx,gs[0,idx])


class PerStimulusPlot(LinePlot):
    
    required_parameters = ParameterSet({
      'sheet_name' : str,  #the name of the sheet for which to plot
	})

    def  __init__(self,datastore,parameters):
        Plotting.__init__(self,datastore,parameters)
        self.dsv = select_result_sheet_query(datastore,self.parameters.sheet_name)
        self.dsvs = partition_by_stimulus_paramter_query(self.dsv,7)    
        self.length = len(self.dsvs)

class RasterPlot(PerStimulusPlot):
      required_parameters = ParameterSet({
        'trial_averaged_histogram' : bool,  #should the plot show also the trial averaged histogram
      })
    
      def _subplot(self,idx,gs):
          
         gs = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs)  
         
         pylab.rc('axes', linewidth=3)
         
         ax = pylab.subplot(gs[:3,0])   
         sp = self.dsvs[idx].get_spike_lists()  
         
         t_stop = sp[0].t_stop
         num_n = len(sp[0])
         num_t = len(sp)
         
         for i,spike_list in enumerate(sp):
            for j,spike_train in enumerate(spike_list):
                ax.plot(spike_train.spike_times,[j*(num_t+1) + i + 1 for x in xrange(0,len(spike_train.spike_times))],',',color='#848484')
             
         for j in xrange(0,num_n-1):   
            ax.axhline(j*(num_t+1)+num_t+1,c='k')
         
         pylab.xlim(0,t_stop)
         
         disable_xticks(ax)
         disable_top_right_axis(ax)
         remove_x_tick_labels()
         remove_y_tick_labels()
         if idx == 0:
           pylab.ylabel('Nuron/Trial #')
         else:
           pylab.ylabel('')
        
         ### lets do the histogram
         ax = pylab.subplot(gs[3,0])   
         all_spikes = []
         
         for i,spike_list in enumerate(sp):
            for j,spike_train in enumerate(spike_list):
                all_spikes.extend(spike_train.spike_times)
         
         if all_spikes != []:
             ax.hist(all_spikes,bins=numpy.arange(0,t_stop,1),color='k')
             if idx == 0:
               pylab.ylabel('(spk/ms)')
             else:
               pylab.ylabel('')
         
         pylab.xlim(0,t_stop)
         pylab.xticks([0,t_stop/2,t_stop])
         
         disable_top_right_axis(ax)
         three_tick_axis(ax.yaxis)
                 
         pylab.rc('axes', linewidth=1)

class VmPlot(PerStimulusPlot):
      required_parameters = ParameterSet({
        'neuron' : int,  #we can only plot one neuron - which one ?
	  })
    
      def _subplot(self,idx,gs):
          pylab.rc('axes', linewidth=3)
          ax = pylab.subplot(gs)
          vms = self.dsvs[idx].get_vm_lists()
          mean_v = numpy.zeros(numpy.shape(vms[0][self.parameters.neuron].signal))
          time_axis = vms[0][self.parameters.neuron].time_axis()
          t_stop =  vms[0][self.parameters.neuron].t_stop - (time_axis[1] - time_axis[0])
          
          
                    
          for vm in vms:
            ax.plot(time_axis,vm[self.parameters.neuron].signal,color="#848484")                
            mean_v = mean_v + vm[self.parameters.neuron].signal   
          
          mean_v = mean_v / len(vms)
          ax.plot(time_axis,mean_v,color='k',linewidth=2)              
          disable_top_right_axis(ax)            
          
          pylab.xlim(0,t_stop)
          pylab.xticks([0,t_stop/2,t_stop])
          three_tick_axis(ax.yaxis)
          
          if idx == 0:
            pylab.ylabel('Vm(mV)')
          else:
            pylab.ylabel('')
          pylab.rc('axes', linewidth=1)            

class GSynPlot(PerStimulusPlot):

      required_parameters = ParameterSet({
        'neuron' : int  #we can only plot one neuron - which one ?
	  })
    
      def _subplot(self,idx,gs):
          pylab.rc('axes', linewidth=3)
          ax = pylab.subplot(gs)
          gsyn_es = self.dsvs[idx].get_gsyn_e_lists()  
          gsyn_is = self.dsvs[idx].get_gsyn_i_lists()  
          mean_gsyn_e = numpy.zeros(numpy.shape(gsyn_es[0][self.parameters.neuron].copy().signal))
          mean_gsyn_i = numpy.zeros(numpy.shape(gsyn_is[0][self.parameters.neuron].copy().signal))
          time_axis = gsyn_es[0][self.parameters.neuron].time_axis()
          t_stop = gsyn_es[0][self.parameters.neuron].t_stop - (time_axis[1] - time_axis[0])
          
          for e,i in zip(gsyn_es,gsyn_is):
            p1 = ax.plot(time_axis,e[self.parameters.neuron].signal*10**9,color='#F5A9A9')            
            p2 = ax.plot(time_axis,i[self.parameters.neuron].signal*10**9,color='#A9BCF5')              
            mean_gsyn_e = mean_gsyn_e + e[self.parameters.neuron].signal
            mean_gsyn_i = mean_gsyn_i + i[self.parameters.neuron].signal
          
          mean_gsyn_i = mean_gsyn_i / len(gsyn_is)
          mean_gsyn_e = mean_gsyn_e / len(gsyn_es)
            
          p1 = ax.plot(time_axis,mean_gsyn_e*10**9,color='r',linewidth=2)            
          p2 = ax.plot(time_axis,mean_gsyn_i*10**9,color='b',linewidth=2)              
          
          disable_top_right_axis(ax)
          
          if idx == 0:
            pylab.ylabel('G(nS)')
          else:
            pylab.ylabel('')
          
          pylab.xlim(0,t_stop)
          pylab.xticks([0,t_stop/2,t_stop])
          three_tick_axis(ax.yaxis)
                    
          pylab.legend([p1,p2],['exc','inh'])  
          pylab.rc('axes', linewidth=1)
          
class OverviewPlot(Plotting):
      required_parameters = ParameterSet({
            'sheet_name' : str,  #the name of the sheet for which to plot
      })
      def subplot(self,subplotspec):
          gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=subplotspec)  
          RasterPlot(self.datastore,ParameterSet({'sheet_name' : self.parameters.sheet_name, 'trial_averaged_histogram' : True})).subplot(gs[0,0])
          GSynPlot(self.datastore,ParameterSet({'sheet_name' : self.parameters.sheet_name,'neuron' : 0})).subplot(gs[1,0])
          VmPlot(self.datastore,ParameterSet({'sheet_name' : self.parameters.sheet_name,'neuron' : 0})).subplot(gs[2,0])          

          
class AnalogSignalListPlot(LinePlot):
        required_parameters = ParameterSet({
            'sheet_name' : str,  #the name of the sheet for which to plot
            'ylabel' : str,  #what to put as ylabel
        })
    
        def  __init__(self,datastore,parameters):
            LinePlot.__init__(self,datastore,parameters)
            self.analog_signal_list = self.datastore.get_analysis_result('AnalogSignalList',sheet_name = parameters.sheet_name)    
            if len(self.analog_signal_list) > 1:
              print 'ERROR: Warning currently only the first AnalogSignalList will be plotted'
            self.analog_signal_list = self.analog_signal_list[0]
            self.asl = self.analog_signal_list.asl
            self.length = len(self.asl)
            
    
        def _subplot(self,idx,gs):
              pylab.rc('axes', linewidth=3)
              ax = pylab.subplot(gs)
              self.asl[idx].plot(display=ax,kwargs={'color':'b'})
              if idx == 0:
                 pylab.ylabel(self.parameters.ylabel)
              
              disable_top_right_axis(ax)  
              three_tick_axis(ax.yaxis)            
              pylab.rc('axes', linewidth=1)

class ConductanceSignalListPlot(LinePlot):
        required_parameters = ParameterSet({
            'sheet_name' : str,  #the name of the sheet for which to plot
        })
    
        def  __init__(self,datastore,parameters):
            LinePlot.__init__(self,datastore,parameters)
            self.conductance_signal_list = self.datastore.get_analysis_result('ConductanceSignalList',sheet_name = parameters.sheet_name)    
            if len(self.conductance_signal_list) > 1:
              print 'ERROR: Warning currently only the first ConductanceSignalList will be plotted'
            self.conductance_signal_list = self.conductance_signal_list[0]
            self.e_con = self.conductance_signal_list.e_con
            self.i_con = self.conductance_signal_list.i_con
            self.length = len(self.e_con)
            
    
        def _subplot(self,idx,gs):
              pylab.rc('axes', linewidth=3)
              ax = pylab.subplot(gs)
              self.e_con[idx].plot(display=ax,kwargs={'color':'r','label':'exc'})
              self.i_con[idx].plot(display=ax,kwargs={'color':'b','label':'inh'})
              if idx == 0:
                 pylab.ylabel('G(S)')

              disable_top_right_axis(ax)      
              three_tick_axis(ax.yaxis)        
              three_tick_axis(ax.xaxis)                      
              pylab.rc('axes', linewidth=1)  
