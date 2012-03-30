"""
The visualization for mozaik objects is based on the matplotlib and the GridSpec object

The plotting framwork is divided into two main concepts, represented by the two high-level
classes Plotting (this file) and SimplePlot (see simple_plot.py).

The SimplePlot represent the low-level plotting. It is assumed that this plot has only a single 
axis that is drawn into the region defined by the GridSpec instance that is passed into it. The 
role of the set of classes derived from SimplePlot is to standardize the low level looks of all 
figures (mainly related to axis, lables, titles etc.), and should assume data given to them in a 
format that is easy to use by the given plot. In order to unify the look of figures
it defines four functions pre_axis_plot,pre_plot, plot, and post_plot. The actual plotting that 
user defines is typically defined in 
the plot function while the pre_axis_plot, pre_plot and post_plot functions handle the pre and post plotting 
adjustments to the plot (i.e. the typical post_plot function for example adjusts the ticks of 
the axis to a common format other such axis related properties). When defining a new SimplePlot 
function user is encoureged to push as much of it's 'decorating' funcitonality into the post 
and pre plot function and define only the absolute minimum
in the plot function. At the same time, there is already a set of classes implementing 
a general common look provided, and so users are encoureged to use these as much as possible. If 
their formatting features are not sufficient or incompatible with a given plot, users are encoureged
to define new virtual class that defines the formatting in the pre and post plot functions 
(and thus sepparating it from the plot itself), and incorporating these as low as possible within 
the hierarchy of the SimplePlot classes to re-use as much of the previous work as possible.

NOTE SimplePlot now resides in sepparate module visualization.simple_plot but its description
stays as it is inegral to how Plotting class works.

The Plotting class (and its children) define the high level plotting mechanisms. They 
are mainly responsible for hierarchical organization of figures with multiple plots, 
any mechanisms that require consideration of several plots at the same time, 
and the translation of the data form the general format provided by Datastore,
to specific format that the SimplePlot plots require. In general the Plotting 
instances should not do any plotting of axes them selves (but instead calling the 
SimplePlot instances to do the actual plotting), with the exception
of multi-axis figures whith complicated inter-axis dependencies, for which it is
not practical to break them down into single SimplePlot instances.

Each Plotting class should implement two plotting functions:  plot and subplot 
The much more important one is subplot that accepts a SubplotSpec object (see matplotlib doc) 
as input which will tell it where to plot. It can in turn create another SubplotSpec within
the given SubplotSpec and call other plot commands to plot within specific subregions
of the SubplotSpec. This allows natural way of nesting plots.

The subplot function has a second parameter which corresponds to dictionary of parameters that
have to be passed onto the eventual call to SimplePlot class! New parameters can be added to the
dictionary but they should not be overwritten! This way higher-level Plotting classes can modify the
behaviour of their nested classes. Also whenever a class is passing this dictionary to multiple subplots
it should always pasa a _copy_ of the parameter dictionary.

The plot function can either not be defined in which case it defaults to the Plotting.plot, 
which simply creates a figure and calls subplot with SuplotSpec spanning the whole figure.
Alternatively, one can define the plot function if one wants to add some additional decorations,
in case the figure is plotted on its own (i.e. becomes the highest-level), and that would otherwise 
prevent flexible use in nesting via the subplot.
"""

import pylab
import numpy
import quantities as pq
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata

from mozaik.framework.interfaces import MozaikParametrizeObject
from mozaik.stimuli.stimulus_generator import parse_stimuls_id,load_from_string, fromat_stimulus_id
from NeuroTools.parameters import ParameterSet, ParameterDist
from mozaik.storage.queries import *
from simple_plot import *
from plot_constructors import *
from mozaik.tools import units


class Plotting(MozaikParametrizeObject):
    
    def  __init__(self,datastore,parameters):
         MozaikParametrizeObject.__init__(self,parameters)
         self.datastore = datastore
    
    def subplot(self,subplotspec,params):
        raise NotImplementedError
        pass
    
    def plot(self,params={}):
        self.fig = pylab.figure(facecolor='w')
        gs = gridspec.GridSpec(1, 1)
        gs.update(left=0.05, right=0.95, top=0.9, bottom=0.1)
        self.subplot(gs[0,0],params)
        

          
              
class PlotTuningCurve(Plotting):
    """
    values - contain a list of lists of values, the outer list corresponding
    to stimuli the inner to neurons.
    
    stimuli_ids - contain list of stimuli ids corresponding to the values
    
    parameter_index - corresponds to the parameter that should be plotted as 
                    - a tuning curve
    """

    required_parameters = ParameterSet({
      'tuning_curve_name' : str,  #the name of the tuning curve
      'neuron': int, # which neuron to plot
      'sheet_name' : str, # from which layer to plot the tuning curve
      'ylabel': str, # ylabel to write on the graph
    })

    def  __init__(self,datastore,parameters):
        Plotting.__init__(self,datastore,parameters)
        self.tuning_curves = self.datastore.get_analysis_result(parameters.tuning_curve_name,sheet_name=parameters.sheet_name)
    
    def subplot(self,subplotspec,params):
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
    def subplot(self,subplotspec,params):
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
            pylab.xticks(numpy.arange(len(val)),["%.2f"% float(a) for a in par])
            pylab.title('Neuron: %d' % n)
        pylab.legend()
        

class LinePlot(Plotting):          
      """
      Plot multiple plots with common x or y axis in a row or column.
      This is a smaller helper class that mitigates some of the code repetition in such cases.
      
      Note that the inherited class has to implement:
        _subplot(self,idx,ax,params) which plots the individual plot. 
        The idx is index in whatever datastructure list we are plotting and
        axis is the axis that has to be used for plotting.
      """ 
    
    
      def  __init__(self,datastore,parameters):
           Plotting.__init__(self,datastore,parameters)    
           self.length = None
      
      def subplot(self,subplotspec,params): 
          if not self.length:
             print 'Error, class that derives from LinePlot has to specify the length parameter'
             return
          
          gs = gridspec.GridSpecFromSubplotSpec(1, self.length, subplot_spec=subplotspec)  
          for idx in xrange(0,self.length):
            p = params.copy()
            if idx > 0:
                p.setdefault("y_label",None)
            self._subplot(idx,gs[0,idx],p)


class PerStimulusPlot(Plotting):
    """
    Line plot where each plot corresponds to stimulus with the same parameter except trials.
    
    The self.dsvs will contain the datastores you want to plot in each of the subplots - i.e. all recordings
    in the given datastore come from the same stimulus of the same parameters except for the trial parameter.
    """
    required_parameters = ParameterSet({
      'sheet_name' : str,  #the name of the sheet for which to plot
    })

    def  __init__(self,datastore,parameters):
        Plotting.__init__(self,datastore,parameters)
        self.dsv = select_result_sheet_query(datastore,self.parameters.sheet_name)
        self.dsvs = partition_by_stimulus_paramter_query(self.dsv,8)    
        self.length = len(self.dsvs)

    def subplot(self,subplotspec,params): 
          if not self.length:
             print 'Error, class that derives from LinePlot has to specify the length parameter'
             return
          
          gs = gridspec.GridSpecFromSubplotSpec(1, self.length, subplot_spec=subplotspec)

          for idx in xrange(0,self.length):
                p = params.copy()
                stimulus = self.dsvs[idx].get_stimuli()[0]
                p.setdefault("title",str(stimulus))
                if idx > 0:
                        p.setdefault("y_label",None)
                self._subplot(idx,gs[0,idx],p)

            
class RasterPlot(Plotting):
      required_parameters = ParameterSet({
        'trial_averaged_histogram' : bool,  #should the plot show also the trial averaged histogram
        'neurons' : list,
      })

      def  __init__(self,datastore,parameters):
           PerStimulusPlot.__init__(self,datastore,parameters)
           if self.parameters.neurons == []:
              self.parameters.neurons = None 
      
      def subplot(self,subplotspec,params):
        PerStimulusPlot(self.datastore,function=self.ploter).make_line_plot(self,subplotspec,params)

      def ploter(self,dsv,gs,params):
         sp = [[s.spiketrains for s in self.dsv.get_segments()]]
         stimulus = self.dsv.get_stimuli()[0]
         
         if self.parameters.trial_averaged_histogram:
            gs = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs)  
            # first the raster
            ax = pylab.subplot(gs[:3,0])
            SpikeRasterPlot(sp,neurons=self.parameters.neurons,x_axis=False,x_label=None,**params.copy())(gs[:3,0])
            SpikeHistogramPlot(sp,neurons=self.parameters.neurons,**params.copy())(gs[3,0])
         else:
            SpikeRasterPlot(sp,neurons=self.parameters.neurons,**params.copy())(gs)


class VmPlot(Plotting):
      required_parameters = ParameterSet({
        'neuron' : int,  #we can only plot one neuron - which one ?
      })

      def subplot(self,subplotspec,params):
        PerStimulusPlot(self.datastore,function=self.ploter).make_line_plot(self,subplotspec,params)


      def ploter(self,dsv,gs,params):
          pylab.rc('axes', linewidth=3)
          ax = pylab.subplot(gs)
          vms = [s.get_vm() for s in self.dsv.get_segments()]
          
          mean_v = numpy.zeros(numpy.shape(vms[0][:,self.parameters.neuron]))
          
          sampling_period = vms[0].sampling_period
          time_axis = numpy.arange(0,len(vms[0]),1) /  float(len(vms[0])) * float(vms[0].t_stop) + float(vms[0].t_start)
          t_stop =  float(vms[0].t_stop - sampling_period)
          
          for vm in vms:
            ax.plot(time_axis,vm[:,self.parameters.neuron].tolist(),color="#848484")                
            mean_v = mean_v + numpy.array(vm[:,self.parameters.neuron])
          
         
          mean_v = mean_v / len(vms)
          ax.plot(time_axis,mean_v.tolist(),color='k',linewidth=2)              
          disable_top_right_axis(ax)            
          
          pylab.xlim(0,t_stop)
          pylab.xticks([0,t_stop/2,t_stop])
          three_tick_axis(ax.yaxis)
          
          if idx == 0:
            pylab.ylabel('Vm(mV)')
          else:
            pylab.ylabel('')
          pylab.rc('axes', linewidth=1)            

class GSynPlot(Plotting):

      required_parameters = ParameterSet({
        'neuron' : int  #we can only plot one neuron - which one ?
      })

      def subplot(self,subplotspec,params):
        PerStimulusPlot(self.datastore,function=self.ploter).make_line_plot(self,subplotspec,params)

      def ploter(self,dsv,gs,params):
          pylab.rc('axes', linewidth=3)
          ax = pylab.subplot(gs)
          gsyn_es = [s.get_esyn() for s in self.dsv.get_segments()]
          gsyn_is = [s.get_isyn() for s in self.dsv.get_segments()]
          mean_gsyn_e = numpy.zeros(numpy.shape(gsyn_es[0][:,self.parameters.neuron].copy()))
          mean_gsyn_i = numpy.zeros(numpy.shape(gsyn_is[0][:,self.parameters.neuron].copy()))
          sampling_period = gsyn_es[0].sampling_period
          time_axis = numpy.arange(0,len(gsyn_es[0]),1) /  float(len(gsyn_es[0])) * float(gsyn_es[0].t_stop) + float(gsyn_es[0].t_start)
          
          t_stop = float(gsyn_es[0].t_stop - sampling_period)
          for e,i in zip(gsyn_es,gsyn_is):
            e = e[:,self.parameters.neuron]*10**3 
            i = i[:,self.parameters.neuron]*10**3
            
            p1 = ax.plot(time_axis,e.tolist(),color='#F5A9A9')            
            p2 = ax.plot(time_axis,i.tolist(),color='#A9BCF5')              
            mean_gsyn_e = mean_gsyn_e + numpy.array(e.tolist())
            mean_gsyn_i = mean_gsyn_i + numpy.array(i.tolist())
         
          mean_gsyn_i = mean_gsyn_i / len(gsyn_is) 
          mean_gsyn_e = mean_gsyn_e / len(gsyn_es) 
            
          p1 = ax.plot(time_axis,mean_gsyn_e.tolist(),color='r',linewidth=2)            
          p2 = ax.plot(time_axis,mean_gsyn_i.tolist(),color='b',linewidth=2)              
          
          ax.legend([p1,p2],['exc','inh'])  
          disable_top_right_axis(ax)
          
	  pylab.xlim(0,t_stop)
          pylab.xticks([0,t_stop/2,t_stop])
          three_tick_axis(ax.yaxis)
                    
          
          pylab.rc('axes', linewidth=1)
          
class OverviewPlot(Plotting):
      required_parameters = ParameterSet({
            'sheet_name' : str,  #the name of the sheet for which to plot
            'neuron' : int,
            'sheet_activity' : ParameterSet, #if not empty the ParameterSet is passed to ActivityMovie which is displayed in to top row
      })
      
      def subplot(self,subplotspec,params):
          offset = 0 
          if len(self.parameters.sheet_activity.keys()) != 0:
             gs = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=subplotspec)      
             ActivityMovie(self.datastore,self.parameters.sheet_activity).subplot(gs[0,0],parmas)
             offset = 1 
          else:
             gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=subplotspec)      
             RasterPlot(self.datastore,ParameterSet({'sheet_name' : self.parameters.sheet_name, 'trial_averaged_histogram' : False, 'neurons' : []})).subplot(gs[0+offset,0],params.copy())
             GSynPlot(self.datastore,ParameterSet({'sheet_name' : self.parameters.sheet_name,'neuron' : self.parameters.neuron})).subplot(gs[1+offset,0],params.copy())
             VmPlot(self.datastore,ParameterSet({'sheet_name' : self.parameters.sheet_name,'neuron' : self.parameters.neuron})).subplot(gs[2+offset,0],params.copy())

          
class AnalogSignalListPlot(Plotting):
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
            
	def subplot(self,subplotspec,params):
	  LinePlot(self.datastore,function=self.ploter).make_line_plot(self,subplotspec,params)
  
	def ploter(self,idx,gs,params):
              pylab.rc('axes', linewidth=3)
              ax = pylab.subplot(gs)
              times = numpy.linspace(self.asl[idx].t_start.magnitude,self.asl[idx].t_stop.magnitude,len(self.asl[idx]))
              pylab.plot(times,self.asl[idx],color = 'b')
              if idx == 0:
                 pylab.ylabel(self.parameters.ylabel)
              
              disable_top_right_axis(ax)  
              pylab.xlim(times[0],times[-1])
              three_tick_axis(ax.yaxis)            
              three_tick_axis(ax.xaxis)                      
              pylab.rc('axes', linewidth=1)

class ConductanceSignalListPlot(Plotting):
        required_parameters = ParameterSet({
            'sheet_name' : str,  #the name of the sheet for which to plot
            'normalize_individually' : bool #each trace will be normalized individually by divding it with its maximum
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
            
        def subplot(self,subplotspec,params):
	    LinePlot(self.datastore,function=self.ploter).make_line_plot(self,subplotspec,params)

        def ploter(self,idx,gs,params):
            pylab.rc('axes', linewidth=3)
            pylab.rc('axes', linewidth=3)
	    ax = pylab.subplot(gs)
	    times = numpy.linspace(self.e_con[idx].t_start.magnitude,self.e_con[idx].t_stop.magnitude,len(self.e_con[idx]))
	    if parameters.normalize_individually:
	      pylab.plot(times,self.e_con[idx]/numpy.max(self.e_con[idx]),color = 'r',label = 'exc')
	      pylab.plot(times,self.i_con[idx]/numpy.max(self.i_con[idx]),color = 'b',label = 'inh')
	      pylab.yticks([0.0,1.0],[0,Max])
	    else:
	      pylab.plot(times,self.e_con[idx],color = 'r',label = 'exc')
	      pylab.plot(times,self.i_con[idx],color = 'b',label = 'inh')
	      
	    if idx == 0:
	       pylab.ylabel('G(S)')
	    
	    pylab.xlim(times[0],times[-1])  
	    disable_top_right_axis(ax)      
	    three_tick_axis(ax.yaxis)        
	    three_tick_axis(ax.xaxis)                      
	    pylab.rc('axes', linewidth=1)
	    
	    
class RetinalInputMovie(LinePlot):
      required_parameters = ParameterSet({
            'frame_rate' : int,  #the desired frame rate (per sec), it might be less if the computer is too slow
      })
      
    
      def  __init__(self,datastore,parameters):
           Plotting.__init__(self,datastore,parameters)    
           self.length = None
           self.retinal_input = datastore.get_retinal_stimulus()
           self.length = len(self.retinal_input)
       
      def subplot(self,subplotspec,params):
	    LinePlot(self.datastore,function=self.ploter).make_line_plot(self,subplotspec,params)

      def ploter(self,idx,gs,params):
          PixelMovie(self.retinal_input[idx],1.0/self.parameters.frame_rate*1000,x_axis=False,y_axis=False)(gs)

class ActivityMovie(Plotting):
      required_parameters = ParameterSet({
            'frame_rate' : int,  #the desired frame rate (per sec), it might be less if the computer is too slow
            'bin_width' : float, # in ms the width of the bins into which to sample spikes 
            'scatter' :  bool,  # whether to plot neurons activity into a scatter plot (if True) or as an interpolated pixel image
            'resolution' : int, # the number of pixels into which the activity will be interpolated in case scatter = True
      })
      

      def  __init__(self,datastore,parameters):
           PerStimulusPlot.__init__(self,datastore,parameters)
    
      def subplot(self,subplotspec,params):
        PerStimulusPlot(self.datastore,function=self.ploter).make_line_plot(self,subplotspec,params)

      def ploter(self,dsv,gs,params):
         sp = [s.spiketrains for s in self.dsvs[idx].get_segments()]
         
         start = sp[0][0].t_start.magnitude
         stop = sp[0][0].t_stop.magnitude
         units = sp[0][0].t_start.units
         bw = self.parameters.bin_width * pq.ms
         bw = bw.rescale(units).magnitude
         bins = numpy.arange(start,stop,bw)
             
         
         h = []
         for spike_trains in sp:
             hh = []
             for st in spike_trains:
                hh.append(numpy.histogram(st.magnitude,bins, (start,stop))[0])
             h.append(numpy.array(hh))
         h = numpy.sum(h,axis=0)

         pos = self.dsvs[0].get_neuron_postions()[self.parameters.sheet_name]
         
         if not self.parameters.scatter:
             xi = numpy.linspace(numpy.min(pos[0])*1.1,numpy.max(pos[0])*1.1,self.parameters.resolution)
             yi = numpy.linspace(numpy.min(pos[1])*1.1,numpy.max(pos[1])*1.1,self.parameters.resolution)
      
             movie = []
             for i in xrange(0,numpy.shape(h)[1]):
                 movie.append(griddata((pos[0], pos[1]), h[:,i], (xi[None,:], yi[:,None]), method='cubic'))
                
             PixelMovie(movie,1.0/self.parameters.frame_rate*1000,x_axis=False,y_axis=False)(gs)
         else:
             ScatterPlotMovie(pos[0],pos[1],h.T,1.0/self.parameters.frame_rate*1000,x_axis=False,y_axis=False, dot_size = 40)(gs)

class PerNeuronValuePlot(Plotting):
    """
    Plots PerNeuronValuePlots, one for each sheet.
    
    #JAHACK, so far doesn't support the situation where several types of PerNeuronValue analysys data structures are present in the
    supplied datastore.
    """
    
    def  __init__(self,datastore,parameters):
        Plotting.__init__(self,datastore,parameters)
        self.poss = []
        self.pnvs = []
        for sheet in datastore.sheets():
            z = datastore.get_analysis_result('PerNeuronValue',sheet_name=sheet)
            if len(z) != 0:
                self.poss.append(datastore.get_neuron_postions()[sheet])
                self.pnvs.append(z)
        self.length = len(self.poss)

    def subplot(self,subplotspec,params):
	    LinePlot(self.datastore,function=self.ploter).make_line_plot(self,subplotspec,params)

    def ploter(self,idx,gs,params):
         posx = self.poss[idx][0]
         posy = self.poss[idx][1]
         values = self.pnvs[idx][0].values
         if self.pnvs[idx][0].period != None:
            periodic = True
            period = self.pnvs[idx][0].period
         else:
            periodic = False
            period = None
                    
         params.setdefault("x_label",'x')
         params.setdefault("y_label",'y')
         params.setdefault("title",self.pnvs[idx][0].value_name)
         params.setdefault("colorbar_label",self.pnvs[idx][0].value_units.dimensionality.latex)
         if periodic:
            if idx ==self.length-1:
                params.setdefault("colorbar",True)
         else:
            params.setdefault("colorbar",True)  
         ScatterPlot(posx,posy,values,periodic=periodic,period=period,**params)(gs)

	 
	 
	 	
