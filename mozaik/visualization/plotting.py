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
    
    def plot(self,params=None):
        if params == None:
           params = {}
        self.fig = pylab.figure(facecolor='w')
        gs = gridspec.GridSpec(1, 1)
        gs.update(left=0.05, right=0.95, top=0.9, bottom=0.1)
        self.subplot(gs[0,0],params)
        

          
              
class PlotTuningCurve(Plotting):
    """
    values - contain a list of lists of values, the outer list corresponding
    to stimuli the inner to neurons.
    stimuli_ids - contain list of stimuli ids corresponding to the values
    parameter_index - corresponds to the parameter that should be plotted as a tuning curve
    """

    required_parameters = ParameterSet({
      'tuning_curve_name' : str,  #the name of the tuning curve
      'neuron': int, # which neuron to plot
      'sheet_name' : str, # from which layer to plot the tuning curve
    })

    def  __init__(self,datastore,parameters):
        Plotting.__init__(self,datastore,parameters)
        self.tuning_curves = self.datastore.get_analysis_result(parameters.tuning_curve_name,sheet_name=parameters.sheet_name)
    
    def subplot(self,subplotspec,params):
      LinePlot(function=self.ploter,length = len(self.tuning_curves)).make_line_plot(subplotspec,params)
    
    def ploter(self,idx,gs,params):
        tc = self.tuning_curves[idx]
        tc = tc.to_dictonary_of_tc_parametrization()
        xs = []
        ys = []
        labels = []
        for k in  tc:
            (a,b) = tc[k]
            par,val = zip(*sorted(zip(b,a[:,self.parameters.neuron])))
            xs.append(par)
            ys.append(val)
            labels.append(fromat_stimulus_id(parse_stimuls_id(k)))
        
        params.setdefault("title",('Neuron: %d' % self.parameters.neuron))
        params.setdefault("y_label",self.tuning_curves[idx].y_axis_name)
        params.setdefault("x_lim",(xs[0],xs[-1]))
        StandardStyleLinePlot(xs,ys,labels=labels,**params)(subplotspec)
            
            
class CyclicTuningCurvePlot(PlotTuningCurve):
    """
    Tuning curve over cyclic domain
    """
    
    def ploter(self,idx,gs,params):
        tc = self.tuning_curves[idx]
        tc = tc.to_dictonary_of_tc_parametrization()
        xs = []
        ys = []
        labels = []
        for k in  tc:
            (a,b) = tc[k]
            par,val = zip(*sorted(zip(b,a[:,self.parameters.neuron])))
            par = list(par)
            val = list(val)
            par.append(par[0]+self.tuning_curves[0].period)
            val.append(val[0])
            xs.append(numpy.array(par))
            ys.append(numpy.array(val))
            labels.append(fromat_stimulus_id(parse_stimuls_id(k)))

        params.setdefault("title",('Neuron: %d' % self.parameters.neuron))
        params.setdefault("y_label",self.tuning_curves[idx].y_axis_name)
        
        if self.tuning_curves[0].period == numpy.pi:
            params.setdefault("x_ticks",[0,numpy.pi/2.0,numpy.pi])
            params.setdefault("x_lim",(0,numpy.pi))
            params.setdefault("x_tick_style","Custom")
            params.setdefault("x_tick_labels",["0","$\\frac{\\pi}{2}$","$\\pi$"])
        if self.tuning_curves[0].period == 2*numpy.pi:
            params.setdefault("x_ticks",[0,numpy.pi,2*numpy.pi])
            params.setdefault("x_lim",(0,2*numpy.pi))
            params.setdefault("x_tick_labels",["0","$\\pi$","$2\\pi$"])
            params.setdefault("x_tick_style","Custom")

        StandardStyleLinePlot(xs,ys,labels=labels,**params)(subplotspec)

class RasterPlot(Plotting):
      required_parameters = ParameterSet({
        'trial_averaged_histogram' : bool,  #should the plot show also the trial averaged histogram
        'neurons' : list,
        'sheet_name' : str,
      })

      def  __init__(self,datastore,parameters):
           Plotting.__init__(self,datastore,parameters)
           if self.parameters.neurons == []:
              self.parameters.neurons = None 
      
      def subplot(self,subplotspec,params):
          dsv = select_result_sheet_query(self.datastore,self.parameters.sheet_name)
          PerStimulusPlot(dsv,function=self.ploter).make_line_plot(subplotspec,params)

      def ploter(self,dsv,gs,params):
         sp = [[s.spiketrains for s in dsv.get_segments()]]
         stimulus = dsv.get_stimuli()[0]
         
         if self.parameters.trial_averaged_histogram:
            gs = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs)  
            # first the raster
            SpikeRasterPlot(sp,neurons=self.parameters.neurons,x_axis=False,x_label=None,**params.copy())(gs[:3,0])
            SpikeHistogramPlot(sp,neurons=self.parameters.neurons,**params.copy())(gs[3,0])
         else:
            SpikeRasterPlot(sp,neurons=self.parameters.neurons,**params.copy())(gs)


class VmPlot(Plotting):
      
      required_parameters = ParameterSet({
        'neuron' : int,  #we can only plot one neuron - which one ?
        'sheet_name' : str,
      })

      def subplot(self,subplotspec,params):
        dsv = select_result_sheet_query(self.datastore,self.parameters.sheet_name)
        PerStimulusPlot(dsv,function=self.ploter).make_line_plot(subplotspec,params)


      def ploter(self,dsv,gs,params):
          vms = [s.get_vm() for s in dsv.get_segments()]         
          sampling_period = vms[0].sampling_period
          time_axis = numpy.arange(0,len(vms[0]),1) /  float(len(vms[0])) * float(vms[0].t_stop) + float(vms[0].t_start)
          t_stop =  float(vms[0].t_stop - sampling_period)
            
          xs = []
          ys = []
          colors = []
          for vm in vms:
               xs.append(time_axis)
               ys.append(numpy.array(vm[:,self.parameters.neuron].tolist()))
               colors.append("#848484")
            
          params.setdefault("x_lim",(0,t_stop))
          params.setdefault("x_ticks",[0,t_stop/2,t_stop])
          params.setdefault("x_label",'time(' + vms[0].t_stop.dimensionality.latex  + ')')
          params.setdefault("y_label",'Vm(' + vms[0].dimensionality.latex +  ')')
          StandardStyleLinePlot(xs,ys,colors=colors,mean=True,**params)(gs)
        
        

class GSynPlot(Plotting):

      required_parameters = ParameterSet({
        'neuron' : int,  #we can only plot one neuron - which one ?
        'sheet_name' : str,
      })

      def subplot(self,subplotspec,params):
        dsv = select_result_sheet_query(self.datastore,self.parameters.sheet_name)
        PerStimulusPlot(dsv,function=self.ploter).make_line_plot(subplotspec,params)

      def ploter(self,dsv,gs,params):
          exc =[]
          inh =[]
          gsyn_es = [s.get_esyn() for s in dsv.get_segments()]
          gsyn_is = [s.get_isyn() for s in dsv.get_segments()]

          for e,i in zip(gsyn_es,gsyn_is):
              exc.append(e[:,self.parameters.neuron])
              inh.append(i[:,self.parameters.neuron])
          ConductancesPlot(exc,inh,**params)(gs)

          
          
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
                  
            gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=subplotspec)      
            p = params.copy()
            if offset == 1:
               p.setdefault('title',None)
            p.setdefault('x_axis',False)
            p.setdefault('x_label',False)
            RasterPlot(self.datastore,ParameterSet({'sheet_name' : self.parameters.sheet_name, 'trial_averaged_histogram' : False, 'neurons' : []})).subplot(gs[0+offset,0],p)
            
            p = params.copy()
            p.setdefault('x_axis',False)
            p.setdefault('x_label',False)
            p.setdefault('title',None)
            GSynPlot(self.datastore,ParameterSet({'sheet_name' : self.parameters.sheet_name,'neuron' : self.parameters.neuron})).subplot(gs[1+offset,0],p)
            
            p = params.copy()
            p.setdefault('title',None)
            VmPlot(self.datastore,ParameterSet({'sheet_name' : self.parameters.sheet_name,'neuron' : self.parameters.neuron})).subplot(gs[2+offset,0],p)

          
class AnalogSignalListPlot(Plotting):
        required_parameters = ParameterSet({
            'sheet_name' : str,  #the name of the sheet for which to plot
            'ylabel' : str,  #what to put as ylabel
        })
    
        def  __init__(self,datastore,parameters):
            Plotting.__init__(self,datastore,parameters)
            self.analog_signal_list = self.datastore.get_analysis_result('AnalogSignalList',sheet_name = parameters.sheet_name)    
            if len(self.analog_signal_list) > 1:
              print 'ERROR: Warning currently only the first AnalogSignalList will be plotted'
            self.analog_signal_list = self.analog_signal_list[0]
            self.asl = self.analog_signal_list.asl
        
        
        def subplot(self,subplotspec,params):
              xs = []
              ys = []
              colors = []
              for a in self.asl:
                   times = numpy.linspace(a.t_start.magnitude,a.t_stop.magnitude,len(a))
                   xs.append(times)
                   ys.append(a)
                   colors.append("#848484")
                
              params.setdefault("x_lim",(a.t_start.magnitude,a.t_stop.magnitude))
              params.setdefault("x_label",self.analog_signal_list.x_axis_name)
              params.setdefault("y_label",self.analog_signal_list.y_axis_name)
              params.setdefault("x_ticks",[a.t_start.magnitude,a.t_stop.magnitude])
              params.setdefault("mean",True)
              StandardStyleLinePlot(xs,ys,colors=colors,**params)(subplotspec)


 
class ConductanceSignalListPlot(Plotting):
        required_parameters = ParameterSet({
            'sheet_name' : str,  #the name of the sheet for which to plot
            'normalize_individually' : bool #each trace will be normalized individually by divding it with its maximum
        })
    
        def  __init__(self,datastore,parameters):
            Plotting.__init__(self,datastore,parameters)
            self.conductance_signal_list = self.datastore.get_analysis_result('ConductanceSignalList',sheet_name = parameters.sheet_name)    
            if len(self.conductance_signal_list) > 1:
              print 'ERROR: Warning currently only the first ConductanceSignalList will be plotted'
            self.conductance_signal_list = self.conductance_signal_list[0]
            self.e_con = self.conductance_signal_list.e_con
            self.i_con = self.conductance_signal_list.i_con
            
        def subplot(self,subplotspec,params):
            exc =[]
            inh =[]
            print 'DASDASDA'
            print len(self.e_con)
            print len(self.i_con)
            
            for e,i in zip(self.e_con,self.i_con):
              exc.append(e)
              inh.append(i)
            ConductancesPlot(exc,inh,**params)(subplotspec)
        
        
class RetinalInputMovie(Plotting):
      required_parameters = ParameterSet({
            'frame_rate' : int,  #the desired frame rate (per sec), it might be less if the computer is too slow
      })
      
    
      def  __init__(self,datastore,parameters):
           Plotting.__init__(self,datastore,parameters)    
           self.length = None
           self.retinal_input = datastore.get_retinal_stimulus()
           
      def subplot(self,subplotspec,params):
          LinePlot(function=self.ploter,length = len(self.retinal_input)).make_line_plot(subplotspec,params)

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
           Plotting.__init__(self,datastore,parameters)
    
      def subplot(self,subplotspec,params):
        PerStimulusPlot(self.datastore,function=self.ploter).make_line_plot(subplotspec,params)

      def ploter(self,dsv,gs,params):
         sp = [s.spiketrains for s in dsvs.get_segments()]
         
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

         pos = dsvs[0].get_neuron_postions()[self.parameters.sheet_name]
         
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
        self.length=len(self.poss)
 
    def subplot(self,subplotspec,params):
        LinePlot(function=self.ploter,length=self.length).make_line_plot(subplotspec,params)

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

     
     
        
