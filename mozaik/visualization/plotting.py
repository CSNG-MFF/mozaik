"""
This module defines the :class:`mozaik.visualization.plotting.Plotting` API and offers
a library of plotting commands derived from it. 

The plotting API
----------------

The high-level plotting API is defined by the :class:`mozaik.visualization.plotting.Plotting` class.
Each  class derived from `Plotting` should implement two plotting functions:  plot and subplot.
The subplot function accepts a SubplotSpec object (see matplotlib doc) as input which 
will tell it where to plot. The subplot function returns a dictionary, where each key
is a name of a subplot it generates (and will correspond to a creation of another Plotting or 
SimplePlot plot), and the associated value is a tuple that contains the:
    * Plotting or SimplePlot object
    * the SubplotSpec subregion into which the plot is supposed to be drawn
    * dictionary of parameters that to be passed to the plot.
This way one can nest Plotting instances and eventualy simple_plot instances as the leafs of the tree.

The third element of the tuple discussed above is a dictionary of parameters that will be  passed onto the eventual call to SimplePlot class.
At each level of hierarchy the Plotting instances can add some parameters via the third element of the tuple. But note that the parameters at 
higher levels of hierarchy will overide the parameters given at lower level of hierarchy. 

The names of the subplots that 
are returned by the subplot function as the keys of the dictionary should be documented as they will be used 
to identify the subplots by user should he wish to pass specific parameters to them. The user can do this by 
passing a dictionary to the plot function. The keys of the dictionary are comma sepparated strings containing the 
path to the plot in the plotting hierarchy to which the parameters will be passed, e.g: top_plot_name.first_level_subplot_name.second_level_subplot_name.parameter_name = value
One can also use '*' instead of a plot name to indicated that the parameter should be passed to all the subplots at that level. Also if the path
ends before it reaches a simple_plot level it is automatically assumed the parameter will be passed into the rest of the subtree.
Finaly the subplots function can return plots which are names which use the '.' character. This will be processed in the same way as above
and it allows for Plotting classes that create multiple groups of plots to let user sepparately target the different groups with parameters.

The plot function can either not be defined in which case it defaults to the
Plotting.plot, which simply creates a figure and calls subplot with SuplotSpec
spanning the whole figure. Alternatively, one can define the plot function if
one wants to add some additional figure level decorations, in case the figure is plotted on
its own (i.e. becomes the highest-level), and that would otherwise prevent
flexible use in nesting via the subplot.
"""

import pylab
import numpy
import time
import quantities as pq
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata

from parameters import ParameterSet

from mozaik.core import ParametrizedObject
from mozaik.storage import queries
from mozaik.controller import Global
from mozaik.tools.mozaik_parametrized import colapse_to_dictionary, MozaikParametrized, varying_parameters, matching_parametrized_object_params
from numpy import pi
from neo.core.analogsignal import AnalogSignal as NeoAnalogSignal
from neo.core.spiketrain import SpikeTrain as NeoSpikeTrain
from simple_plot import StandardStyleLinePlot, SpikeRasterPlot, \
                        SpikeHistogramPlot, ConductancesPlot, PixelMovie, \
                        ScatterPlotMovie, ScatterPlot, ConnectionPlot, SimplePlot, HistogramPlot
from plot_constructors import LinePlot, PerStimulusPlot, PerStimulusADSPlot, ADSGridPlot

import mozaik
logger = mozaik.getMozaikLogger()



class Plotting(ParametrizedObject):
    """
    The high level Plotting API. See the module information on more detailed description.

    Parameters
    ----------
    
    datastore : Datastore
              the DataStore from which to plot the data
              
    parameters : ParameterSet
               The mozaik parameters.
               
    plot_file_name : str
                   Either None, in which case the plot is not saved onto
                   HD, or path to a file into which to save the file (formats
                   accepted by matplotlib).
                   
    fig_params : dict
               The parameters that are passed to the matplotlib figure
               command (but note facecolor='w' is already supplied).
    """

    def  __init__(self, datastore, parameters, plot_file_name=None,fig_param=None,frame_duration=0):
        ParametrizedObject.__init__(self, parameters)
        self.datastore = datastore
        self.plot_file_name = plot_file_name
        self.animation_update_functions = []
        self.frame_duration = frame_duration
        self.fig_param = fig_param if fig_param != None else {}

    def subplot(self, subplotspec):
        """
        This is the function that each Plotting instance has to implement.
        See the module documentation for more details.
        """
        raise NotImplementedError
    
    def _nip_parameters(self,n,p):
        d = {}
        fd = {}
        for (k,v) in p.iteritems():
            l = k.split('.')
            assert len(l) > 1, "Parameter %s not matching the simple plot" % (k)
            if l[0] == n or l[0] == '*':
                if len(l[1:]) >1: 
                   d['.'.join(l[1:])] = v
                else:
                   fd[l[1]] = v
                
        return d,fd
    
    def _handle_parameters_and_execute_plots(self,parameters,user_parameters,gs):
        d = self.subplot(gs)
        for (k,(pl,gs,p)) in d.iteritems():
            p.update(parameters)
            ### THIS IS WRONG 'UP' DO NOT WORK        
            up = user_parameters
            for z in k.split('.'): 
                up,fp = self._nip_parameters(z,up)
                p.update(fp)
            param = p
            if isinstance(pl,SimplePlot):
                # print check whether all user_parameters have been nipped to minimum 
                pl(gs,param,self)
            elif isinstance(pl,Plotting):
                pl._handle_parameters_and_execute_plots(param,up,gs)     
            else:
               raise TypeError("The subplot object is not of type Plotting or SimplePlot") 
    
    def plot(self, params=None):
        """
        The top level plot function. It is this function that the user will call on the plot instance to 
        execute the plotting.
        
        Parameters
        ----------
        
        params : dict
               The dictionary of parameters modifying the defaults across the plotting hierarchy.
               Keys are comma sepparated path in the plot hierarchy, values are values to be substituted. 
               See the module level info for more details. 
        """
        t1 = time.time()
        if params == None:
            params = {}
        self.fig = pylab.figure(facecolor='b', **self.fig_param)
        gs = gridspec.GridSpec(1, 1)
        gs.update(left=0.05, right=0.95, top=0.95, bottom=0.05)
        self._handle_parameters_and_execute_plots({}, params,gs[0, 0])

        # ANIMATION Handling
        if self.animation_update_functions != []:
          import matplotlib.animation as animation
          self.animation = animation.FuncAnimation(self.fig,
                                      Plotting.update_animation_function,
                                      frames = 400,
                                      repeat=False,
                                      fargs=(self,),
                                      interval=self.frame_duration,
                                      blit=False,save_count=0)
        gs.tight_layout(self.fig)
        if self.plot_file_name:
            #if there were animations, save them
            if self.animation_update_functions != []:
                self.animation.save(Global.root_directory+self.plot_file_name+'.mov', writer='avconv', fps=10,bitrate=5000) 
            else:
                # save the analysis plot
                pylab.savefig(Global.root_directory+self.plot_file_name)              
        t2 = time.time()
        logger.warning(self.__class__.__name__ + ' plotting took: ' + str(t2 - t1) + 'seconds')

    def register_animation_update_function(self,auf,parent):
        self.animation_update_functions.append((auf,parent))

    @staticmethod
    def update_animation_function(b,self):
        for auf,parent in self.animation_update_functions:
            auf(parent)



class PlotTuningCurve(Plotting):
    """
    Plots tuning curves, one plot in line per each neuron. This plotting function assumes a set of PerNeuronValue 
    ADSs in the datastore associated with certain stimulus type. It will plot
    the values stored in these  PerNeuronValue instances (corresponding to neurons in `neurons`) across the 
    varying parameter `parameter_name` of thier associated stimuli. If other parameters of the 
    stimuli are varying withing the datastore it will automatically plot one tuning curve per each combination
    of values of the other parameters and label the curve accordingly.
    
    Other parameters
    ----------------
    neurons : list
            List of neuron ids for which to plot the tuning curves.
            
    sheet_name : str
               From which layer to plot the tuning curves.
               
    parameter_name : str
                   The parameter_name through which to plot the tuning curve.
                   
    centered : bool
             If True it will center each set of tuning curves on the parameter value with the larges mean response across the other parameter variations
    
    mean : bool 
         If True it will plot the mean tuning curve over all neurons (in case centered=True it will first center the TCs before computing the mean)
    
    pool : bool
         If True it will not plot each different value_name found in datastore on a sepparete line of plots but pool them together.             
    
    polar : bool
          If True it will plot the tuning curves in polar coordinates, not that the stimulus parameter through which the tuning curves are  plotted has to be periodic, and this period will be mapped on to the (0,360) degrees interval of the polar plot.
            
    Defines 'TuningCurve_' + value_name +  '.Plot0' ... 'TuningCurve_' + value_name +  '.Plotn'
    where n goes through number of neurons, and value_name creates one row for each value_name found in the different PerNeuron found
    """

    required_parameters = ParameterSet({
      'neurons':  list,  # which neurons to plot
      'sheet_name': str,  # from which layer to plot the tuning curves
      'parameter_name': str,  # the parameter_name through which to plot the tuning curve
      'centered' : bool, # if True it will center each set of tuning curves on the parameter value with the larges mean response across the other parameter variations
      'mean' : bool, # if True it will plot the mean tuning curve over the neurons (in case centered=True it will first center the TCs before computing the mean)
      'pool' : bool, # if True it will not plot each different value_name found in datastore on a sepparete line of plots but pool them together.
      'polar' : bool # if True polar coordinates will be used
    })

    def __init__(self, datastore, parameters, plot_file_name=None,fig_param=None,frame_duration=0):
        Plotting.__init__(self, datastore, parameters, plot_file_name, fig_param,frame_duration)
        
        self.st = []
        self.tc_dict = []
        self.pnvs = []
        self.max_mean_response_indexes = []
        assert queries.ads_with_equal_stimulus_type(datastore)
        assert len(self.parameters.neurons) > 0 , "ERROR, empty list of neurons specified"
        #if self.parameters.mean:
        #    assert self.parameters.centered , "Average tuning curve can be plotted only if the tuning curves are centerd"
        
        dsvs = queries.partition_analysis_results_by_parameters_query(self.datastore,parameter_list=['value_name'],excpt=True)
        for dsv in dsvs:
            dsv = queries.param_filter_query(dsv,identifier='PerNeuronValue',sheet_name=self.parameters.sheet_name)
            assert matching_parametrized_object_params(dsv.get_analysis_result(), params=['value_name'])
            self.pnvs.append(dsv.get_analysis_result())
            # get stimuli
            st = [MozaikParametrized.idd(s.stimulus_id) for s in self.pnvs[-1]]
            self.st.append(st)
            
            dic = colapse_to_dictionary([z.get_value_by_id(self.parameters.neurons) for z in self.pnvs[-1]],st,self.parameters.parameter_name)
            #sort the entries in dict according to the parameter parameter_name values 
            for k in  dic:
                (b, a) = dic[k]
                par, val = zip(
                             *sorted(
                                zip(b,
                                    numpy.array(a))))
                dic[k] = (par,numpy.array(val))
            self.tc_dict.append(dic)
            if self.parameters.centered:
               self.max_mean_response_indexes.append(numpy.argmax(sum([a[1] for a in dic.values()]),axis=0))
               # lets find the highest average value for the neuron
        
        if self.parameters.pool:
           assert all([p[0].value_units == self.pnvs[0][0].value_units for p in self.pnvs]), "You asked to pool tuning curves across different value_names, but the datastore contains PerNeuronValue datastructures with different units"
            
            
    def subplot(self, subplotspec):
        if self.parameters.mean:
            return LinePlot(function=self._ploter, length=1).make_line_plot(subplotspec)
        else:
            return LinePlot(function=self._ploter, length=len(self.parameters.neurons)).make_line_plot(subplotspec)

    def _ploter(self, idx, gs):
        plots  = []
        xs = []
        labels=[]
        ys = []
        
        po = None if not self.parameters.polar else {'projection':'polar'}
        
        if not self.parameters.pool:
            gs = gridspec.GridSpecFromSubplotSpec(len(self.st), 1, subplot_spec=gs)
        
        for i,(dic, st, pnv) in enumerate(zip(self.tc_dict,self.st,self.pnvs)):
            if not self.parameters.pool:
               xs = [] 
               ys = []
               labels = []
                            
            period = st[0].params()[self.parameters.parameter_name].period
            if self.parameters.centered:        
               assert period != None, "ERROR: You asked for centering of tuning curves even though the domain over which it is measured is not periodic." 
            
            if self.parameters.polar:
               assert period != None, "ERROR: You asked to plot the tuning curve on polar axis even though the domain over which it is measured is not periodic." 
                
            for k in dic.keys():    
                (par, val) = dic[k]
                if self.parameters.mean:
                    v = 0
                    for j in xrange(0,len(self.parameters.neurons)):
                        if self.parameters.centered:
                            vv,p = self.center_tc(val[:,j],par,period,self.max_mean_response_indexes[i][j])
                        else:
                            vv = val[:,j]
                            p = par
                        v = v + vv
                    val = v / len(self.parameters.neurons)
                    par = p
                elif self.parameters.centered:
                    val,par = self.center_tc(val[:,idx],par,period,self.max_mean_response_indexes[i][idx])
                else:
                    val = val[:,idx]
                    
                    
                if period != None:
                    par = list(par)
                    val = list(val)
                    par.append(par[0] + period)
                    val.append(val[0])

                if self.parameters.polar:
                   # we have to map the interval (0,period)  to (0,2*pi)
                   par = [p/period*2*numpy.pi for p in par]

              
                xs.append(numpy.array(par))
                ys.append(numpy.array(val))
                
                l = ""
                
                if self.parameters.pool:
                   l = pnv[0].value_name + " "
                
                for p in varying_parameters([MozaikParametrized.idd(e) for e in dic.keys()]):
                    l = l + str(p) + " : " + str(getattr(MozaikParametrized.idd(k),p))
                labels.append(l)

            if not self.parameters.pool:
                params = self.create_params(pnv[0].value_name,pnv[0].value_units,i==0,i==(len(self.pnvs)-1),period,self.parameters.neurons[idx],len(xs),self.parameters.polar,labels,idx)
                plots.append(("TuningCurve_" + pnv[0].value_name,StandardStyleLinePlot(xs, ys,subplot_kw=po),gs[i],params))   

        if self.parameters.pool:
           params = self.create_params('mix',self.pnvs[0][0].value_units,True,True,period,self.parameters.neurons[idx],len(xs),self.parameters.polar,labels,idx)
           if not self.parameters.polar:
              plots.append(("TuningCurve_Stacked",StandardStyleLinePlot(xs, ys),gs,params))
           else:
              plots.append(("TuningCurve_Stacked",StandardStyleLinePlot(xs, ys,subplot_kw=po),gs,params)) 
                
        return plots

    def create_params(self,value_name,units,top_row,bottom_row,period,neuron_id,number_of_curves,polar,labels,idx):
            params={}
            
            params["x_label"] = self.parameters.parameter_name
            if idx == 0:
                params["y_label"] = value_name + '(' + units.dimensionality.latex + ')'
                
            params['labels']=labels
            params['linewidth'] = 2
            params['colors'] = [cm.jet(j/float(number_of_curves)) for j in xrange(0,number_of_curves)] 
            
            if top_row:
                params["title"] =  'Neuron ID: %d' % neuron_id
            
            if not polar:
                if self.parameters.centered:        
                    if period == pi:
                        params["x_ticks"] = [-pi/2, 0, pi/2]
                        params["x_lim"] = (-pi/2, pi/2)
                        params["x_tick_style"] = "Custom"
                        params["x_tick_labels"] = ["-$\\frac{\\pi}{2}$", "0", "$\\frac{\\pi}{2}$"]
                   
                    if period == 2*pi:
                        params["x_ticks"] = [-pi, 0, pi]
                        params["x_lim"] = (-pi, pi)
                        params["x_tick_style"] = "Custom"
                        params["x_tick_labels"] = ["-$\\pi$","0", "$\\pi$"]
                else:
                    if period == pi:
                        params["x_ticks"] = [0, pi/2, pi]
                        params["x_lim"] = (0, pi)
                        params["x_tick_style"] = "Custom"
                        params["x_tick_labels"] = ["0", "$\\frac{\\pi}{2}$", "$\\pi$"]
                   
                    if period == 2*pi:
                        params["x_ticks"] = [0, pi, 2*pi]
                        params["x_lim"] = (0, 2*pi)
                        params["x_tick_style"] = "Custom"
                        params["x_tick_labels"] = ["0", "$\\pi$", "$2\\pi$"]
            
            else:
               params["y_tick_style"] = "Custom"
               params["x_tick_style"] = "Custom"
               params["x_ticks"]  = []
               params["y_ticks"]  = []
               params["x_tick_labels"]  = []
               params["y_tick_labels"]  = []
               params['grid'] = True
               params['fill'] = False
            
            if not bottom_row:
                params["x_axis"] = None
                
            return params
            
    def center_tc(self,val,par,period,center_index):
           # first lets make the maximum to be at zero                   
           q = center_index+len(val)/2 if center_index < len(val)/2 else center_index-len(val)/2
           z = par[center_index]
           c =  period/2.0
           par = numpy.array([(p - z + c) % period for p in par])  - c
           if q != 0:
               a = val[:q].copy()[:] 
               b = par[:q].copy()[:] 
               val[:-q] = val[q:].copy()[:]   
               par[:-q] = par[q:].copy()[:]   
               val[-q:] = a
               par[-q:] = b
           
           return val,par 
        

    
class RasterPlot(Plotting):
    """ 
    It plots raster plots of spikes stored in the recordings.
    It assumes a datastore with a set of recordings. It will plot a line of raster 
    plots, one per each recording, showing the raster plot corresponding to the given 
    recording.
    
    Defines 'RasterPlot.Plot0' ... 'RasterPlot.PlotN'
    
    Other parameters
    ----------------
    
    sheet_name : str
               From which layer to plot the raster plots.
               
    neurons : list
            List of neuron ids for which to plot the tuning curves.
    
    trial_averaged_histogram : bool
                             Should the plot show also the trial averaged histogram?
    
    spontaneous : bool
                Whether to also show the spontaneous activity the preceded the stimulus.
    """
    
    required_parameters = ParameterSet({
        'trial_averaged_histogram': bool,  # should the plot show also the trial averaged histogram
        'neurons': list,
        'sheet_name': str,
        'spontaneous' : bool, # whether to also show the spontaneous activity the preceded the stimulus         
    })

    def subplot(self, subplotspec):
        dsv = queries.param_filter_query(self.datastore,sheet_name=self.parameters.sheet_name)
        return PerStimulusPlot(dsv, function=self._ploter).make_line_plot(subplotspec)

    def _ploter(self, dsv,gs):
        sp = [s.get_spiketrain(self.parameters.neurons) for s in sorted(dsv.get_segments(),key = lambda x : MozaikParametrized.idd(x.annotations['stimulus']).trial)]             
        
        x_ticks = [0.0,float(sp[0][0].t_stop/2), float(sp[0][0].t_stop)]
        
        if self.parameters.spontaneous:
           spont_sp = [s.get_spiketrain(self.parameters.neurons) for s in sorted(dsv.get_segments(null=True),key = lambda x : MozaikParametrized.idd(x.annotations['stimulus']).trial)]             
           sp = [RasterPlot.concat_spiketrains(sp1,sp2) for sp1,sp2 in zip(spont_sp,sp)]
           x_ticks = [float(spont_sp[0][0].t_start),0.0,float(sp[0][0].t_stop/2), float(sp[0][0].t_stop)]

        d = {} 
        if self.parameters.trial_averaged_histogram:
            gs = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs)
            # first the raster
            return [ ('SpikeRasterPlot',SpikeRasterPlot([sp]),gs[:3, 0],{'x_axis': False , 'x_label' :  None,"x_ticks": x_ticks}),
                     ('SpikeHistogramPlot',SpikeHistogramPlot([sp]),gs[3, 0],{"x_ticks": x_ticks})]
        else:
            return [('SpikeRasterPlot',SpikeRasterPlot([sp]),gs,{"x_ticks": x_ticks})]


    @staticmethod
    def concat_spiketrains(ssp1,ssp2):
        l = []
        for sp1,sp2 in zip(ssp1,ssp2):
            assert sp1.units == sp2.units
            assert sp2.t_start == sp1.t_start == 0
            assert sp1.units == sp1.t_stop.units == sp1.t_start.units
            assert sp2.units == sp2.t_stop.units == sp2.t_start.units
            
            l.append(NeoSpikeTrain(numpy.concatenate((sp1.magnitude-sp1.t_stop.magnitude,sp2.magnitude)),t_start=-sp1.t_stop,t_stop=sp2.t_stop,units=sp1.units))
        return l
        
class VmPlot(Plotting):
    """
    It plots the membrane potentials stored in the recordings.
    It assumes a datastore with a set of recordings. It will plot a line of vm
    plots, one per each recording, showing the vm corresponding to the given 
    recording.
    
    It defines one plot named: 'VmPlot.Plot0' ... 'VmPlot.PlotN'.
    
    Other parameters
    ----------------
    
    sheet_name : str
               From which layer to plot the vms.
               
    neuron : int
            Id of the neuron to plot.
            
    spontaneous : bool
                Whether to also show the spontaneous activity the preceded the stimulus.
    """

    required_parameters = ParameterSet({
      'neuron': int,  # we can only plot one neuron - which one ?
      'sheet_name': str,
      'spontaneous' : bool, # whether to also show the spontaneous activity the preceded the stimulus
    })

    def subplot(self, subplotspec):
        dsv = queries.param_filter_query(self.datastore,sheet_name=self.parameters.sheet_name)
        return PerStimulusPlot(dsv, function=self._ploter, title_style="Clever"
                                         ).make_line_plot(subplotspec)

    def _ploter(self, dsv,gs):
        vms = [s.get_vm(self.parameters.neuron) for s in sorted(dsv.get_segments(),key = lambda x : MozaikParametrized.idd(x.annotations['stimulus']).trial)]
        time_axis = numpy.arange(0, len(vms[0]), 1) / float(len(vms[0])) * float(vms[0].t_stop) + float(vms[0].t_start)
        t_stop = float(vms[0].t_stop - vms[0].sampling_period)
        t_start = 0
        x_ticks = [t_start, t_stop/2, t_stop]
        
        if self.parameters.spontaneous:
           spont_vms = [s.get_vm(self.parameters.neuron) for s in sorted(dsv.get_segments(null=True),key = lambda x : MozaikParametrized.idd(x.annotations['stimulus']).trial)]             
           t_start = - float(spont_vms[0].t_stop)
           x_ticks = [t_start,0.0,t_stop/2, t_stop]
           spont_time_axis = numpy.arange(0, len(spont_vms[0]), 1) / float(len(spont_vms[0])) * float(spont_vms[0].t_stop) - float(spont_vms[0].t_stop) + float(vms[0].t_start)
           time_axis = numpy.concatenate((spont_time_axis,time_axis))
           vms1 = [numpy.concatenate((svm.magnitude,vm.magnitude)) for vm,svm in zip(vms,spont_vms)]
        else:
           vms1 = vms 
        
        
        xs = []
        ys = []
        colors = []
        for vm in vms1:
            xs.append(time_axis)
            ys.append(numpy.array(vm.tolist()))
            colors.append("#848484")
        
        return [('VmPlot',StandardStyleLinePlot(xs, ys),gs,{
                    "mean" : True,
                    "colors" : colors,
                    "x_lim" : (t_start, t_stop), 
                    "y_lim" : (-80.0, -40.0), 
                    "x_ticks": x_ticks,
                    "x_label": 'time(' + vms[0].t_stop.dimensionality.latex + ')',
                    "y_label": 'Vm(' + vms[0].dimensionality.latex + ')'
               })]



class GSynPlot(Plotting):
    """
    It plots the conductances stored in the recordings.
    It assumes a datastore with a set of recordings. It will plot a line of conductance 
    plots, one per each recording, showing the excitatory and inhibitory conductances corresponding to the given 
    recording.
    
    It defines one plot named: 'ConductancesPlot.Plot0' ... 'ConductancesPlot.PlotN'.
    
    Other parameters
    ----------------
    
    sheet_name : str
               From which layer to plot the conductances.
               
    neuron : int
            Id of the neuron to plot.
    
    spontaneous : bool
                Whether to also show the spontaneous activity the preceded the stimulus.
    """
    
    required_parameters = ParameterSet({
        'neuron': int,  # we can only plot one neuron - which one ?
        'sheet_name': str,
        'spontaneous' : bool, # whether to also show the spontaneous activity the preceded the stimulus
    })

    def subplot(self, subplotspec):
        dsv = queries.param_filter_query(self.datastore,sheet_name=self.parameters.sheet_name)
        return PerStimulusPlot(dsv, function=self._ploter, title_style="Standard"
                                        ).make_line_plot(subplotspec)
    
    @staticmethod
    def concat_asl(asl1,asl2):
        assert asl1.sampling_period == asl2.sampling_period
        assert asl1.units == asl2.units
        
        return NeoAnalogSignal(numpy.concatenate((asl1.magnitude,asl2.magnitude)),t_start=-asl1.t_stop,
                            sampling_period=asl1.sampling_period,
                            units=asl1.units)


    def _ploter(self, dsv,gs):
        segs = sorted(dsv.get_segments(),key = lambda x : MozaikParametrized.idd(x.annotations['stimulus']).trial)
        gsyn_es = [s.get_esyn(self.parameters.neuron) for s in segs]
        gsyn_is = [s.get_isyn(self.parameters.neuron) for s in segs]
        params = {}
        
        if self.parameters.spontaneous:
           segs = sorted(dsv.get_segments(null=True),key = lambda x : MozaikParametrized.idd(x.annotations['stimulus']).trial)
           spont_gsyn_es = [s.get_esyn(self.parameters.neuron) for s in segs] 
           spont_gsyn_is = [s.get_isyn(self.parameters.neuron) for s in segs] 
           
           gsyn_es = [GSynPlot.concat_asl(s,n) for n,s in zip(gsyn_es,spont_gsyn_es)] 
           gsyn_is = [GSynPlot.concat_asl(s,n) for n,s in zip(gsyn_is,spont_gsyn_is)] 
           t_start = - spont_gsyn_es[0].t_stop.magnitude
           t_stop = gsyn_es[0].t_stop.magnitude
           params = {'x_ticks' : [t_start,0, t_stop/2, t_stop]}
        
        return [("ConductancesPlot",ConductancesPlot(gsyn_es, gsyn_is),gs,params)]
    
        
        
        
    


class OverviewPlot(Plotting):
    
    """
    It defines 4 (or 3 depending on the sheet_activity option) plots named: 'Activity_plot', 'Spike_plot', 'Vm_plot', 'Conductance_plot'
    corresponding to the ActivityMovie RasterPlot, VmPlot, GSynPlot plots respectively. And than a line of the with the extensions .Plot1 ... .PlotN 
    
    Other parameters
    ----------------
    
    sheet_name : str
               From which layer to plot.
               
    neuron : int
            Id of the neuron to plot.
            
    sheet_activity: bool
            Whether to also show the sheet activity plot as the first row.
            
    spontaneous : bool
                Whether to also show the spontaneous activity the preceded the stimulus.
    """
    
    required_parameters = ParameterSet({
        'sheet_name': str,  # the name of the sheet for which to plot
        'spontaneous': bool,  # the name of the sheet for which to plot
        'neuron': int,
        'sheet_activity': ParameterSet,  # if not empty the ParameterSet is passed to ActivityMovie which is displayed in to top row, note that the sheet_name will be set by OverviewPlot
    })
    
    def subplot(self, subplotspec):
        dsv = queries.param_filter_query(self.datastore,sheet_name=self.parameters.sheet_name)
        return PerStimulusPlot(dsv, function=self._ploter, title_style="Clever").make_line_plot(subplotspec)

    def _ploter(self, dsv,subplotspec):
        offset = 0
        d = []
        if len(self.parameters.sheet_activity.keys()) != 0:
            gs = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=subplotspec)
            self.parameters.sheet_activity['sheet_name'] = self.parameters.sheet_name
            d.append(("Activity_plot",ActivityMovie(dsv,self.parameters.sheet_activity),gs[0, 0],{}))
            offset = 1
        else:
            gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=subplotspec)

        params = {}
        if offset == 1:
            params['title'] = None
        params['x_label']  = None
        params['x_tick_labels'] = None
        params['x_tick_style'] ='Custom'
        
        d.extend([ ("Spike_plot",RasterPlot(dsv,
                   ParameterSet({'sheet_name': self.parameters.sheet_name,
                                 'trial_averaged_histogram': False,'spontaneous' : self.parameters.spontaneous,
                                 'neurons': [self.parameters.neuron]})
                   ),gs[0 + offset, 0],params),
                
                 ("Conductance_plot",GSynPlot(dsv,
                   ParameterSet({'sheet_name': self.parameters.sheet_name,'spontaneous' : self.parameters.spontaneous,
                               'neuron': self.parameters.neuron})
                 ),gs[1 + offset, 0], {'x_label' : None, 'x_tick_style' : 'Custom' , 'x_tick_labels' : None, 'title' : None}),

                 ("Vm_plot",VmPlot(dsv,
                   ParameterSet({'sheet_name': self.parameters.sheet_name,'spontaneous' : self.parameters.spontaneous,
                             'neuron': self.parameters.neuron})
                 ),gs[2 + offset, 0], {'title' : None})
              ])
        return d



class AnalogSignalListPlot(Plotting):
    """
    This plot shows a line of plots each showing analog signals for different neurons (in the same plot), one plot per each AnalogSignalList instance
    present in the datastore.
    
    It defines line of plots named: 'AnalogSignalPlot.Plot0' ... 'AnalogSignalPlot.PlotN'.
    
    Other parameters
    ----------------
    
    sheet_name : str
               From which layer to plot.
               
    neurons : list
            List of neuron ids for which to plot the analog signals.
            
    """
    
    required_parameters = ParameterSet({
        'sheet_name': str,  # the name of the sheet for which to plot
        'neurons' : list, # list of neuron IDs to show, if empty all neurons are shown
        'mean' : bool, # if true the mean over the neurons is shown
    })

    def subplot(self, subplotspec):
        dsv = queries.param_filter_query(self.datastore,sheet_name=self.parameters.sheet_name,identifier='AnalogSignalList')
        return PerStimulusADSPlot(dsv, function=self._ploter, title_style="Clever").make_line_plot(subplotspec)

    def _ploter(self, dsv,subplotspec):
        self.analog_signal_list = dsv.get_analysis_result()
        assert len(self.analog_signal_list) != 0, "ERROR, empty datastore"
        assert len(self.analog_signal_list) == 1, "Currently only one AnalogSignalList per stimulus can be plotted"
        self.analog_signal_list = self.analog_signal_list[0]
        xs = []
        ys = []
        if self.parameters.neurons == []:
           self.parameters.neurons = self.analog_signal_list.ids
       
        for idd in self.parameters.neurons:
            a = self.analog_signal_list.get_asl_by_id(idd)
            times = numpy.linspace(a.t_start, a.t_stop, len(a))
            xs.append(times)
            ys.append(a)
        
        if self.parameters.mean:
            xs = [xs[0]]
            ys = [numpy.mean(ys,axis=0)]
        
        params = {}
        params["x_lim"] = (a.t_start.magnitude, a.t_stop.magnitude)
        params["x_label"] = self.analog_signal_list.x_axis_name + '(' + a.t_start.dimensionality.latex + ')'
        params["y_label"] = self.analog_signal_list.y_axis_name
        params["x_ticks"] = [a.t_start.magnitude, a.t_stop.magnitude]
        params["mean"] = True
        return [("AnalogSignalPlot" ,StandardStyleLinePlot(xs, ys),subplotspec,params)]



class AnalogSignalPlot(Plotting):
    """
    This plot shows a line of plots each showing the given AnalogSignal, one plot per each AnalogSignal instance present in the datastore.
    
    It defines line of plots named: 'AnalogSignalPlot.Plot0' ... 'AnalogSignalPlot.PlotN'.
    
    Other parameters
    ----------------
    
    sheet_name : str
               From which layer to plot.
    """
    
    required_parameters = ParameterSet({
        'sheet_name': str,  # the name of the sheet for which to plot
    })

    def subplot(self, subplotspec):
        dsv = queries.param_filter_query(self.datastore,sheet_name=self.parameters.sheet_name,identifier='AnalogSignal')
        return PerStimulusADSPlot(dsv, function=self._ploter, title_style="Clever").make_line_plot(subplotspec)

    def _ploter(self, dsv,subplotspec):
        self.analog_signal = dsv.get_analysis_result()
        assert len(self.analog_signal) != 0, "ERROR, empty datastore"
        assert len(self.analog_signal) == 1, "Currently only one AnalogSignal per stimulus can be plotted"
        self.analog_signal = self.analog_signal[0]
        a = self.analog_signal.analog_signal
        times = numpy.linspace(a.t_start, a.t_stop, len(a))
        
        params = {}
        params["x_lim"] = (a.t_start.magnitude, a.t_stop.magnitude)
        params["x_label"] = self.analog_signal.x_axis_name + '(' + a.t_start.dimensionality.latex + ')'
        params["y_label"] = self.analog_signal.y_axis_name
        params["x_ticks"] = [a.t_start.magnitude, a.t_stop.magnitude]
        params["mean"] = True
        return [("AnalogSignalPlot" ,StandardStyleLinePlot([times], [a]),subplotspec,params)]



class ConductanceSignalListPlot(Plotting):
    """
    This plot shows a line of plots each showing excitatory and inhibitory conductances, one plot per each ConductanceSignalList instance 
    present in the datastore.
    
    It defines line of plots named: 'ConductancePlot.Plot0' ... 'ConductancePlot.PlotN'.
    
    Other parameters
    ----------------
               
    normalize_individually : bool
                           Whether to normalize each trace individually by dividing it with its maximum.
                           
    neurons : list 
            
    """
    
    required_parameters = ParameterSet({
        'normalize_individually': bool,  # each trace will be normalized individually by dividing it with its maximum
        'neurons' : list, # list of neuron ids for which to plot the conductances
    })


    def subplot(self, subplotspec):
        dsv = queries.param_filter_query(self.datastore,identifier='ConductanceSignalList')
        return PerStimulusADSPlot(dsv, function=self._ploter, title_style="Clever").make_line_plot(subplotspec)


    def _ploter(self, dsv,subplotspec):
        conductance_signal_list = dsv.get_analysis_result()
        assert len(conductance_signal_list) == 1, "Currently only one ConductanceSignalList per stimulus can be plotted"
        assert len(conductance_signal_list) != 0, "ERROR, empty datastore"
        conductance_signal_list = conductance_signal_list[0]
        e_con = conductance_signal_list.e_con
        i_con = conductance_signal_list.i_con
               
        exc = conductance_signal_list.get_econ_by_id(self.parameters.neurons)
        inh = conductance_signal_list.get_icon_by_id(self.parameters.neurons)
        
        if len(self.parameters.neurons) == 1:
                exc = [exc]
                inh = [inh]
            
        return [("ConductancePlot",ConductancesPlot(exc, inh),subplotspec,{})]



class PerNeuronPairAnalogSignalListPlot(Plotting):
    """
    This plot shows a line of plots each showing analog signals for pairs of neurons (in the same plot), one plot per each AnalogSignalList instance
    present in the datastore.
    
    It defines line of plots named: 'AnalogSignalPlot.Plot0' ... 'AnalogSignalPlot.PlotN'.
    
    Other parameters
    ----------------
    
    sheet_name : str
               From which layer to plot.
            
    """
    required_parameters = ParameterSet({
        'sheet_name': str,  # the name of the sheet for which to plot
    })

    def subplot(self, subplotspec):
        dsv = queries.param_filter_query(self.datastore, sheet_name=self.parameters.sheet_name, identifier='AnalogSignalList')
        return PerStimulusADSPlot(dsv, function=self._ploter, title_style="Clever").make_line_plot( subplotspec )

    def _ploter(self, dsv, subplotspec):
        self.analog_signal_list = dsv.get_analysis_result()
        assert len(self.analog_signal_list) != 0, "ERROR, empty datastore"
        assert len(self.analog_signal_list) == 1, "Currently only one AnalogSignalList per stimulus can be plotted"
        self.analog_signal_list = self.analog_signal_list[0] # take only one, the first
        # get the asl from the first tuple id pair
        asl = self.analog_signal_list.get_asl_by_id_pair( (self.analog_signal_list.ids)[0] )
        times = numpy.linspace(asl.t_start, asl.t_stop, len(asl))
        params = {}
        params["x_lim"] = (asl.t_start.magnitude, asl.t_stop.magnitude)
        params["x_label"] = self.analog_signal_list.x_axis_name + '(' + asl.t_start.dimensionality.latex + ')'
        params["y_label"] = self.analog_signal_list.y_axis_name
        params["x_ticks"] = [asl.t_start.magnitude, asl.t_stop.magnitude]
        params["mean"] = True
        # generate the plot
        return [ ("AnalogSignalPlot", StandardStyleLinePlot([times],[asl]), subplotspec, params) ]



class RetinalInputMovie(Plotting):
    """
    This plots one plot showing the retinal input per each recording in the datastore. 
    
    It defines line of plots named: 'PixelMovie.Plot0' ... 'PixelMovie.PlotN'.
    """
    def __init__(self, datastore, parameters, plot_file_name=None, fig_param=None, frame_duration=0):
        Plotting.__init__(self, datastore, parameters, plot_file_name, fig_param, frame_duration)
        self.length = None
        # currently there is no way to check whether the sensory input is retinal
        self.retinal_input = datastore.get_sensory_stimulus()
        self.st = datastore.sensory_stimulus.keys()
        
        # remove internal stimuli from the list 
        self.retinal_input = [self.retinal_input[i] for i in xrange(0,len(self.st)) if MozaikParametrized.idd(self.st[i]).name != 'InternalStimulus']
        self.st = [self.st[i]  for i in xrange(0,len(self.st)) if MozaikParametrized.idd(self.st[i]).name != 'InternalStimulus']
        
        
        
    def subplot(self, subplotspec):
        return LinePlot(function=self._ploter,
                 length=len(self.retinal_input)
                 ).make_line_plot(subplotspec)

    def _ploter(self, idx, gs):
        stimulus = MozaikParametrized.idd(self.st[idx])
        title = ''
        title = title + stimulus.name + '\n'
        for pn, pv in stimulus.get_param_values():
                title = title + pn + ' : ' + str(pv) + '\n'
        return [('PixelMovie',PixelMovie(self.retinal_input[idx],MozaikParametrized.idd(self.st[idx]).background_luminance),gs,{'x_axis':False, 'y_axis':False, "title" : title})]



class ActivityMovie(Plotting):
    """
    This plots one plot per each recording, each showing the activity during that recording 
    based on the spikes stored in the recording. The activity is showed localized in the sheet cooridantes.
    
    It defines line of plots named: 'ScatterPlot.Plot0' ... 'ScatterPlot.PlotN' or 'PixelMovie.Plot0' ... 'PixelMovie.PlotN'
    depending on the parameter `scatter`.
    
    Other parameters
    ----------------
    
    bin_width : float
              In ms the width of the bins into which to sample spikes.
    
    scatter :  bool   
            Whether to plot neurons activity into a scatter plot (if True) or as an interpolated pixel image.
            
    resolution : int 
               The number of pixels into which the activity will be interpolated in case scatter = False.
               
    sheet_name: str
              The sheet for which to display the actvity movie.
    
    """
    
    required_parameters = ParameterSet({
          'bin_width': float,  # in ms the width of the bins into which to sample spikes
          'scatter':  bool,   # whether to plot neurons activity into a scatter plot (if True) or as an interpolated pixel image
          'resolution': int,  # the number of pixels into which the activity will be interpolated in case scatter = False
          'sheet_name': str,  # the sheet for which to display the actvity movie
    })

    def subplot(self, subplotspec):
        dsv = queries.param_filter_query(self.datastore,sheet_name=self.parameters.sheet_name)
        return PerStimulusPlot(dsv, function=self._ploter, title_style="Clever").make_line_plot(subplotspec)

    def _ploter(self, dsv, gs):
        sp = [s.spiketrains for s in dsv.get_segments()]

        start = sp[0][0].t_start.magnitude
        stop = sp[0][0].t_stop.magnitude
        units = sp[0][0].t_start.units
        bw = self.parameters.bin_width * pq.ms
        bw = bw.rescale(units).magnitude
        bins = numpy.arange(start, stop, bw)
        h = []
        for spike_trains in sp:
            hh = []
            for st in spike_trains:
                hh.append(numpy.histogram(st.magnitude, bins, (start, stop))[0])
                #lets make activity of each neuron relative to it's maximum activity
            h.append(numpy.array(hh))
        
        h = numpy.mean(h, axis=0)
        
        #lets normalize against the maximum response for given neuron
        
        pos = dsv.get_neuron_postions()[self.parameters.sheet_name]

        posx = pos[0,self.datastore.get_sheet_indexes(self.parameters.sheet_name,dsv.get_segments()[0].get_stored_spike_train_ids())]
        posy = pos[1,self.datastore.get_sheet_indexes(self.parameters.sheet_name,dsv.get_segments()[0].get_stored_spike_train_ids())]

                
        if not self.parameters.scatter:
            xi = numpy.linspace(numpy.min(posx)*1.1,
                                numpy.max(posy)*1.1,
                                self.parameters.resolution)
            yi = numpy.linspace(numpy.min(posx)*1.1,
                                numpy.max(posy)*1.1,
                                self.parameters.resolution)

            movie = []
            for i in xrange(0, numpy.shape(h)[1]):
                movie.append(griddata((posx, posy),
                                      h[:, i],
                                      (xi[None, :], yi[:, None]),
                                      method='nearest'))
            w = numpy.isnan(numpy.array(movie))
            numpy.array(movie)[w]=0
            
            return [("PixelMovie",PixelMovie(40000.0*numpy.array(movie)),gs,{'x_axis':False, 'y_axis':False})]
        else:
            return [("ScatterPlot",ScatterPlotMovie(posx, posy, h.T),gs,{'x_axis':False, 'y_axis':False,'dot_size':40})]



class PerNeuronValuePlot(Plotting):
    """
    Plots the values for all PerNeuronValue ADSs in the datastore, one for each sheet.
    
    If the paramter cortical_view is true it will plot the given PerNeuronValue data
    structure values in a scatter plot where the coordinates of the points correspond 
    to coordinates of the corresponding neurons in the cortical space and the colors 
    correspond to the values. 
    
    If the paramter cortical_view is false, it will show the histogram of the values.
    
    It defines line of plots named: 'ScatterPlot.Plot0' ... 'ScatterPlot.PlotN

    Other parameters
    ----------------
    
    cortical_view : bool  
                Whether to show cortical view or histogram (see class description for full detail.)
                
    Notes
    -----
    So far doesn't support the situation where several types of PerNeuronValue analysys data structures are present in the supplied
    datastore.
    """
    
    required_parameters = ParameterSet({
          'cortical_view': bool,  #Whether to show cortical view or histogram (see class description for full detail.)
    })
    
    def __init__(self, datastore, parameters, plot_file_name=None,
                 fig_param=None):
        Plotting.__init__(self, datastore, parameters, plot_file_name, fig_param)
        self.poss = []
        self.pnvs = []
        self.sheets = []
        self.dsv = queries.param_filter_query(self.datastore,identifier='PerNeuronValue')

    def subplot(self, subplotspec):
        return ADSGridPlot(self.dsv,function=self._ploter,x_axis_parameter='value_name',y_axis_parameter='sheet_name').make_grid_plot(subplotspec)

    def _ploter(self, dsv, gs):
         z = dsv.get_analysis_result(identifier='PerNeuronValue')
         if len(z) > 1:
            logger.error('Warning sheet name and value name does\'t seem to uniquely identify and PerNeuronValue ADS in the datastore, we cannot plot more than one!')
        
         pnv = z[0]
         sheet_name = pnv.sheet_name
         pos = self.dsv.get_neuron_postions()[sheet_name]            
        
         if self.parameters.cortical_view:
            posx = pos[0,self.datastore.get_sheet_indexes(sheet_name,pnv.ids)]
            posy = pos[1,self.datastore.get_sheet_indexes(sheet_name,pnv.ids)]
            values = pnv.values
            if pnv.period != None:
                periodic = True
                period = pnv.period
            else:
                periodic = False
                period = None
            params = {}
            params["x_label"] = 'x'
            params["y_label"] = 'y'
            params["colorbar_label"] = pnv.value_units.dimensionality.latex
            params["colorbar"]  = True

            return [("ScatterPlot",ScatterPlot(posx, posy, values, periodic=periodic,period=period),gs,params)]
         else:
            params = {}
            params["y_label"] = '# neurons'
            return [("HistogramPlot",HistogramPlot([pnv.values]),gs,params)]



class PerNeuronValueScatterPlot(Plotting):
    """
    Takes each pair of PerNeuronValue ADSs in the datastore that have the same units and plots a scatter plot of each such pair.
    
    It defines line of plots named: 'ScatterPlot.Plot0' ... 'ScatterPlot.PlotN
    """
    
    required_parameters = ParameterSet({
          'only_matching_units':bool,  # only plot combinations of PNVs that have the same value units.
    })
    
    
    def __init__(self, datastore, parameters, plot_file_name=None,fig_param=None):
        Plotting.__init__(self, datastore, parameters, plot_file_name, fig_param)

        self.pairs = []
        self.sheets = []
        for sheet in datastore.sheets():
            pnvs = datastore.get_analysis_result(identifier='PerNeuronValue',sheet_name=sheet)
            if len(pnvs) < 2:
               raise ValueError('At least 2 DSVs have to be provided') 
            for i in xrange(0,len(pnvs)):
                for j in xrange(i+1,len(pnvs)):
                    if (pnvs[i].value_units == pnvs[j].value_units) or not self.parameters.only_matching_units:
                       self.pairs.append((pnvs[i],pnvs[j]))
                       self.sheets.append(sheet) 
                       
        assert len(self.pairs) > 0, "Error, not pairs of PerNeuronValue ADS in datastore seem to have the same value_units"
        self.length=len(self.pairs)
        
    def subplot(self, subplotspec):
        return LinePlot(function=self._ploter,length=self.length,shared_axis=False).make_line_plot(subplotspec)

    def _ploter(self, idx, gs):
        pair = self.pairs[idx]
        # Let's figure out the varying parameters
        p1 = varying_parameters(pair)
        if pair[0].stimulus_id == None or pair[1].stimulus_id == None:
            p2 = []
        elif MozaikParametrized.idd(pair[0].stimulus_id).name != MozaikParametrized.idd(pair[1].stimulus_id).name:
            p2 = ['name']
        else:
            p2 = varying_parameters([MozaikParametrized.idd(p.stimulus_id) for p in pair])
        p1 = [x for x in p1 if ((x != 'value_name') and (x != 'stimulus_id'))]

        x_label = pair[0].value_name + '(' + pair[0].value_units.dimensionality.latex + ')'
        y_label = pair[1].value_name + '(' + pair[1].value_units.dimensionality.latex + ')'

        for p in p1:
            x_label += '\n' + str(p) + " = " + str(getattr(pair[0],p))
            y_label += '\n' + str(p) + " = " + str(getattr(pair[1],p))
        
        for p in p2:
            x_label += '\n' + str(p) + " = " + str(getattr(MozaikParametrized.idd(pair[0].stimulus_id),p))
            y_label += '\n' + str(p) + " = " + str(getattr(MozaikParametrized.idd(pair[1].stimulus_id),p))
        
        params = {}
        params["x_label"] = x_label
        params["y_label"] = y_label
        params["title"] = self.sheets[idx]
        if pair[0].value_units != pair[1].value_units or pair[1].value_units == pq.dimensionless:
           params["equal_aspect_ratio"] = False
        
        ids = list(set(pair[0].ids) & set(pair[1].ids))
        return [("ScatterPlot",ScatterPlot(pair[0].get_value_by_id(ids), pair[1].get_value_by_id(ids)),gs,params)]
        


class ConnectivityPlot(Plotting):
    """
    Plots Connectivity, one for each projection originating or targeting
    (depending on parameter reversed) sheet `sheet_name` for a single neuron in the
    sheet.

    This plot can accept second DSV that contains the PerNeuronValues
    corresponding to the target sheets (or source sheet if reversed is True) to be displayed that will be plotted as
    well.
    
    It defines line of plots named: 'ConnectionsPlot.Plot0' ... 'DelaysPlot.PlotN` and 'DelaysPlot.Plot0' ... 'DelaysPlot.PlotN
    where N is number of projections.
    
    Parameters
    ----------
    
    pnv_dsv : Datastore
            The datastore holding PerNeuronValues - one per each sheet that will be displayed as colors for the connections.
            
    Other parameters
    ----------------
    neuron : int  
           The target neuron whose connections are to be displayed.
    
    reversed : bool
             If false the outgoing connections from the given neuron are shown. if true the incomming connections are shown.
    
    sheet_name : str
               For neuron in which sheet to display connectivity.
        
    Notes
    -----
    
    One PerNeuronValue can be present per target sheet!
    """

    required_parameters = ParameterSet({
        'neuron': int,  # the target neuron whose connections are to be displayed
        'reversed': bool,  # if false the outgoing connections from the given neuron are shown. if true the incomming connections are shown
        'sheet_name': str,  # for neuron in which sheet to display connectivity
    })

    def __init__(self, datastore, parameters, pnv_dsv=None,
                 plot_file_name=None, fig_param=None):
        Plotting.__init__(self, datastore, parameters, plot_file_name, fig_param)
        self.connecting_neurons_positions = []
        self.connected_neuron_position = []
        self.connections = []

        _connections = datastore.get_analysis_result(identifier='Connections')
        self.pnvs = None
        if pnv_dsv != None:
            self.pnvs = []
            z = queries.partition_analysis_results_by_parameters_query(
                                                pnv_dsv,
                                                parameter_list=['sheet_name'],excpt=True)
            for dsv in z:
                a = dsv.get_analysis_result(identifier='PerNeuronValue')
                self.pnvs.append(a[0])
        print len(self.pnvs)
        for conn in _connections:
            print conn
            if not self.parameters.reversed and conn.source_name == self.parameters.sheet_name:
                # add outgoing projections from sheet_name
                self.connecting_neurons_positions.append(
                            datastore.get_neuron_postions()[conn.target_name])
                z = datastore.get_neuron_postions()[conn.source_name]
                idx = self.datastore.get_sheet_indexes(conn.source_name,self.parameters.neuron)
                self.connected_neuron_position.append(
                            (z[0][idx],
                             z[1][idx]))
                self.connections.append(conn)
            elif (self.parameters.reversed and conn.target_name == self.parameters.sheet_name):
                # add incomming projections from sheet_name
                self.connecting_neurons_positions.append(
                            datastore.get_neuron_postions()[conn.source_name])
                z = datastore.get_neuron_postions()[conn.target_name]
                idx = self.datastore.get_sheet_indexes(conn.target_name,self.parameters.neuron)
                self.connected_neuron_position.append(
                            (z[0][idx],
                             z[1][idx]))
                self.connections.append(conn)

        self.length=len(self.connections)

    def subplot(self, subplotspec):
        return LinePlot(function=self._ploter, length=self.length, shared_axis=True
                 ).make_line_plot(subplotspec)

    def _ploter(self, idx, gs):
        #ix = numpy.flatnonzero(self.connections[idx].weights[:,1]==index)
        #x = self.proj.pre.positions[0][self.connections[idx].weights[ix,0]]
        #y = self.proj.pre.positions[1][self.connections[idx].weights[ix,0]]
        #w = weights[idx,2]
        tx = self.connected_neuron_position[idx][0]
        ty = self.connected_neuron_position[idx][1]
        if not self.parameters.reversed:
            index = self.datastore.get_sheet_indexes(self.connections[idx].source_name,self.parameters.neuron)
            ix = numpy.flatnonzero(numpy.array(self.connections[idx].weights)[:,0]==index)
            ix = numpy.array(self.connections[idx].weights)[:,1][ix].astype(int)
        else:
            index = self.datastore.get_sheet_indexes(self.connections[idx].target_name,self.parameters.neuron)
            ix = numpy.flatnonzero(numpy.array(self.connections[idx].weights)[:,1]==index)
            ix = numpy.array(self.connections[idx].weights)[:,0][ix].astype(int)
            
        sx = self.connecting_neurons_positions[idx][0][ix]
        sy = self.connecting_neurons_positions[idx][1][ix]
        w = numpy.array(self.connections[idx].weights)[ix,2]
        d = numpy.array(self.connections[idx].delays)[ix,2]

        assert numpy.shape(w) == numpy.shape(d)
        # pick the right PerNeuronValue to show
        pnv = []
        if self.pnvs != None:
            for p in self.pnvs:
                if not self.parameters.reversed and p.sheet_name == self.connections[idx].target_name:
                    pnv.append(p)
                if self.parameters.reversed and p.sheet_name == self.connections[idx].source_name:
                    pnv.append(p)
            
            if len(pnv) > 1:
                raise ValueError('ERROR: too many matching PerNeuronValue ADSs')
            elif len(pnv) != 0:
                pnv = pnv[0]

        gss = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs)
        
        # Plot the weights
        gs = gss[0,0]
        params = {}
        
        if self.parameters.reversed:
            xs = self.connections[idx].source_size[0]
            ys = self.connections[idx].source_size[1]
        else:
            xs = self.connections[idx].target_size[0]
            ys = self.connections[idx].target_size[1]
        
        params["x_lim"] = (-xs/2.0,xs/2.0)
        params["y_lim"] = (-ys/2.0,ys/2.0)
        
        if pnv != []:
            from mozaik.tools.circ_stat import circ_mean
            (angle, mag) = circ_mean(numpy.array(pnv.get_value_by_id(self.datastore.get_sheet_ids(pnv.sheet_name,ix))),
                                     weights=w,
                                     high=pnv.period)
            params["title"] = str(self.connections[idx].proj_name) + "\n Mean: " + str(angle)
            params["colorbar_label"] =  pnv.value_name
            params["colorbar"] = True

            if self.connections[idx].source_name == self.connections[idx].target_name:
                params["line"] = False
                plots = [("ConnectionsPlot",ConnectionPlot(sx, sy, tx, ty, w,colors=pnv.get_value_by_id(self.datastore.get_sheet_ids(pnv.sheet_name,ix)),period=pnv.period),gs,params)]
            else:
                params["line"] = True
                plots = [("ConnectionsPlot",ConnectionPlot(sx, sy, numpy.min(sx)*1.2, numpy.min(sy)*1.2, w,colors=pnv.get_value_by_id(self.datastore.get_sheet_ids(pnv.sheet_name,ix)),period=pnv.period),gs,params)]
        else:
            params["title"] = 'Weights: '+ self.connections[idx].proj_name
            
            if self.connections[idx].source_name == self.connections[idx].target_name:
                params["line"] = False
                plots = [("ConnectionsPlot",ConnectionPlot(sx, sy, tx, ty, w),gs,params)]
            else:
                params["line"] = True
                plots = [("ConnectionsPlot",ConnectionPlot(sx, sy, numpy.min(sx)*1.2, numpy.min(sy)*1.2,w),gs,params)]


        # Plot the delays
        gs = gss[1,0]
        params = {}
        params["x_lim"] = (-xs/2.0,xs/2.0)
        params["y_lim"] = (-ys/2.0,ys/2.0)
        if idx == self.length-1:
           params["colorbar"] = True
        
        params["title"]  = 'Delays: '+ self.connections[idx].proj_name
        if self.connections[idx].source_name == self.connections[idx].target_name:
            params["line"] = False
            plots.append(("DelaysPlot",ConnectionPlot(sx, sy, tx, ty, (numpy.zeros(w.shape)+0.3)*(w!=0),colors=d),gs,params))
        else:
            params["line"] = True
            plots.append(("DelaysPlot",ConnectionPlot(sx, sy, numpy.min(sx)*1.2, numpy.min(sy)*1.2,(numpy.zeros(w.shape)+0.3)*(w!=0),colors=d),gs,params))
        
        return plots
        


class PerNeuronAnalogSignalScatterPlot(Plotting):
    """
    This plot expects exactly two AnalogSignalList ADS in the datastore. It then for each neuron
    specified in the parameters creates a scatter plot of the values occuring at the same time in the two
    AnalogSignalList ADSs.
    
    It defines line of plots named: 'AnalogSignalScatterPlot.Plot0' ... 'AnalogSignalScatterPlot.PlotN'.
    
    Other parameters
    ----------------
    
    neurons : list
            List of neuron ids for which to plot the tuning curves.
    """
    
    required_parameters = ParameterSet({
        'neurons': list,
    })
    
    def __init__(self, datastore, parameters, plot_file_name=None, fig_param=None, frame_duration=0):
        Plotting.__init__(self, datastore, parameters, plot_file_name, fig_param, frame_duration)
        self.asls = queries.param_filter_query(datastore, name='AnalogSignalList').get_analysis_result()
        assert len(self.asls)==2 , "PerNeuronAnalogSignalScatterPlot expects exactly two AnalogSignalList ADS in the datastore, found %d" % len(self.asls)
            
    def subplot(self, subplotspec):
        return LinePlot(function=self._ploter,
                 length=len(self.parameters.neurons)
                 ).make_line_plot(subplotspec)

    def _ploter(self, idx, gs):
        a = self.asls[0].get_asl_by_id(self.parameters.neurons[idx])
        b = self.asls[1].get_asl_by_id(self.parameters.neurons[idx])
        
        assert a.t_start == b.t_start
        assert a.t_stop == b.t_stop
        assert a.sampling_rate == b.sampling_rate
        
        return [('AnalogSignalScatterPlot',ScatterPlot(a.magnitude,b.magnitude),gs,{'x_label': self.asls[0].y_axis_name, 'y_label': self.asls[1].y_axis_name, "title" : "Neuron id: %d" % self.parameters.neurons[idx]})]

