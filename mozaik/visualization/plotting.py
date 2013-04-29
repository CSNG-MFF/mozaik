"""
The visualization for Mozaik objects is based on matplotlib and the GridSpec
object.

The plotting framwork is divided into two main concepts, represented by the two
high-level classes Plotting (this file) and SimplePlot (see simple_plot.py).

The SimplePlot represent low-level plotting. It is assumed that this plot has
only a single axis that is drawn into the region defined by the GridSpec
instance that is passed into it. The role of the set of classes derived from
SimplePlot is to standardize the low level looks of all figures (mainly related
to axis, lables, titles etc.), and should assume data given to them in a format
that is easy to use by the given plot. In order to unify the look of figures it
defines four functions pre_axis_plot, pre_plot, plot, and post_plot. The actual
plotting that user defines is typically defined in the plot function while the
pre_axis_plot, pre_plot and post_plot functions handle the pre and post
plotting adjustments to the plot (i.e. the typical post_plot function for
example adjusts the ticks of the axis to a common format other such axis
related properties). When defining a new SimplePlot function user is encoureged
to push as much of it's 'decorating' funcitonality into the post and pre plot
function and define only the absolute minimum in the plot function. At the same
time, there is already a set of classes implementing a general common look
provided, and so users are encouraged to use these as much as possible. If
their formatting features are not sufficient or incompatible with a given plot,
users are encoureged to define new virtual class that defines the formatting in
the pre and post plot functions (and thus sepparating it from the plot itself),
and incorporating these as low as possible within the hierarchy of the
SimplePlot classes to re-use as much of the previous work as possible.

NOTE SimplePlot now resides in a separate module visualization.simple_plot but
its description stays as it is integral to how Plotting class works.

The Plotting class (and its children) define the high level plotting
mechanisms. They are mainly responsible for hierarchical organization of
figures with multiple plots, any mechanisms that require consideration of
several plots at the same time, and the translation of the data form the general
format provided by Datastore, to specific format that the SimplePlot plots
require. In general the Plotting instances should not do any plotting of axes
them selves (but instead calling the SimplePlot instances to do the actual
plotting), with the exception of multi-axis figures whith complicated inter-axis
dependencies, for which it is not practical to break them down into single
SimplePlot instances.

Each Plotting class should implement two plotting functions:  plot and subplot.
The subplot function accepts a SubplotSpec object (see matplotlib doc) as input which 
will tell it where to plot. The subplot function returns a dictionary, where each key
is a name of a subplot it generates (and will correspond to a creation of another Plotting or 
SimplePlot plot), and the associated value is a tuple that contains the 
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

from NeuroTools.parameters import ParameterSet

from mozaik.framework.interfaces import MozaikParametrizeObject
from mozaik.storage import queries
from mozaik.tools.mozaik_parametrized import colapse_to_dictionary, MozaikParametrized, varying_parameters, matching_parametrized_object_params
from numpy import pi

from simple_plot import StandardStyleLinePlot, SpikeRasterPlot, \
                        SpikeHistogramPlot, ConductancesPlot, PixelMovie, \
                        ScatterPlotMovie, ScatterPlot, ConnectionPlot, SimplePlot
from plot_constructors import LinePlot, PerStimulusPlot, PerStimulusADSPlot

import mozaik
logger = mozaik.getMozaikLogger("Mozaik")


class Plotting(MozaikParametrizeObject):
    """
    The high level Plotting API.

    datastore - the DataStore from which to plot the data
    parameters - no comment
    plot_file_name - either None, in which case the plot is not saved onto
                     HD, or path to a file into which to save the file (formats
                     accepted by matplotlib)
    fig_params - the parameters that are passed to the matplotlib figure
                 command. (but note facecolor='w' is already supplied)
    """

    def  __init__(self, datastore, parameters, plot_file_name=None,
                  fig_param=None):
        MozaikParametrizeObject.__init__(self, parameters)
        self.datastore = datastore
        self.plot_file_name = plot_file_name
        self.fig_param = fig_param if fig_param != None else {}

    def subplot(self, subplotspec):
        raise NotImplementedError
    
    def nip_parameters(self,n,p):
        d = {}
        fd = {}
        for (k,v) in p.iteritems():
            l = k.split('.')
            assert len(l) > 1, "Parameter %s not matching the simple plot" % (k)
            if l[0] == n or l[0] == '*':
                if len(l[1:]) >1 : 
                   d[l[1:].join('.')] = v
                else:
                   fd[l[1]] = v
        return d,fd
    
    def handle_parameters_and_execute_plots(self,parameters,user_parameters,gs):
        d = self.subplot(gs)
        for (k,(pl,gs,p)) in d.iteritems():
            p.update(parameters)
            ### THIS IS WRONG 'UP' DO NOT WORK        
            up = user_parameters
            for z in k.split('.'):    
                up,fp = self.nip_parameters(z,up)
                p.update(fp)

            param = p
            
            if isinstance(pl,SimplePlot):
               # print check whether all user_parameters have been nipped to minimum 
               pl(gs,param) 
            elif isinstance(pl,Plotting):
               pl.handle_parameters_and_execute_plots(param,up,gs)     
            else:
               raise TypeError("The subplot object is not of type Plotting or SimplePlot") 
    
    def plot(self, params=None):
        t1 = time.time()
        if params == None:
            params = {}
        self.fig = pylab.figure(facecolor='w', **self.fig_param)
        gs = gridspec.GridSpec(1, 1)
        gs.update(left=0.07, right=0.97, top=0.95, bottom=0.05)
        self.handle_parameters_and_execute_plots({}, params,gs[0, 0])
        if self.plot_file_name:
            pylab.savefig(self.plot_file_name)
        t2 = time.time()
        logger.warning(self.__class__.__name__ + ' plotting took: ' + str(t2 - t1) + 'seconds')


class PlotTuningCurve(Plotting):
    """
    Defines 'TuningCurve_' + value_name +  '.Plot0' ... 'TuningCurve_' + value_name +  '.Plotn'
    
    where n goes through number of neurons, and value_name creates one row for each value_name found in the different PerNeuron found
    """

    required_parameters = ParameterSet({
      'neurons':  list,  # which neurons to plot
      'sheet_name': str,  # from which layer to plot the tuning curves
      'parameter_name': str  # the parameter_name through which to plot the tuning curve
    })

    def __init__(self, datastore, parameters, plot_file_name=None,
                 fig_param=None):
        Plotting.__init__(self, datastore, parameters, plot_file_name, fig_param)
        
        self.st = []
        self.tc_dict = []
        self.pnvs = []
        assert queries.ads_with_equal_stimulus_type(self.datastore)
        assert len(self.parameters.neurons) > 0 , "ERROR, empty list of neurons specified"
        dsvs = queries.partition_analysis_results_by_parameters_query(self.datastore,parameter_list=['value_name'],excpt=True)
        for dsv in dsvs:
            dsv = queries.param_filter_query(dsv,identifier='PerNeuronValue',sheet_name=self.parameters.sheet_name)
            assert matching_parametrized_object_params(dsv.get_analysis_result(), params=['value_name'])
            self.pnvs.append(dsv.get_analysis_result())
            # get stimuli
            st = [MozaikParametrized.idd(s.stimulus_id) for s in self.pnvs[-1]]
            self.st.append(st)
            # transform the pnvs into a dictionary of tuning curves along the parameter_name
            # also make sure the values are ordered according to ids in the first pnv
            self.tc_dict.append(colapse_to_dictionary([z.get_value_by_id(self.parameters.neurons) for z in self.pnvs[-1]],st,self.parameters.parameter_name))
            

    def subplot(self, subplotspec):
        return LinePlot(function=self.ploter, length=len(self.parameters.neurons)).make_line_plot(subplotspec)

    def ploter(self, idx, gs):
        plots  = []
        gs = gridspec.GridSpecFromSubplotSpec(len(self.st), 1, subplot_spec=gs)
        for i,(dic, st, pnvs) in enumerate(zip(self.tc_dict,self.st,self.pnvs)):
            period = st[0].params()[self.parameters.parameter_name].period
            xs = []
            ys = []
            labels = []
            for k in  dic:
                (b, a) = dic[k]
                par, val = zip(
                             *sorted(
                                zip(b,
                                    numpy.array(a)[:, idx])))
                if period != None:
                    par = list(par)
                    val = list(val)
                    par.append(par[0] + period)
                    val.append(val[0])

                xs.append(numpy.array(par))
                ys.append(numpy.array(val))
                labels.append(str(k))
            
            params={}
            params["y_label"] = pnvs[0].value_name
            params['labels']=None
            params['linewidth'] = 2
            params['colors'] = [cm.jet(j/50.,1) for j  in xrange(0,len(xs))] 

            if pnvs == self.pnvs[0]:
                params["title"] =  'Neuron ID: %d' % self.parameters.neurons[idx]
            
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

            if pnvs != self.pnvs[-1]:
                params["x_axis"] = None
            plots.append(("TuningCurve_" + pnvs[0].value_name,StandardStyleLinePlot(xs, ys),gs[i],params))
        return plots

class RasterPlot(Plotting):
    """ 
    Defines 'RasterPlot.Plot0' ... 'RasterPlot.PlotN'
    """
    required_parameters = ParameterSet({
        'trial_averaged_histogram': bool,  # should the plot show also the trial averaged histogram
        'neurons': list,
        'sheet_name': str,
    })

    def __init__(self, datastore, parameters, plot_file_name=None, fig_param=None):
        Plotting.__init__(self, datastore, parameters, plot_file_name, fig_param)

    def subplot(self, subplotspec):
        dsv = queries.param_filter_query(self.datastore,sheet_name=self.parameters.sheet_name)
        return PerStimulusPlot(dsv, function=self.ploter).make_line_plot(subplotspec)

    def ploter(self, dsv,gs):
        sp = [[s.get_spiketrain(self.parameters.neurons) for s in dsv.get_segments()]]
        d = {} 
        if self.parameters.trial_averaged_histogram:
            gs = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs)
            # first the raster
            return [ ('SpikeRasterPlot',SpikeRasterPlot(sp),gs[:3, 0],{'x_axis': False , 'x_label' :  None}),
                     ('SpikeHistogramPlot',SpikeHistogramPlot(sp),gs[3, 0],{})]
        else:
            return [('SpikeRasterPlot',SpikeRasterPlot(sp),gs,{})]

        
class VmPlot(Plotting):
    """
    It defines one plot named: 'VmPlot.Plot0' ... 'VmPlot.PlotN'.
    """

    required_parameters = ParameterSet({
      'neuron': int,  # we can only plot one neuron - which one ?
      'sheet_name': str,
    })

    def subplot(self, subplotspec):
        dsv = queries.param_filter_query(self.datastore,sheet_name=self.parameters.sheet_name)
        return PerStimulusPlot(dsv, function=self.ploter, title_style="Clever"
                                         ).make_line_plot(subplotspec)

    def ploter(self, dsv,gs):
        vms = [s.get_vm(self.parameters.neuron) for s in dsv.get_segments()]
        sampling_period = vms[0].sampling_period
        time_axis = numpy.arange(0, len(vms[0]), 1) / float(len(vms[0])) * float(vms[0].t_stop) + float(vms[0].t_start)
        t_stop = float(vms[0].t_stop - sampling_period)

        xs = []
        ys = []
        colors = []
        for vm in vms:
            xs.append(time_axis)
            ys.append(numpy.array(vm.tolist()))
            colors.append("#848484")

        return [('VmPlot',StandardStyleLinePlot(xs, ys),gs,{
                    "mean" : True,
                    "colors" : colors,
                    "x_lim" : (0, t_stop), 
                    "x_ticks": [0, t_stop/2, t_stop],
                    "x_label": 'time(' + vms[0].t_stop.dimensionality.latex + ')',
                    "y_label": 'Vm(' + vms[0].dimensionality.latex + ')'
               })]


class GSynPlot(Plotting):
    """
    It defines one plot named: 'ConductancesPlot.Plot0' ... 'ConductancesPlot.PlotN'.
    """
    
    required_parameters = ParameterSet({
        'neuron': int,  # we can only plot one neuron - which one ?
        'sheet_name': str,
    })

    def subplot(self, subplotspec):
        dsv = queries.param_filter_query(self.datastore,sheet_name=self.parameters.sheet_name)
        return PerStimulusPlot(dsv, function=self.ploter, title_style="Standard"
                                        ).make_line_plot(subplotspec)

    def ploter(self, dsv,gs):
        gsyn_es = [s.get_esyn(self.parameters.neuron) for s in dsv.get_segments()]
        gsyn_is = [s.get_isyn(self.parameters.neuron) for s in dsv.get_segments()]
        return [("ConductancesPlot",ConductancesPlot(gsyn_es, gsyn_is),gs,{})]


class OverviewPlot(Plotting):
    
    """
    It defines 4 (or 3 depending on the sheet_activity option) plots named: 'Activity_plot', 'Spike_plot', 'Vm_plot', 'Conductance_plot'
    corresponding to the ActivityMovie RasterPlot, VmPlot, GSynPlot plots respectively. And than a line of the with the extensions .Plot1 ... .PlotN 
    """
    
    required_parameters = ParameterSet({
        'sheet_name': str,  # the name of the sheet for which to plot
        'neuron': int,
        'sheet_activity': ParameterSet,  # if not empty the ParameterSet is passed to ActivityMovie which is displayed in to top row, note that the sheet_name will be set by OverviewPlot
    })
    
    def subplot(self, subplotspec):
        dsv = queries.param_filter_query(self.datastore,sheet_name=self.parameters.sheet_name)
        return PerStimulusPlot(dsv, function=self.ploter, title_style="Clever").make_line_plot(subplotspec)

    def ploter(self, dsv,subplotspec):
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
        params['x_label']  = False
        d.extend([ ("Spike_plot",RasterPlot(dsv,
                   ParameterSet({'sheet_name': self.parameters.sheet_name,
                                 'trial_averaged_histogram': False,
                                 'neurons': [self.parameters.neuron]})
                   ),gs[0 + offset, 0],params),
                
                 ("Conductance_plot",GSynPlot(dsv,
                   ParameterSet({'sheet_name': self.parameters.sheet_name,
                               'neuron': self.parameters.neuron})
                 ),gs[1 + offset, 0], {'x_label' : False, 'title' : None}),

                 ("Vm_plot",VmPlot(dsv,
                   ParameterSet({'sheet_name': self.parameters.sheet_name,
                             'neuron': self.parameters.neuron})
                 ),gs[2 + offset, 0], {'title' : None})
              ])
        return d

class AnalogSignalListPlot(Plotting):
    required_parameters = ParameterSet({
        'sheet_name': str,  # the name of the sheet for which to plot
        'neurons' : list, # list of neuron IDs to show, if empty all neurons are shown
    })

    def __init__(self, datastore, parameters, plot_file_name=None,
                 fig_param=None):
        Plotting.__init__(self, datastore, parameters, plot_file_name, fig_param)
        

    def subplot(self, subplotspec):
        dsv = queries.param_filter_query(self.datastore,sheet_name=self.parameters.sheet_name,identifier='AnalogSignalList')
        return PerStimulusADSPlot(dsv, function=self.ploter, title_style="Clever").make_line_plot(subplotspec)

    def ploter(self, dsv,subplotspec):
        self.analog_signal_list = dsv.get_analysis_result()
        assert len(self.analog_signal_list) == 1, "Currently only one AnalogSignalList per stimulus can be plotted"
        assert len(self.analog_signal_list) != 0, "ERROR, empty datastore"
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
        
        params = {}
        params["x_lim"] = (a.t_start.magnitude, a.t_stop.magnitude)
        params["x_label"] = self.analog_signal_list.x_axis_name + '(' + a.t_start.dimensionality.latex + ')'
        params["y_label"] = self.analog_signal_list.y_axis_name
        params["x_ticks"] = [a.t_start.magnitude, a.t_stop.magnitude]
        params["mean"] = True
        return [("AnalogSignalPlot" ,StandardStyleLinePlot(xs, ys),subplotspec,params)]


class ConductanceSignalListPlot(Plotting):
    required_parameters = ParameterSet({
        'sheet_name': str,  # the name of the sheet for which to plot
        'normalize_individually': bool  # each trace will be normalized individually by dividing it with its maximum
    })

    def __init__(self, datastore, parameters, plot_file_name=None,
                 fig_param=None):
        Plotting.__init__(self, datastore, parameters, plot_file_name, fig_param)
        self.conductance_signal_list = self.datastore.get_analysis_result(
                                            identifier='ConductanceSignalList',
                                            sheet_name=parameters.sheet_name)
        if len(self.conductance_signal_list) > 1:
            logger.error('Warning currently only the first ConductanceSignalList will be plotted')
        self.conductance_signal_list = self.conductance_signal_list[0]
        self.e_con = self.conductance_signal_list.e_con
        self.i_con = self.conductance_signal_list.i_con

    def subplot(self, subplotspec):
        exc =[]
        inh =[]
        for e, i in zip(self.e_con, self.i_con):
            exc.append(e)
            inh.append(i)
        return {"ConductancePlot" : (ConductancesPlot(exc, inh),subplotspec,{})}


class RetinalInputMovie(Plotting):
    required_parameters = ParameterSet({
        'frame_rate': int,  # the desired frame rate (per sec), it might be less if the computer is too slow
    })

    def __init__(self, datastore, parameters, plot_file_name=None,
                 fig_param=None):
        Plotting.__init__(self, datastore, parameters, plot_file_name, fig_param)
        self.length = None
        self.retinal_input = datastore.get_retinal_stimulus()
        print len(self.retinal_input)
        self.st = datastore.retinal_stimulus.keys()
        
    def subplot(self, subplotspec):
        return LinePlot(function=self.ploter,
                 length=len(self.retinal_input)
                 ).make_line_plot(subplotspec)

    def ploter(self, idx, gs):
        return [('PixelMovie',PixelMovie(self.retinal_input[idx],1.0/self.parameters.frame_rate*1000),gs,{'x_axis':False, 'y_axis':False, "title" : str(self.st[idx])})]


class ActivityMovie(Plotting):
    required_parameters = ParameterSet({
          'frame_rate': int,  # the desired frame rate (per sec), it might be less if the computer is too slow
          'bin_width': float,  # in ms the width of the bins into which to sample spikes
          'scatter':  bool,   # whether to plot neurons activity into a scatter plot (if True) or as an interpolated pixel image
          'resolution': int,  # the number of pixels into which the activity will be interpolated in case scatter = False
          'sheet_name': str,  # the sheet for which to display the actvity movie
    })

    def subplot(self, subplotspec):
        dsv = queries.param_filter_query(self.datastore,sheet_name=self.parameters.sheet_name)
        return PerStimulusPlot(dsv, function=self.ploter, title_style="Standard"
                        ).make_line_plot(subplotspec)

    def ploter(self, dsv, gs):
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
            h.append(numpy.array(hh))
        h = numpy.sum(h, axis=0)

        pos = dsv.get_neuron_postions()[self.parameters.sheet_name]

        if not self.parameters.scatter:
            xi = numpy.linspace(numpy.min(pos[0])*1.1,
                                numpy.max(pos[0])*1.1,
                                self.parameters.resolution)
            yi = numpy.linspace(numpy.min(pos[1])*1.1,
                                numpy.max(pos[1])*1.1,
                                self.parameters.resolution)

            movie = []
            for i in xrange(0, numpy.shape(h)[1]):
                movie.append(griddata((pos[0], pos[1]),
                                      h[:, i],
                                      (xi[None, :], yi[:, None]),
                                      method='cubic'))

            return [("PixelMovie",PixelMovie(movie, 1.0/self.parameters.frame_rate*1000),gs,{'x_axis':False, 'y_axis':False})]
        else:
            return [("ScatterPlot",ScatterPlotMovie(pos[0], pos[1], h.T,1.0/self.parameters.frame_rate*1000),gs,{'x_axis':False, 'y_axis':False,'dot_size':40})]


class PerNeuronValuePlot(Plotting):
    """
    Plots PerNeuronValuePlots, one for each sheet.

    #JAHACK, so far doesn't support the situation where several types of
    PerNeuronValue analysys data structures are present in the supplied
    datastore.
    """

    def __init__(self, datastore, parameters, plot_file_name=None,
                 fig_param=None):
        Plotting.__init__(self, datastore, parameters, plot_file_name, fig_param)
        self.poss = []
        self.pnvs = []
        self.sheets = []
        for sheet in datastore.sheets():
            dsv = queries.param_filter_query(self.datastore,sheet_name=sheet)
            z = dsv.get_analysis_result(identifier='PerNeuronValue')
            if len(z) != 0:
                if len(z) > 1:
                    logger.error('Warning currently only one PerNeuronValue per sheet will be plotted!!!')
                self.poss.append(datastore.get_neuron_postions()[sheet])
                self.pnvs.append(z)
                self.sheets.append(sheet)

        self.length=len(self.poss)

    def subplot(self, subplotspec):
        return LinePlot(function=self.ploter,length=self.length).make_line_plot(subplotspec)

    def ploter(self, idx, gs):
        
        posx = self.poss[idx][0,self.datastore.get_sheet_indexes(self.sheets[idx],self.pnvs[idx][0].ids)]
        posy = self.poss[idx][1,self.datastore.get_sheet_indexes(self.sheets[idx],self.pnvs[idx][0].ids)]
        values = self.pnvs[idx][0].values
        if self.pnvs[idx][0].period != None:
            periodic = True
            period = self.pnvs[idx][0].period
        else:
            periodic = False
            period = None
        params = {}
        params["x_label"] = 'x'
        params["y_label"] = 'y'
        params["title"] = self.sheets[idx] + '\n' + self.pnvs[idx][0].value_name
        params["colorbar_label"] = self.pnvs[idx][0].value_units.dimensionality.latex

        if periodic:
            if idx == self.length - 1:
                params["colorbar"]  = True
        else:
            params["colorbar"]  = True
        return [("ScatterPlot",ScatterPlot(posx, posy, values, periodic=periodic,period=period),gs,params)]


class PerNeuronValueScatterPlot(Plotting):
    """
    Takes each pair of PerNeuronValue analysis datastructures in the datastore that have the same units 
    and plots a scatter plot of each such pair.
    """
    
    def __init__(self, datastore, parameters,**kwargs):
        Plotting.__init__(self, datastore, parameters,**kwargs)
        self.pairs = []
        self.sheets = []
        for sheet in datastore.sheets():
            dsvs = datastore.get_analysis_result(identifier='PerNeuronValue',sheet_name=sheet)
            if len(dsvs) < 2:
               raise ValueError('At least 2 DSVs have to be provided') 
            for i in xrange(0,len(dsvs)):
                for j in xrange(i+1,len(dsvs)):
                    if dsvs[i].value_units == dsvs[j].value_units:
                       self.pairs.append((dsvs[i],dsvs[j]))
                       self.sheets.append(sheet) 
                       
        assert len(self.pairs) > 0, "Error, not pairs of PerNeuronValue ADS in datastore seem to have the same value_units"
        self.length=len(self.pairs)
        
    def subplot(self, subplotspec):
        return LinePlot(function=self.ploter,length=self.length,shared_axis=False).make_line_plot(subplotspec)

    def ploter(self, idx, gs):
        pair = self.pairs[idx]
        # Let's figure out the varying parameters
        p1 = varying_parameters(pair)
        if MozaikParametrized.idd(pair[0].stimulus_id).name != MozaikParametrized.idd(pair[1].stimulus_id).name:
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
        
        return [("ScatterPlot",ScatterPlot(pair[0].values, pair[1].get_value_by_id(pair[0].ids)),gs,params)]
        

class ConnectivityPlot(Plotting):
    """
    Plots Connectivity, one for each projection originating or targeting
    (depending on parameter reversed) sheet_name for a single neuron in the
    sheet sheet_name.

    This plot can accept second DSV that contains the PerNeuronValues
    corresponding to the target sheets (or source sheet if reversed is True) to be displayed that will be plotted as
    well.

    Note one PerNeuronValue can be present per target sheet!
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
                if len(a) > 1:
                    logger.error('ERROR: Only one PerNeuronValue value per sheet is allowed in ConnectivityPlot. Ignoring PNVs')
                    self.pnvs = None
                    break
                elif len(a) != 0:
                    self.pnvs.append(a[0])

        for conn in _connections:
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
            elif (self.parameters.reversed
                  and conn.target_name == self.parameters.sheet_name):
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
        return LinePlot(function=self.ploter, length=self.length, shared_axis=True
                 ).make_line_plot(subplotspec)

    def ploter(self, idx, gs):
        # the 
        ix = numpy.flatnonzero(self.connections[idx][:,1]==index)
        x = self.proj.pre.positions[0][self.connections[idx][ix,0]]
        y = self.proj.pre.positions[1][self.connections[idx][ix,0]]
        w = weights[idx,2]
        tx = self.connected_neuron_position[idx][0]
        ty = self.connected_neuron_position[idx][1]
        w = self.connections[idx].weights[ix,2]
        d = self.connections[idx].delays[ix,2]
        if not self.parameters.reversed:
            i = self.datastore.get_sheet_indexes(self.connections[idx].source_name,self.parameters.neuron)
            sx = self.connecting_neurons_positions[idx][0][self.connections[idx][ix,0]]
            sy = self.connecting_neurons_positions[idx][1][self.connections[idx][ix,0]]
        else:
            i = self.datastore.get_sheet_indexes(self.connections[idx].target_name,self.parameters.neuron)
            sx = self.connecting_neurons_positions[idx][0][self.connections[idx][ix,1]]
            sy = self.connecting_neurons_positions[idx][1][self.connections[idx][ix,1]]

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

                if len(pnv.values) != len(w):
                    raise ValueError('ERROR: length of colors does not match length of weights \[%d \!\= %d\]. Ignoring colors!' % (len(pnv.values), len(w)))

        gss = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs)
        
        # Plot the weights
        gs = gss[0,0]
        params = {}
        if pnv != []:
            from mozaik.tools.circ_stat import circ_mean
            (angle, mag) = circ_mean(numpy.array(pnv.values),
                                     weights=w,
                                     high=pnv.period)
            params["title"] = 'Weights: '+ str(self.connections[idx].proj_name) + "| Weighted mean: " + str(angle)
            params["colorbar_label"] =  pnv.value_name
            params["colorbar"] = True
            
            if self.connections[idx].source_name == self.connections[idx].target_name:
                params["line"] = False
                plots = [("ConnectionsPlot",ConnectionPlot(sx, sy, tx, ty, w,colors=pnv.values,period=pnv.period),gs,params)]
            else:
                params["line"] = True
                plots = [("ConnectionsPlot",ConnectionPlot(sx, sy, numpy.min(sx)*1.2, numpy.min(sy)*1.2, w,colors=pnv.values,period=pnv.period),gs,params)]
        else:
            params["title"] = 'Weights: '+ self.connections[idx].proj_name
            
            if self.connections[idx].source_name == self.connections[idx].target_name:
                params["line"] = False
                plots = [("ConnectionsPlot",ConnectionPlot(sx, sy, tx, ty, w,colors=pnv.values,period=pnv.period),gs,params)]
            else:
                params["line"] = True
                plots = [("ConnectionsPlot",ConnectionPlot(sx, sy, numpy.min(sx)*1.2, numpy.min(sy)*1.2,w),gs,params)]

        # Plot the delays
        gs = gss[1,0]
        params = {}
        params["title"]  = 'Delays: '+ self.connections[idx].proj_name
        if self.connections[idx].source_name == self.connections[idx].target_name:
            params["line"] = False
            plots.append(("DelaysPlot",ConnectionPlot(sx, sy, tx, ty, (numpy.zeros(w.shape)+0.3)*(w!=0),colors=d),gs,params))
        else:
            params["line"] = True
            plots.append(("DelaysPlot",ConnectionPlot(sx, sy, numpy.min(sx)*1.2, numpy.min(sy)*1.2,(numpy.zeros(w.shape)+0.3)*(w!=0),colors=d),gs,params))
        
        return plots
        

