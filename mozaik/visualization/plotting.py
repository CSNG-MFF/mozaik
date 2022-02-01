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
import os
import quantities as pq
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata
import mozaik.tools.units 
from parameters import ParameterSet
from collections import OrderedDict
from mozaik.tools.circ_stat import *
from mozaik.core import ParametrizedObject
from mozaik.storage import queries
from mozaik.controller import Global
from mozaik.tools.mozaik_parametrized import colapse_to_dictionary, MozaikParametrized, varying_parameters, matching_parametrized_object_params
from numpy import pi
from neo.core.analogsignal import AnalogSignal as NeoAnalogSignal
from neo.core.spiketrain import SpikeTrain as NeoSpikeTrain
from .simple_plot import StandardStyleLinePlot, SpikeRasterPlot, \
                        SpikeHistogramPlot, ConductancesPlot, PixelMovie, \
                        ScatterPlotMovie, ScatterPlot, ConnectionPlot, SimplePlot, HistogramPlot, CorticalColumnSpikeRasterPlot, OrderedAnalogSignalListPlot
from .plot_constructors import LinePlot, PerStimulusPlot, PerStimulusADSPlot, ADSGridPlot

import mozaik
logger = mozaik.getMozaikLogger()

from builtins import zip


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
        self.animation_num_frames = None
        self.frame_duration = frame_duration
        self.fig_param = fig_param if fig_param != None else OrderedDict()
        self.caption = "Caption not specified."

    def subplot(self, subplotspec):
        """
        This is the function that each Plotting instance has to implement.
        See the module documentation for more details.
        """
        raise NotImplementedError

    
    def _nip_parameters(self,plot_name,user_params):
        new_user_params = OrderedDict()
        params_to_update = OrderedDict()
        for (k,v) in user_params.items():
            l = k.split('.')
            assert len(l) > 1, "Parameter %s not matching the simple plot" % (k)
            if l[0] == plot_name or l[0] == '*':
                if len(l[1:]) >1: 
                   new_user_params['.'.join(l[1:])] = v
                else:
                   params_to_update[l[1]] = v
                
        return new_user_params,params_to_update
    
    def _handle_parameters_and_execute_plots(self,parameters,user_parameters,gs):
        d = self.subplot(gs)
        for (k,(pl,gs,p)) in d.items():
            p.update(parameters)
            ### THIS IS WRONG 'UP' DO NOT WORK        
            up = user_parameters
            for z in k.split('.'): 
                up,fp = self._nip_parameters(z,up)
                p.update(fp)
            param = p
            if isinstance(pl,SimplePlot):
                # check whether all user_parameters have been nipped to minimum 
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
            params = OrderedDict()
        self.fig = pylab.figure(facecolor='w', **self.fig_param)
        gs = gridspec.GridSpec(1, 1)
        gs.update(left=0.05, right=0.95, top=0.95, bottom=0.05)
        self._handle_parameters_and_execute_plots(OrderedDict(), params,gs[0, 0])

        # ANIMATION Handling
        if self.animation_update_functions != []:
          import matplotlib.animation as animation
          self.animation = animation.FuncAnimation(self.fig,
                                      Plotting.update_animation_function,
                                      frames = self.animation_num_frames,
                                      repeat=False,
                                      fargs=(self,),
                                      interval=self.frame_duration,
                                      blit=False)
        gs.tight_layout(self.fig)
        if self.plot_file_name:
            #if there were animations, save them
            if self.animation_update_functions != []:
                logger.info(str(animation.writers.list()))
                cwd = os.getcwd()
                os.chdir(Global.root_directory)
                # use this save command variant if you want movie for html, otherwise the pillow output for animated gif. Output through ffmpeg is extremely unstable. 
                self.animation.save(self.plot_file_name+'.html', writer='html', fps=30,bitrate=5000, extra_args=['--verbose-debug'])
                #self.animation.save(self.plot_file_name+'.gif', writer='pillow', fps=30,bitrate=5000, extra_args=['--verbose-debug']) 
                os.chdir(cwd)
            else:
                # save the analysis plot
                pylab.savefig(Global.root_directory+self.plot_file_name,transparent=True)       
            
            # and store the record
            with open(Global.root_directory+'results','a+') as f:
                 entry = {'parameters' : self.parameters, 'file_name' : self.plot_file_name, 'class_name' : str(self.__class__)}
                 f.write(str(entry)+'\n')
                 f.close()
            
        t2 = time.time()
        logger.warning(self.__class__.__name__ + ' plotting took: ' + str(t2 - t1) + 'seconds')

    def register_animation_update_function(self,auf,parent):
        self.animation_update_functions.append((auf,parent))

    @staticmethod
    def progress(current_frame, total_frames):
        print("Frame %d/%d\n" % (current_frame,total_frames)) 


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
    
    Parameters
    ----------
    centering_pnv : PerNeuronValue 
                  If not none, centered has to be true. The centering_pnv has to be a PerNeuronValue containing values in the domain corresponding to 
                  parameter `parameter_name`. The tuning curves of each neuron will be cenetered around the value in this pnv corresponding to the given neuron.
                  This will be applied also if mean is True (so the tuning curves will be centered based on the values in centering_pnv and than averaged).
    
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
      'centered' : bool, # if True it will center each set of tuning curves on the parameter value with the largest mean response across the other parameter variations
      'mean' : bool, # if True it will plot the mean tuning curve over the neurons (in case centered=True it will first center the TCs before computing the mean)
      'pool' : bool, # if True it will not plot each different value_name found in datastore on a sepparete line of plots but pool them together.
      'polar' : bool # if True polar coordinates will be used
    })

    def __init__(self, datastore, parameters, plot_file_name=None,fig_param=None,frame_duration=0,centering_pnv=None,spont_level_pnv=None):
        Plotting.__init__(self, datastore, parameters, plot_file_name, fig_param,frame_duration)
        
        self.st = []
        self.tc_dict = []
        self.pnvs = []
        self.spont_level_pnv = spont_level_pnv
        self.max_mean_response_indexes = []

        self.caption = """
                       Each column contains a tuning curve plot.
                       """

        assert queries.ads_with_equal_stimulus_type(datastore)
        assert len(self.parameters.neurons) > 0 , "ERROR, empty list of neurons specified"
        #if self.parameters.mean:
        #   assert self.parameters.centered , "Average tuning curve can be plotted only if the tuning curves are centerd"
        
        assert not (centering_pnv!=None and self.parameters.centered==False) , "Supplied centering_pnv but did not set centered to True."
        
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
                
                if self.pnvs[-1][0].value_units == mozaik.tools.units.uS:
                   a = [[d * 1000.0 for d in c] for c in a]
                
                par, val = zip(
                             *sorted(
                                zip(b,
                                    numpy.array(a))))
                dic[k] = (par,numpy.array(val))
            self.tc_dict.append(dic)
            
            if self.parameters.centered and centering_pnv==None:
               # if centering_pnv was not supplied we are centering on maximum values 
               # lets find the highest average value for the neuron
               self.max_mean_response_indexes.append(numpy.argmax(numpy.sum([a[1] for a in dic.values()],axis=0),axis=0))

            elif self.parameters.centered and centering_pnv!=None:
               # if centering_pnv was supplied we are centering on maximum values  
               period = st[0].getParams()[self.parameters.parameter_name].period
               assert period != None, "ERROR: You asked for centering of tuning curves even though the domain over which it is measured is not periodic." 
               centers = centering_pnv.get_value_by_id(self.parameters.neurons)
               self.max_mean_response_indexes.append(numpy.array([numpy.argmin(circular_dist(par,centers[i],period)) for i in range(0,len(centers))]))

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
        errors = [] 
        
        po = None if not self.parameters.polar else {'projection':'polar'}
        
        if not self.parameters.pool:
            gs = gridspec.GridSpecFromSubplotSpec(len(self.st), 1, subplot_spec=gs)
        
        for i,(dic, st, pnv) in enumerate(zip(self.tc_dict,self.st,self.pnvs)):
            
            if not self.parameters.pool:
               xs = [] 
               ys = []
               labels = []
               errors = []
                            
            period = st[0].getParams()[self.parameters.parameter_name].period
            
            if self.parameters.centered:        
               assert period != None, "ERROR: You asked for centering of tuning curves even though the domain over which it is measured is not periodic." 
            
            if self.parameters.polar:
               assert period != None, "ERROR: You asked to plot the tuning curve on polar axis even though the domain over which it is measured is not periodic." 
                
            for k in sorted(dic.keys()):    
                (par, val) = dic[k]
                error = None
                if self.parameters.mean:
                    v = []
                    for j in range(0,len(self.parameters.neurons)):
                        if self.parameters.centered:
                            vv,p = self.center_tc(val[:,j],par,period,self.max_mean_response_indexes[i][j])
                        else:
                            vv = val[:,j]
                            p = par
                        #v = v + vv
                        v.append(vv)
                    error = numpy.std(v,axis=0) / numpy.sqrt(len(self.parameters.neurons))  
                    val = numpy.mean(v,axis=0)    
                    #val = v / len(self.parameters.neurons)
                    par = p
                elif self.parameters.centered:
                    val,par = self.center_tc(val[:,idx],par,period,self.max_mean_response_indexes[i][idx])
                else:
                    val = val[:,idx]
                    
                    
                par,val = zip(*sorted(zip(numpy.array(par),val)))
                # if we have a period of pi or 2*pi
                if period != None and numpy.isclose(period,pi) and self.parameters.centered==False:
                   par = [(p-pi if p > pi/2 else p) for p in par]
                   par,val = zip(*sorted(zip(numpy.array(par),val)))
                   par = list(par)
                   val = list(val)
                   par.insert(0,-pi/2)
                   val.insert(0,val[-1])
                   if error != None:
                        error = list(error)
                        error.insert(0,error[-1])
                elif period != None and numpy.isclose(period,2*pi) and self.parameters.centered==False:
                   par = [(p-2*pi if p > pi/2 else p) for p in par]
                   par,val = zip(*sorted(zip(numpy.array(par),val)))
                   par = list(par)
                   val = list(val)
                   par.insert(0,-pi)
                   val.insert(0,val[-1])
                   if error != None:
                        error = list(error)
                        error.insert(0,error[-1])
                elif self.parameters.centered==True:                        
                     par = list(par)
                     val = list(val)
                     if numpy.isclose(par[0],-period/2):
                        par.append(period/2)
                        val.append(val[0])
                        if isinstance(error,numpy.ndarray) or isinstance(error,list):
                            error = list(error)
                            error.append(error[0])

                     if numpy.isclose(par[-1],period/2):
                        par.insert(0,-period/2)
                        val.insert(0,val[-1])
                        if isinstance(error,numpy.ndarray) or isinstance(error,list):
                            error = list(error)
                            error.append(error[0])


                elif period != None:
                    par = list(par)
                    val = list(val)
                    par.append(par[0] + period)
                    val.append(val[0])
                    if isinstance(error,numpy.ndarray) or isinstance(error,list):
                        error = list(error)
                        error.append(error[0])
                   

                if self.parameters.polar:
                   # we have to map the interval (0,period)  to (0,2*pi)
                   par = [p/period*2*numpy.pi for p in par]
                    

                xs.append(numpy.array(par))
                ys.append(numpy.array(val))
                errors.append(numpy.array(error))
                
                l = ""
                
                if self.parameters.pool:
                   if len(varying_parameters([MozaikParametrized.idd(e) for e in dic.keys()]))>0:
                        l = pnv[0].value_name + " "
                   else:
                        l = pnv[0].value_name

                                
                for p in varying_parameters([MozaikParametrized.idd(e) for e in dic.keys()]):
                    l = l + str(p) + " : " + str(MozaikParametrized.idd(k).getParamValue(p))
                labels.append(l)

            # add the spontaneous level
            if self.spont_level_pnv != None:
                if not self.parameters.mean:
                   sp_level = self.spont_level_pnv.get_value_by_id(self.parameters.neurons)[idx]
                else:
                   sp_level = numpy.mean(self.spont_level_pnv.get_value_by_id(self.parameters.neurons))
                xs.insert(0,numpy.array([min(xs[-1]),max(xs[-1])]))
                ys.insert(0,numpy.array([sp_level,sp_level]))
                labels.insert(0,'spont.')
                errors.insert(0,numpy.array([0,0]))
                
            if not self.parameters.pool:
                if not self.parameters.mean:
                    errors = None 
                params = self.create_params(pnv[0].value_name,pnv[0].value_units,i==0,i==(len(self.pnvs)-1),period,self.parameters.neurons[idx],len(xs),self.parameters.polar,labels,idx)

                plots.append(("TuningCurve_" + pnv[0].value_name,StandardStyleLinePlot(xs, ys,error=errors,subplot_kw=po),gs[i],params))   
        
        
        
        if not self.parameters.mean:
           errors = None 

        if self.parameters.pool:
           params = self.create_params('mix',self.pnvs[0][0].value_units,True,True,period,self.parameters.neurons[idx],len(xs),self.parameters.polar,labels,idx)
           if not self.parameters.polar:
              plots.append(("TuningCurve_Stacked",StandardStyleLinePlot(xs, ys,error=errors),gs,params))
           else:
              plots.append(("TuningCurve_Stacked",StandardStyleLinePlot(xs, ys,error=errors,subplot_kw=po),gs,params)) 
                
        return plots

    def create_params(self,value_name,units,top_row,bottom_row,period,neuron_id,number_of_curves,polar,labels,idx):
            params=OrderedDict()
            
            params["x_label"] = self.parameters.parameter_name
            if idx == 0:
                if units == mozaik.tools.units.uS:
                    params["y_label"] = value_name + '(nS)'
                else:
                    params["y_label"] = value_name + '(' + units.dimensionality.latex + ')'
                
            params['labels']=labels
            params['linewidth'] = 2
            #params['colors'] = [cm.jet(j/float(number_of_curves)) for j in range(0,number_of_curves)] 
            
            if top_row:
                params["title"] =  'Neuron ID: %d' % neuron_id
            

            if (not polar) and (period != None):
                    if numpy.isclose(period,pi):
                        params["x_ticks"] = [-pi/2, 0, pi/2]
                        params["x_lim"] = (-pi/2, pi/2)
                        params["x_tick_style"] = "Custom"
                        params["x_tick_labels"] = ["-$\\frac{\\pi}{2}$", "0", "$\\frac{\\pi}{2}$"]
                   
                    if numpy.isclose(period,2*pi):
                        params["x_ticks"] = [-pi, 0, pi]
                        params["x_lim"] = (-pi, pi)
                        params["x_tick_style"] = "Custom"
                        params["x_tick_labels"] = ["-$\\pi$","0", "$\\pi$"]
            elif polar:
               params["y_tick_style"] = "Custom"
               params["x_tick_style"] = "Custom"
               params["x_ticks"]  = []
               params["y_ticks"]  = []
               params["x_tick_labels"]  = []
               params["y_tick_labels"]  = []
               params['grid'] = True
               params['fill'] = False
            else:
               pass

            if not bottom_row:
                params["x_axis"] = None
                
            return params
            
    def center_tc(self,val,par,period,center_index):
           # first lets make the maximum to be at zero  
           q = int(center_index+len(val)/2 if center_index < len(val)/2 else center_index-len(val)/2)
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
        
        num_trials = numpy.max([MozaikParametrized.idd(s.annotations['stimulus']).trial for s in dsv.get_segments()])+1

        x_ticks = [0.0,float(sp[0][0].t_stop.rescale(pq.s)/2), float(sp[0][0].t_stop.rescale(pq.s))]
        
        if self.parameters.spontaneous:
           spont_sp = [s.get_spiketrain(self.parameters.neurons) for s in sorted(dsv.get_segments(null=True),key = lambda x : MozaikParametrized.idd(x.annotations['stimulus']).trial)]             
           sp = [RasterPlot.concat_spiketrains(sp1,sp2) for sp1,sp2 in zip(spont_sp,sp)]
           x_ticks = [float(spont_sp[0][0].t_start.rescale(pq.s)),0.0,float(sp[0][0].t_stop.rescale(pq.s)/2), float(sp[0][0].t_stop.rescale(pq.s))]

        d = OrderedDict()
        if self.parameters.trial_averaged_histogram:
            gs = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs)
            # first the raster
            return [ ('SpikeRasterPlot',SpikeRasterPlot([sp]),gs[:3,0],{'x_axis': False , 'x_label' :  None}),
                     ('SpikeHistogramPlot',SpikeHistogramPlot([sp],num_trials),gs[3,0],{"x_ticks": x_ticks})]
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
                    "x_label": 'time (' + vms[0].t_stop.dimensionality.latex + ')',
                    "y_label": 'Vm (' + vms[0].dimensionality.latex + ')'
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
        params = OrderedDict()
        
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
            d.append(("Activity_plot",ActivityMovie(dsv,self.parameters.sheet_activity),gs[0, 0],OrderedDict()))
            offset = 1
        else:
            gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=subplotspec)

        params = OrderedDict()
        if offset == 1:
            params['title'] = None
        params['x_label']  = None
        params['x_tick_labels'] = None
        params['x_tick_style'] ='Custom'
        params['y_label'] = 'trial'
        
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
            times = numpy.linspace(a.t_start.magnitude, a.t_stop.magnitude, len(a))
            xs.append(times)
            ys.append(a)
        
        if self.parameters.mean:
            xs = [xs[0]]
            ys = [numpy.mean(ys,axis=0)]
        
        params = OrderedDict()
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
        
        params = OrderedDict()
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
            
        return [("ConductancePlot",ConductancesPlot(exc, inh),subplotspec,OrderedDict())]



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
        params = OrderedDict()
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
        self.retinal_input = [self.retinal_input[i] for i in range(0,len(self.st)) if MozaikParametrized.idd(self.st[i]).name != 'InternalStimulus']
        self.st = [self.st[i]  for i in range(0,len(self.st)) if MozaikParametrized.idd(self.st[i]).name != 'InternalStimulus']
        
        
        
    def subplot(self, subplotspec):
        return LinePlot(function=self._ploter,
                 length=len(self.retinal_input)
                 ).make_line_plot(subplotspec)

    def _ploter(self, idx, gs):
        stimulus = MozaikParametrized.idd(self.st[idx])
        title = ''
        title = title + stimulus.name + '\n'
        for pn, pv in stimulus.getParams().items():
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
    
    bin_width : float (ms)
              In ms the width of the bins into which to sample spikes.
    
    scatter :  bool   
            Whether to plot neurons activity into a scatter plot (if True) or as an interpolated pixel image.
            
    resolution : int 
               The number of pixels into which the activity will be interpolated in case scatter = False.
               
    sheet_name: str
              The sheet for which to display the actvity movie.

    exp_time_constant: float (ms)
              Spiking can be very irregular and bursty which makes it difficult to visualize. 
              This parameter is the time-constant of the exponential with which the convolve psth, 0 means no convolution.
    """     
    
    required_parameters = ParameterSet({
          'bin_width': float,  # in ms the width of the bins into which to sample spikes
          'scatter':  bool,   # whether to plot neurons activity into a scatter plot (if True) or as an interpolated pixel image
          'resolution': int,  # the number of pixels into which the activity will be interpolated in case scatter = False
          'sheet_name': str,  # the sheet for which to display the actvity movie
          'exp_time_constant' : float, # the time-constant of the exponential with which the convolve psth, 0 means no convolution
    })

    def __init__(self, datastore, parameters, plot_file_name=None,fig_param=None,frame_duration=0,spont_level_pnv=None):
        Plotting.__init__(self, datastore, parameters, plot_file_name, fig_param,frame_duration)
        self.spont_level_pnv = spont_level_pnv


    def subplot(self, subplotspec):
        dsv = queries.param_filter_query(self.datastore,sheet_name=self.parameters.sheet_name)
        return PerStimulusPlot(dsv, function=self._ploter, title_style="Clever").make_line_plot(subplotspec)

    def _ploter(self, dsv, gs):
        sp = [s.spiketrains for s in dsv.get_segments()]

        start = sp[0][0].t_start.magnitude
        stop = sp[0][0].t_stop.magnitude
        units = sp[0][0].t_start.units
        bw = self.parameters.bin_width * pq.ms
        bw = bw.rescale(units)
        bins = numpy.arange(start, stop, bw.magnitude)
        h = []

        if self.parameters.exp_time_constant != 0:
          etc = self.parameters.exp_time_constant*pq.ms
          etc = etc.rescale(units).magnitude
          exp_kernel = numpy.exp(-(bins[:numpy.int(numpy.floor(3*etc/bw))]-start)/etc)

        for spike_trains in sp:
            hh = []
            for st in spike_trains:
                tmp = numpy.histogram(st.magnitude, bins, (start, stop))[0]/(bw.rescale(pq.s).magnitude)
                if self.parameters.exp_time_constant != 0:
                  # For the rare case where len(exp_kernel) > len(tmp)
                  # Otherwise 'same' would be sufficient
                  tmp = numpy.convolve(tmp,exp_kernel,mode='full')[:len(tmp)]
                hh.append(tmp)

                #lets make activity of each neuron relative to it's maximum activity
            h.append(numpy.array(hh))

        h = numpy.mean(h, axis=0)

        ids = dsv.get_segments()[0].get_stored_spike_train_ids()
        if self.spont_level_pnv != None:
           sl = numpy.array(self.spont_level_pnv.get_value_by_id(ids))
           h = h - 5*sl[:,numpy.newaxis]
        h[h < 0]=0   
        #h[h < 0.2*numpy.mean(numpy.mean(h))]=0

        #lets normalize against the maximum response for given neuron
        

        pos = dsv.get_neuron_positions()[self.parameters.sheet_name]

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
            for i in range(0, numpy.shape(h)[1]):
                movie.append(griddata((posx, posy),
                                      h[:, i],
                                      (xi[None, :], yi[:, None]),
                                      method='nearest'))

            movie = numpy.array(movie)
            movie[numpy.isnan(movie)]=0
            
            return [("PixelMovie",PixelMovie(movie, movie.max()/2),gs,{'x_axis':False, 'y_axis':False})]
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
         pnvs = dsv.get_analysis_result(identifier='PerNeuronValue')            
         
        
         if self.parameters.cortical_view:
            assert len(pnvs) <= 1, logger.error("We can only display single stimulus parametrization in cortical view, but you have suplied multiple in datastore.")
            pnv = pnvs[0]
            pos = self.dsv.get_neuron_positions()[pnv.sheet_name]                            
            posx = pos[0,self.datastore.get_sheet_indexes(pnv.sheet_name,pnv.ids)]
            posy = pos[1,self.datastore.get_sheet_indexes(pnv.sheet_name,pnv.ids)]
            values = pnv.values
            if pnv.period != None:
                periodic = True
                period = pnv.period
            else:
                periodic = False
                period = None
            params = OrderedDict()
            params["x_label"] = 'x'
            params["y_label"] = 'y'
            params["colorbar_label"] = pnv.value_units.dimensionality.latex
            params["colorbar"]  = True

            return [("ScatterPlot",ScatterPlot(posx, posy, values, periodic=periodic,period=period),gs,params)]
         else:
            assert queries.ads_with_equal_stimulus_type(dsv) , logger.error('Warning sheet name and value name does\'t seem to uniquely identify set of PerNeuronValue ADS with the same stimulus type')
            params = OrderedDict()
            params["y_label"] = '# neurons'

            varying_stim_parameters = sorted(varying_parameters([MozaikParametrized.idd(pnv.stimulus_id) for pnv in pnvs]))        
            a = sorted([(','.join([p + ' : ' + str(getattr(MozaikParametrized.idd(pnv.stimulus_id),p)) for p in varying_stim_parameters]),pnv) for pnv in pnvs],key=lambda x: x[0])
            
            if len(a) > 1:
                return [("HistogramPlot",HistogramPlot([z[1].values for z in a],labels=[z[0] for z in a]),gs,params)]
            else:
                return [("HistogramPlot",HistogramPlot([pnvs[0].values]),gs,params)]





class PerNeuronValueScatterPlot(Plotting):
    """
    Takes each pair of PerNeuronValue ADSs in the datastore and plots a scatter plot of each such pair.
    It defines line of plots named: 'ScatterPlot.Plot0' ... 'ScatterPlot.PlotN
    """
    
    required_parameters = ParameterSet({
          'only_matching_units':bool,  # only plot combinations of PNVs that have the same value units.
          'ignore_nan' : bool, # if True NaNs will be removed from the data. In general if there are NaN in the data and this is False it will not be displayed correctly.
          'lexicographic_order': bool # Whether to order the ads in each pair by the descending lexicographic order of their parameter 'value_name' before plotting
    })
    
    
    def __init__(self, datastore, parameters, plot_file_name=None,fig_param=None):
        Plotting.__init__(self, datastore, parameters, plot_file_name, fig_param)

        self.pairs = []
        self.sheets = []
        for sheet in datastore.sheets():
            pnvs = datastore.get_analysis_result(identifier='PerNeuronValue',sheet_name=sheet)
            if len(pnvs) < 2:
               raise ValueError('At least 2 DSVs have to be provided') 
            for i in range(0,len(pnvs)):
                for j in range(i+1,len(pnvs)):
                    if (pnvs[i].value_units == pnvs[j].value_units) or not self.parameters.only_matching_units:
                        if pnvs[j].value_name < pnvs[i].value_name and self.parameters.lexicographic_order:
                            self.pairs.append((pnvs[j],pnvs[i]))
                        else:
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
            x_label += '\n' + str(p) + " = " + str(pair[0].getParamValue(p))
            y_label += '\n' + str(p) + " = " + str(pair[1].getParamValue(p))
        
        for p in p2:
            x_label += '\n' + str(p) + " = " + str(MozaikParametrized.idd(pair[0].stimulus_id).getParamValue(p))
            y_label += '\n' + str(p) + " = " + str(MozaikParametrized.idd(pair[1].stimulus_id).getParamValue(p))
        
        params = OrderedDict()
        params["x_label"] = x_label
        params["y_label"] = y_label
        params["title"] = self.sheets[idx]
        if pair[0].value_units != pair[1].value_units or pair[1].value_units == pq.dimensionless:
           params["equal_aspect_ratio"] = False
        
        ids = list(set(pair[0].ids) & set(pair[1].ids))
        x = pair[0].get_value_by_id(ids)
        y = pair[1].get_value_by_id(ids)
        if self.parameters.ignore_nan:
            idxs = numpy.logical_not(numpy.logical_or(numpy.isnan(numpy.array(x)),numpy.isnan(numpy.array(y))))
            x = numpy.array(x)[idxs]
            y = numpy.array(y)[idxs]
        return [("ScatterPlot",ScatterPlot(x,y),gs,params)]

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
        
        for conn in _connections:
            if not self.parameters.reversed and conn.source_name == self.parameters.sheet_name:
                # add outgoing projections from sheet_name
                self.connecting_neurons_positions.append(
                            datastore.get_neuron_positions()[conn.target_name])
                z = datastore.get_neuron_positions()[conn.source_name]
                idx = self.datastore.get_sheet_indexes(conn.source_name,self.parameters.neuron)
                self.connected_neuron_position.append(
                            (z[0][idx],
                             z[1][idx]))
                self.connections.append(conn)
            elif (self.parameters.reversed and conn.target_name == self.parameters.sheet_name):
                # add incomming projections from sheet_name
                self.connecting_neurons_positions.append(
                            datastore.get_neuron_positions()[conn.source_name])
                z = datastore.get_neuron_positions()[conn.target_name]
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
            ixa = numpy.flatnonzero(numpy.array(self.connections[idx].weights)[:,0]==index)
            ix = numpy.array(self.connections[idx].weights)[:,1][ixa].astype(int)
        else:
            index = self.datastore.get_sheet_indexes(self.connections[idx].target_name,self.parameters.neuron)
            ixa = numpy.flatnonzero(numpy.array(self.connections[idx].weights)[:,1]==index)
            ix = numpy.array(self.connections[idx].weights)[:,0][ixa].astype(int)
        
        assert all(numpy.array(self.connections[idx].weights)[:,0] == numpy.array(self.connections[idx].delays)[:,0])
        assert all(numpy.array(self.connections[idx].weights)[:,1] == numpy.array(self.connections[idx].delays)[:,1])
        
        sx = self.connecting_neurons_positions[idx][0][ix]
        sy = self.connecting_neurons_positions[idx][1][ix]
        w = numpy.array(self.connections[idx].weights)[ixa,2]
        d = numpy.array(self.connections[idx].delays)[ixa,2]

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
        params = OrderedDict()
        
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
                                     weights=numpy.abs(w),
                                     high=pnv.period)
            params["title"] = str(self.connections[idx].proj_name) + "\n Mean: " + str(angle) + '\nNum conn/mean weight/total weight:' +str(len(w)) + '/' + str(numpy.mean(w)) + '/' + str(numpy.sum(w)) + '\n' + str(w[:10 ])
            params["colorbar_label"] =  pnv.value_name
            params["colorbar"] = True

            if self.connections[idx].source_name == self.connections[idx].target_name:
                params["line"] = False
                plots = [("ConnectionsPlot",ConnectionPlot(sx, sy, tx, ty, numpy.abs(w),colors=pnv.get_value_by_id(self.datastore.get_sheet_ids(pnv.sheet_name,ix)),period=pnv.period),gs,params)]
            else:
                params["line"] = True
                plots = [("ConnectionsPlot",ConnectionPlot(sx, sy, numpy.min(sx)*1.2, numpy.min(sy)*1.2, numpy.abs(w),colors=pnv.get_value_by_id(self.datastore.get_sheet_ids(pnv.sheet_name,ix)),period=pnv.period),gs,params)]
        else:
            params["title"] = 'Weights: '+ self.connections[idx].proj_name
            
            if self.connections[idx].source_name == self.connections[idx].target_name:
                params["line"] = False
                plots = [("ConnectionsPlot",ConnectionPlot(sx, sy, tx, ty, numpy.abs(w)),gs,params)]
            else:
                params["line"] = True
                plots = [("ConnectionsPlot",ConnectionPlot(sx, sy, numpy.min(sx)*1.2, numpy.min(sy)*1.2,numpy.abs(w)),gs,params)]


        # Plot the delays
        gs = gss[1,0]
        params = OrderedDict()
        params["x_lim"] = (-xs/2.0,xs/2.0)
        params["y_lim"] = (-ys/2.0,ys/2.0)
        
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


class CorticalColumnRasterPlot(Plotting):
    """ 
    It plots raster plots of spikes stored in the recordings.
    It assumes a datastore with recordings to a set of stimuli in different sheets. 
    It will plot a line of raster plots, one per each stimulus, showing a 'cortical column view' of the spike rasters across all sheets present in the data_store.
    All trials of the same stimulus are superimposed over each other.
    
    Defines 'CorticalColumnRasterPlot.Plot0' ... 'CorticalColumnRasterPlot.PlotN'
    
    Other parameters
    ----------------
    spontaneous : bool
                Whether to also show the spontaneous activity the preceded the stimulus.

    sheet_names : list
                the list and order in which the sheets should be plotted
    
    labels : list
                the list and order in which the sheets should be plotted
    
    colors : list 
           the colors to give to the spikes in the sheets. 
           
           
    NOTES
    -----
    
    spontaneous doesn't work yet
    """
    
    required_parameters = ParameterSet({
        'spontaneous' : bool, # whether to also show the spontaneous activity the preceded the stimulus     
        'sheet_names' : list, # the list and order in which the sheets should be plotted, empty list means all sheets in datastore will be taken
        'labels' : list, # the labels to give to the sheets in the plot, empty list means the names of the sheets will be used as labels
        'colors' : list, # the colors to give to the spikes in the sheets, empty list means all spikes will be given black color
        'neurons' : list # list of list of neuron ids, one list per each sheet, which to display. If empty all recorded neurons are displayed
    })

    def subplot(self, subplotspec):
        
        if self.parameters.sheet_names == []:
            self.sheet_names = dsv.sheets()
        else:
            self.sheet_names = self.parameters.sheet_names
        
        if self.parameters.labels == []:
            self.labels = self.sheet_names
        else:
            self.labels = self.parameters.labels
      
        if self.parameters.colors == []:
            self.colors =  ['#000000' for i in self.sheets]
        else:
            self.colors = self.parameters.colors
            
        assert len(self.sheet_names) == len(self.labels) == len(self.colors) , "Parameter <sheet_names> , <labels> and <colors> have to have the same length or be empty lists."
        
        if len(self.parameters.neurons) != 0:
                assert len(self.sheet_names) == len(self.parameters.neurons), "Parameter <sheet_names> , <neurons> have to have the same length"
        
        
        return PerStimulusPlot(self.datastore,single_trial=True, function=self._ploter).make_line_plot(subplotspec)

    def _ploter(self, dsv,gs):
        
        sp = []
        for i,sn in enumerate(self.sheet_names):
            a = queries.param_filter_query(dsv,sheet_name=sn).get_segments()
            assert len(a) == 1
            if len(self.parameters.neurons) != 0:
                sp.append(a[0].get_spiketrain(self.parameters.neurons[i]))
            else:
                sp.append(a[0].get_spiketrains())
        
        #if self.parameters.spontaneous:
        #   spont_sp = [s.get_spiketrain(self.parameters.neurons) for s in sorted(dsv.get_segments(null=True),key = lambda x : MozaikParametrized.idd(x.annotations['stimulus']).trial)]             
        #   sp = [RasterPlot.concat_spiketrains(sp1,sp2) for sp1,sp2 in zip(spont_sp,sp)]
        #   x_ticks = [float(spont_sp[0][0].t_start),0.0,float(sp[0][0].t_stop/2), float(sp[0][0].t_stop)]
        return [('CorticalColumnRasterPlot',CorticalColumnSpikeRasterPlot(sp),gs,{"labels": self.labels, "colors" : self.colors})]


class PlotTemporalTuningCurve(Plotting):
    """
    Plots tuning curves, one plot in line per each neuron. This plotting function assumes a set of AnalogSignalList 
    ADSs in the datastore associated with certain stimulus type. It will plot
    the values stored in these  AnalogSignalList instances (corresponding to neurons in `neurons`) across the 
    varying parameter `parameter_name` of thier associated stimuli. 

    Parameters
    ----------
    centering_pnv : PerNeuronValue 
                  If not none, centered has to be true. The centering_pnv has to be a PerNeuronValue containing values in the domain corresponding to 
                  parameter `parameter_name`. The tuning curves of each neuron will be cenetered around the value in this pnv corresponding to the given neuron.
                  This will be applied also if mean is True (so the tuning curves will be centered based on the values in centering_pnv and than averaged).
    
    Other parameters
    ----------------
    neurons : list
            List of neuron ids for which to plot the tuning curves.
            
    sheet_name : str
            From which layer to plot the tuning curves.
               
    parameter_name : str
                   The parameter_name through which to plot the tuning curve.

    mean : bool 
         If True it will plot the mean tuning curve over all neurons (in case centered=True it will first center the TCs before computing the mean)
    
            
    Defines 'TuningCurve_' + value_name +  '.Plot0' ... 'TuningCurve_' + value_name +  '.Plotn'
    where n goes through number of neurons, and value_name creates one row for each value_name found in the different PerNeuron found
    """

    required_parameters = ParameterSet({
      'neurons':  list,  # which neurons to plot
      'sheet_name': str,  # from which layer to plot the tuning curves
      'parameter_name': str,  # the parameter_name through which to plot the tuning curve
      'mean' : bool, # if True it will plot the mean tuning curve over the neurons (in case centered=True it will first center the TCs before computing the mean)
    })

    def __init__(self, datastore, parameters, plot_file_name=None,fig_param=None,frame_duration=0,centering_pnv=None):
        Plotting.__init__(self, datastore, parameters, plot_file_name, fig_param,frame_duration)
        
        self.st = []
        self.tc_dict = []
        self.asls = []
        self.center_response_indexes = None
        assert queries.ads_with_equal_stimulus_type(datastore)
        assert len(self.parameters.neurons) > 0 , "ERROR, empty list of neurons specified"
        
        dsvs = queries.partition_analysis_results_by_parameters_query(self.datastore,parameter_list=['y_axis_name'],excpt=True)
        for dsv in dsvs:
            dsv = queries.param_filter_query(dsv,identifier='AnalogSignalList',sheet_name=self.parameters.sheet_name)
            assert matching_parametrized_object_params(dsv.get_analysis_result(), params=['y_axis_name'])
            self.asls.append(dsv.get_analysis_result())
            # get stimuli
            st = [MozaikParametrized.idd(s.stimulus_id) for s in self.asls[-1]]
            self.st.append(st)
            dic = colapse_to_dictionary([z.get_asl_by_id(self.parameters.neurons) for z in self.asls[-1]],st,self.parameters.parameter_name)

            #sort the entries in dict according to the parameter parameter_name values 
            for k in  dic:
                (b, a) = dic[k]
                if self.asls[-1][0].y_axis_units == mozaik.tools.units.uS:
                   a = [[d * 1000.0 for d in c] for c in a]
                
                par, val = zip(
                             *sorted(
                                zip(b,
                                    numpy.array(a))))
                dic[k] = (par,numpy.array(val))
            self.tc_dict.append(dic)
            
        if centering_pnv!=None:
               # if centering_pnv was supplied we are centering on maximum values  
               period = st[0].getParams()[self.parameters.parameter_name].period
               assert period != None, "ERROR: You asked for centering of tuning curves even though the domain over which it is measured is not periodic." 
               centers = centering_pnv.get_value_by_id(self.parameters.neurons)
               self.center_response_indexes.append(numpy.array([numpy.argmin(circular_dist(par,centers[i],period)) for i in range(0,len(centers))]))
            
    def subplot(self, subplotspec):
        if self.parameters.mean:
            return LinePlot(function=self._ploter, length=1).make_line_plot(subplotspec)
        else:
            return LinePlot(function=self._ploter, length=len(self.parameters.neurons)).make_line_plot(subplotspec)

    def _ploter(self, idx, gs):
        plots  = []
        xs = []
        ys = []
        errors = [] 
        
        gs = gridspec.GridSpecFromSubplotSpec(len(self.st), 1, subplot_spec=gs)
        
        for i,(dic, st, asl) in enumerate(zip(self.tc_dict,self.st,self.asls)):
            xs = [] 
            ys = []
            period = st[0].getParams()[self.parameters.parameter_name].period
            for k in sorted(dic.keys()):    
                (par, val) = dic[k]
                error = None
                if self.parameters.mean:
                    v = []
                    for j in range(0,len(self.parameters.neurons)):
                        if self.centered_response_indexes != None:
                            vv,p = self.center_tc(val[:,j],par,period,self.centered_response_indexes[i][j])
                        else:
                            vv = val[:,j]
                            p = par
                        v.append(vv)
                    error = numpy.std(v,axis=0) / numpy.sqrt(len(self.parameters.neurons))  
                    val = numpy.mean(v,axis=0)    
                    par = p
                else:
                    val = val[:,idx]

                par,val = zip(*sorted(zip(numpy.array(par),val)))

                # if we have a period of pi or 2*pi
                if period==pi and self.centered_response_indexes==None:
                   par = [(p-pi if p > pi/2 else p) for p in par]
                   par,val = zip(*sorted(zip(numpy.array(par),val)))
                   par = list(par)
                   val = list(val)
                   par.insert(0,-pi/2)
                   val.insert(0,val[-1])
                   if error != None:
                        error = list(error)
                        error.insert(0,error[-1])
                elif period==2*pi and self.centered_response_indexes==None:
                   par = [(p-2*pi if p > pi/2 else p) for p in par]
                   par,val = zip(*sorted(zip(numpy.array(par),val)))
                   par = list(par)
                   val = list(val)
                   par.insert(0,-pi)
                   val.insert(0,val[-1])
                   if error != None:
                        error = list(error)
                        error.insert(0,error[-1])
                elif period != None:
                    par = list(par)
                    val = list(val)
                    par.append(par[0] + period)
                    val.append(val[0])
                    if error != None:
                        error = list(error)
                        error.append(error[0])
                   
                xs.append(numpy.squeeze(numpy.array(par)))
                ys.append(numpy.squeeze(numpy.array(val)))
            
            if not self.parameters.mean:
                errors = None 
            params = self.create_params(asl[0].y_axis_name,asl[0].y_axis_units,i==0,i==(len(self.asls)-1),period,self.parameters.neurons[idx],numpy.squeeze(xs),idx)
            plots.append(("TuningCurve_" + asl[0].y_axis_name,OrderedAnalogSignalListPlot(numpy.squeeze(ys), numpy.squeeze(xs)),gs[i],params))   
        
        return plots

    def create_params(self,y_axis_name,units,top_row,bottom_row,period,neuron_id,signal_labels,idx):
            params=OrderedDict()
            
            if idx==0:
                params["y_label"] = self.parameters.parameter_name
            
            params["colorbar"] = True
            params["x_label"] = "time (ms)"
            
            if units == mozaik.tools.units.uS:
                params["colorbar_label"] = y_axis_name + ' (nS)'
            else:
                params["colorbar_label"] = y_axis_name + ' (' + units.dimensionality.latex + ')'

            if top_row:
                params["title"] =  'Neuron ID: %d' % neuron_id
            
            if period == pi:
                params["y_ticks"] = [-pi/2, 0, pi/2]
                params["y_lim"] = (--pi/2, pi/2)
                params["y_tick_style"] = "Custom"
                params["y_tick_labels"] = ["-$\\frac{\\pi}{2}$", "0", "$\\frac{\\pi}{2}$"]
            elif period == 2*pi:
                params["y_ticks"] = [-pi, 0, pi]
                params["y_lim"] = (-pi, pi)
                params["y_tick_style"] = "Custom"
                params["y_tick_labels"] = ["-$\\pi$","0", "$\\pi$"]
            else:
                params["y_ticks"] = [0, len(signal_labels)-1]
                #params["y_lim"] = (-pi, pi)
                params["y_tick_style"] = "Custom"
                params["y_tick_labels"] = [signal_labels[0],signal_labels[-1]]
            
            if not bottom_row:
                params["x_axis"] = None

            if not idx==0:
                params["y_axis"] = None

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
