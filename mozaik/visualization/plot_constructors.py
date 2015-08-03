"""
This module contains classes that assist with construction of complicated arrangements of plots.
A typical example is a class that helps with creating a line of plots with a common y axis.
"""

import param
from param.parameterized import Parameterized
from mozaik.storage.queries import partition_by_stimulus_paramter_query, partition_analysis_results_by_stimulus_parameters_query,param_filter_query
import matplotlib.gridspec as gridspec
from mozaik.tools.mozaik_parametrized import MozaikParametrized, varying_parameters, parameter_value_list
import mozaik
import numpy

logger = mozaik.getMozaikLogger()


class LinePlot(Parameterized):
        """
        Plot multiple plots with common x or y axis in a row or column. The user has to specify 
        the function. This one has to return a list of tuples, each containing:
            * a name of a plot
            * a Plotting or SimplePlot instance
            * the simple_plot parameters that should be passed on
        
        Assuming each *function* returns a list of plots with names PlotA,...PlotX
        The LinePlot will create a list of plots named:
                    PlotA.plot0 ... PlotA.plotN
                    PlotX.plot0 ... PlotX.plotN
        where N is defined by the length parameter.
        User can this way target the plots with parameters as desribed in the Plotting class.
        """
        horizontal = param.Boolean(default=True, instantiate=True,
                                   doc="Should the line of plots be horizontal or vertical")

        shared_axis = param.Boolean(default=True, instantiate=True,
                                    doc="should the x axis or y axis (depending on the horizontal flag) considered shared")

        shared_lim = param.Boolean(default=False, instantiate=True,
                                   doc="should the limits of the x axis or y axis (depending on the horizontal flag) be considered shared")
                                   
        length = param.Integer(default=0, instantiate=True,
                               doc="how many plots will there be")
        function = param.Callable(doc="The function that should be called to plot individual plots. It should accept three parameters: self, index in the line, gridspec object into which to plot the plot, the simple_plot parameters")
        max_length = param.Integer(default=10, instantiate=True,
                                   doc="The maximum # plots actually displayed")
        extra_space_top = param.Number(default=0.0, instantiate=True,
                                       doc="Space to be reserved on top of the subplot, defined as fraction of the subplot.")
        extra_space_right = param.Number(default=0.0, instantiate=True,
                                         doc="Space to be reserved on the right side of the subplot, defined as fraction of the subplot.")

        def make_line_plot(self, subplotspec):
            """
            Call to execute the line plot.

            Parameters
            ----------
            subplotspec : subplotspec
                        Is the subplotspec into which the whole lineplot is to be plotted.
            """
            if not self.length:
                raise ValueError('Length not specified')
                

            l = numpy.min([self.max_length, self.length])
            subplotspec = gridspec.GridSpecFromSubplotSpec(
                                    100, 100, subplot_spec=subplotspec
                                )[int(100*self.extra_space_top):100, 0:int(100*(1-self.extra_space_right))]

            if self.horizontal:
                gs = gridspec.GridSpecFromSubplotSpec(1, l, subplot_spec=subplotspec)
            else:
                gs = gridspec.GridSpecFromSubplotSpec(l, 1, subplot_spec=subplotspec)
            
            d = {}
            params = {}
            for idx in xrange(0, l):
                if idx > 0 and self.shared_axis and self.horizontal:
                    params["y_label"]=None
                    if self.shared_lim:
                        params["y_axis"] = False

                if (idx < l-1) and self.shared_axis and not self.horizontal:
                    
                    params["x_label"]=None
                    if self.shared_lim:
                        params["x_axis"] = False

                if self.horizontal:
                    li = self._single_plot(idx,gs[0, idx])
                    for (name,plot,gss,par) in li:
                        par.update(params)
                        d[name + '.' + 'plot' + str(idx)] = (plot,gss,par)
                    
                else:
                    li = self._single_plot(idx,gs[idx, 0])
                    for (name,plot,gss,par) in li:
                        par.update(params)
                        d[name + '.' + 'plot' +str(idx)] = (plot,gss,par)
            return d
                    
        def _single_plot(self, idx,gs):
            return self.function(idx,gs)


class PerDSVPlot(LinePlot):
    """
    This is a LinePlot that automatically partitions the datastore view based
    on some rules, and than executes the individual plots on the line over
    the individual DSVs.

    Unlike LinePlot the function parameter should accept as the first arguemnt
    DSV not the index of the plot on the line.

    The partition dsvs function should perform the partitioning.
    """

    function = param.Callable(instantiate=True,
                              doc="The function that should be called to plot individual plots. It should accept three parameters: self, the DSV from which to generate the plot, gridspec object into which to plot the plot, the simple_plot parameters")

    def  __init__(self, datastore, **params):
        LinePlot.__init__(self, **params)
        self.datastore = datastore
        self.dsvs = self.partiotion_dsvs()
        self.length = len(self.dsvs)

    def partiotion_dsvs(self):
        raise NotImplementedError

    def _single_plot(self, idx, gs):
        return self.function(self.dsvs[idx],gs)


class PerStimulusPlot(PerDSVPlot):
    """
    Line plot where each plot corresponds to stimulus with the same parameter
    except trials.

    The self.dsvs will contain the datastores you want to plot in each of the
    subplots - i.e. all recordings in the given datastore come from the same
    stimulus of the same parameters (except for the trial parameter if `single_trial` is False).

    PerStimulusPlot provides several automatic titling of plots based on the
    stimulus name and parameters. The currently supported styles are:

    "None" - No title

    "Standard" - Simple style where the Stimulus name is plotted on one line
                 and the parameter values on the second line

    "Clever" - This style is valid only for cases where only stimuli of the
               same type are present in the supplied DSV.
               If the style is set to Clever but the conditions doesn't hold it
               falls back to Standard and emits a warning.
               In this case the name of the stimulus and all parameters which
               are the same for all stimuli in DSV are not displayed. The
               remaining parameters are shown line after line in the format
               'stimulus: value'.
               Of course trial parameter is ignored.
    """
    title_style = param.String(default="Clever", instantiate=True,
                               doc="The style of the title")

    def  __init__(self, datastore,single_trial=False, **params):
        self.single_trial = single_trial
        PerDSVPlot.__init__(self, datastore, **params)
        ss = self._get_stimulus_ids()
        assert ss != [], "Error, empty datastore!"
        if self.title_style == "Clever":
            stimulus = MozaikParametrized.idd(ss[0])
            for s in ss:
                s = MozaikParametrized.idd(s)
                if s.name != stimulus.name:
                    logger.warning('Datastore does not contain same type of stimuli: changing title_style from Clever to Standard')
                    self.title_style = "Standard"
                    break

        # lets find parameter indexes that vary if we need 'Clever' title style
        if self.title_style == "Clever":
            self.varied = varying_parameters([MozaikParametrized.idd(s) for s in ss])
            if not self.single_trial:
                self.varied = [x for x in self.varied if x != 'trial']
            
            
        if self.title_style == "Standard":
            self.extra_space_top = 0.07
        if self.title_style == "Clever":
            self.extra_space_top = len(self.varied)*0.005
        
    def _get_stimulus_ids(self):
        return self.datastore.get_stimuli()
         
    def partiotion_dsvs(self):
        if not self.single_trial:
           return partition_by_stimulus_paramter_query(self.datastore,['trial'])
        else:
           return partition_by_stimulus_paramter_query(self.datastore,[]) 

    def _single_plot(self, idx,gs):
        title = self.title(idx)
        li = PerDSVPlot._single_plot(self, idx,gs)
        for (name,plot,gs,par) in li:
            if title != None:
                par.setdefault('title',title)
        return li

    def title(self, idx):
        return self._title(MozaikParametrized.idd(self.dsvs[idx].get_stimuli()[0]))
    
    def _title(self,stimulus):
        if self.title_style == "None":
            return None

        if self.title_style == "Standard":
            title = ''
            title = title + stimulus.name + '\n'
            for pn, pv in stimulus.get_param_values():
                title = title + pn + ' : ' + str(pv) + '\n'
            return title

        if self.title_style == "Clever":
           title = ''
           for pn in self.varied:
               title = title + str(pn) + ' : ' + str(getattr(stimulus,pn)) + '\n' 
           return title


class PerStimulusADSPlot(PerStimulusPlot):
      """
      As PerStimulusPlot, but partitions the ADS not recordings. 
      """
      def _get_stimulus_ids(self):
          return [MozaikParametrized.idd(ads.stimulus_id) for ads in self.datastore.get_analysis_result()]
      
      def title(self, idx):
          return self._title([MozaikParametrized.idd(ads.stimulus_id) for ads in self.dsvs[idx].get_analysis_result()][0])

      def partiotion_dsvs(self):
          return partition_analysis_results_by_stimulus_parameters_query(self.datastore,['trial'])



class ADSGridPlot(Parameterized):
    """
    Set of plots that are placed on a grid, that vary in two parameters and can have shared x or y axis (only at the level of labels for now).
    

    Plot multiple plots with common x and y axis in a grid. 
    
    The user has to specify a plotting function (the function parameter) which has to return a list of tuples, each containing:
        * a name of a plot
        * a Plotting or SimplePlot instance
        * the simple_plot parameters that should be passed on
    
    The ADSGridPlot, automaticall filters the datastore such that the function always receives a DSV where the two parameters are already fixed to the right values.
    
    Assuming each *function* returns a list of plots with names PlotA,...PlotX
    The LinePlot will create a list of plots named:
                PlotA.plot[0,0] ... PlotA.plot[n,m]
                PlotX.plot[0,0] ... PlotX.plot[n,m]
    where n,m is defined by the number of values the x and y _xis_parameter has in the datastore.
    User can this way target the plots with parameters as desribed in the Plotting class.
    """
    

    x_axis_parameter = param.String(default=None,instantiate=True,doc="The parameter whose values should be iterated along x-axis")
    y_axis_parameter = param.String(default=None,instantiate=True,doc="The parameter whose values should be iterated along y-axis")
    
    function = param.Callable(doc="The function that should be called to plot individual plots. It should accept three parameters: self, DSV, gridspec object into which to plot the plot, the simple_plot parameters")

    shared_axis = param.Boolean(default=True, instantiate=True,
                                    doc="should the x axis or y axis (depending on the horizontal flag) considered shared")

    shared_lim = param.Boolean(default=False, instantiate=True,
                                   doc="should the limits of the x axis or y axis (depending on the horizontal flag) be considered shared")

    extra_space_top = param.Number(default=0.0, instantiate=True,
                                       doc="Space to be reserved on top of the subplot, defined as fraction of the subplot.")
    extra_space_right = param.Number(default=0.0, instantiate=True,
                                         doc="Space to be reserved on the right side of the subplot, defined as fraction of the subplot.")

    def  __init__(self, datastore, **params):
        Parameterized.__init__(self, **params)
        self.datastore = datastore

        ### lets first find the values of the two parameters in the datastore
        self.x_axis_values = list(parameter_value_list(datastore.get_analysis_result(),self.x_axis_parameter))
        self.y_axis_values = list(parameter_value_list(param_filter_query(datastore,**{self.x_axis_parameter:self.x_axis_values[0]}).get_analysis_result(),self.y_axis_parameter))
        
        ### and verify it forms a grid
        for v in self.x_axis_values:
            assert set(self.y_axis_values) == parameter_value_list(param_filter_query(datastore,**{self.x_axis_parameter:v}).get_analysis_result(),self.y_axis_parameter)

        
    
    def make_grid_plot(self, subplotspec):
        """
        Call to execute the grid plot.

        Parameters
        ----------
        subplotspec : subplotspec
                    Is the subplotspec into which the whole lineplot is to be plotted.
        """
        subplotspec = gridspec.GridSpecFromSubplotSpec(
                                100, 100, subplot_spec=subplotspec
                            )[int(100*self.extra_space_top):100, 0:int(100*(1-self.extra_space_right))]
       
        gs = gridspec.GridSpecFromSubplotSpec(len(self.y_axis_values),len(self.x_axis_values),subplot_spec=subplotspec)
        
        params = {}
        d = {}
        for i in xrange(0,len(self.x_axis_values)):
            for j in xrange(0,len(self.y_axis_values)):
                if i > 0 and self.shared_axis:
                    params["y_label"]=None
                    if self.shared_lim:
                        params["y_axis"] = False
                else:
                    params["y_label"]=self.y_axis_values[j]
                    
                if j < len(self.y_axis_values)-1 and self.shared_axis:
                    params["x_label"]=None
                    if self.shared_lim:
                        params["x_axis"] = False
                else:
                    params["x_label"]=self.x_axis_values[i]

                dsv = param_filter_query(self.datastore,**{self.x_axis_parameter:self.x_axis_values[i],self.y_axis_parameter:self.y_axis_values[j]})
                li = self._single_plot(dsv,gs[j,i])
                for (name,plot,gss,par) in li:
                    par.update(params)
                    d[name + '.' + 'plot[' +str(i) + ',' +str(j) + ']'] = (plot,gss,par)
        return d
                
    def _single_plot(self, dsv,gs):
        return self.function(dsv,gs)
