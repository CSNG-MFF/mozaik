"""
See :mod:`mozaik.visualization` for more general documentation.
"""

import mozaik.visualization.helper_functions as phf
import pylab
import numpy
import mozaik
import mozaik.tools.units 

logger = mozaik.getMozaikLogger()


class SimplePlot(object):
    """
    The low level plotting API based around matplotlib.
    
    Each simple plot is assumed to create a single axis - this happens in this class.
    Each user defined low-level plotting function is supposed to derive from this class
    and implement the four abstract methods: `pre_axis_plot`, `pre_plot`, `plot`, `post_plot`, to 
    direct the styling and decoration of the plot, and call the actual matplotlib plotting functions in the 
    `plot` function.
    
    The main policy that this API declares is that any overwriting of the standard style default
    values by the `Plotting` mechanisms should take precedence over those done
    by specific instances of `SimplePlot`. In order to enforce this precedence,
    the modifiable parameters of the classes should be stored in the common
    dictionary `parameters`. Each class derived from `SimplePlot` should add its
    modifiable parameters into the `parameters` dictionary in its constructor.
    We will document the modifiable parameters the given class declares 
    in the 'Other parameters' section.
    
    The kwargs passed to the instances of `SimplePlot` from the `Plotting`
    mechanisms will be stored. At the beginning of the __call__ the dictionary
    will be updated with these stored kwargs, and the updated parameters (note
    not all have to be present) will be marked. These marked parameters will not
    be then modifiable any further. In order to do so, the `parameters`
    dictionary is accessible via the __getattr__ and __setattr__ functions.

    *Note, for this reason all `SimplePlot` classes need to take care that none
    of the modifiable attributes is also defined as a class attribute.*
    """
    def pre_axis_plot(self):
        """
        The function that is executed before the axis is created.
        """
        raise NotImplementedError


    def pre_plot(self):
        """
        The function that is executed before the plotting but after the axis is created.
        """
        raise NotImplementedError

    def plot(self):
        """
        The function that has to perform the low level plotting. This is where the 
        main matplotlib plot calls will happen.
        """
        raise NotImplementedError

    def post_plot(self):
        """
        The function that is executed after the plotting.
        """
        raise NotImplementedError

    def __init__(self,subplot_kw=None):
        self.parameters = {}  # the common modifiable parameter dictionary
        self.subplot_kw = subplot_kw
        
        
    def __getattr__(self, name):
        if name == 'parameters':
            return self.__dict__[name]
        if name in self.__dict__['parameters'] and name in self.__dict__:
            raise AttributeError("Error, attribute %s both in __dict__ and self.parameters" % (name))
        elif name in self.__dict__['parameters']:
            return self.__dict__['parameters'][name]
        elif name in self.__dict__:
            return self.__dict__[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name == 'parameters':
            self.__dict__[name] = value
            return

        if name in self.__dict__['parameters'] and name in self.__dict__:
            raise AttributeError("Error, attribute both in __dict__ and self.parameters")
        elif name in self.__dict__['parameters']:
            if not self.__dict__['modified'][name]:
                self.__dict__['parameters'][name] = value
        else:
            self.__dict__[name] = value

    def __call__(self, gs,params,plotting_parent):
        """
        Calls all the plotting styling and execution functions in the right order.
        """
        self.plotting_parent=plotting_parent
        self.update_params(params)
        self.pre_axis_plot()
        if self.subplot_kw != None:
           self.axis = pylab.subplot(gs,**self.subplot_kw)
        else:
           self.axis = pylab.subplot(gs)
        self.pre_plot()
        self.plot()
        self.post_plot()
        return self.axis

    def update_params(self,params):
        """
        Updates the modifiable parameters and sets them to be further un-modifiable
        """
        for key in params:
            if not key in self.__dict__['parameters']:
                raise AttributeError("Error, unknown parameter supplied %s, known parameters: %s" % (key,
                                                                                                     self.__dict__['parameters'].keys()))
        self.__dict__['parameters'].update(params)
        self.modified = {}
        for k in self.__dict__['parameters']:
            self.modified[k] = False
        for k in params:
            self.modified[k] = True



class StandardStyle(SimplePlot):

    def __init__(self,**param):
        """
        This is the standard SimplePlot used in this package. In most cases we recomand users to derive
        all their plots from this class. In case users want to define their own plot styling they will 
        want to create an analogous class as this one and then derive all their plots from it.
        
        The modifiable parameters that this SimplePlot defines
        
        Other parameters
        ----------------
        
        fontsize : int
                 Font size to be used for tick labels and axis labels.
                 
        ?_tick_style : str
               The style of ticks to be plotted. Note that the style interacts with the x_ticks and y_ticks commands in that it formats the ticks set by these
               variables, or by the default ticks set by matplotlib. Currently available styles are: 
                            * Min - plots three ticks, 2 on sides one in the middle (if even number of xticks supplied only the side ones will be plotted)
                            * Custom - will plot tikcs as defined by x/yticks arguments
                            
        x_axis  : bool 
                Whether to plot the x_axis (and  the ticks).
                
        y_axis  : bool
                Whether to plot the y_axis (and  the ticks).
                
        x_label : str
                What is the xlabel (None means no label will be plotted).
                
        y_label : str
                What is the ylabel (None means no label will be plotted).
                
        top_right_border : bool
                         Whether to plot the top and right border of the axis.
        
        left_border : bool
                    Whether to plot the left border of the axis.
                    
        bottom_border  : bool
                       Whether to plot the right border of the axis.
                       
        title : str
              What is the title (None means no label will be plotted)
                       
        x_scale : str
              What is the scaling of the x-axis ('linear' | 'log' | 'symlog'), default is 'linear'
              
        x_scale_base : int
              What is the x base of the logarithm. Active only if x_scale != to 'linear'
               
        y_scale : str
              What is the scaling of the y-axis ('linear' | 'log' | 'symlog'), default is 'linear'
               
        y_scale_base : int
              What is the x base of the logarithm. Active only if x_scale != to 'linear'
             
        x_lim  : tuple
               What are the xlims (None means matplotlib will infer from data).
               
        y_lim : tuple
              What are the ylims (None means matplotlib will infer from data).
        
        x_ticks : list
                What are the xticks (note that the tick style, and x_axis can override/modify these).
                
        y_ticks : list
                What are the yticks (note that the tick style, and y_axis can override/modify these).
                
        x_tick_labels : list
                      What are the x tick lables (note that the tick style, and x_axis can override/modify these, and x_tick_labels have to match x_ticks)
                      
        y_tick_labels : list
                      What are the y tick lables (note that the tick style, and y_axis can override/modify these, and y_tick_labels have to match y_ticks)
        
        x_tick_pad  : float
                    What are the x tick padding of labels for axis.
                    
        y_tick_pad  : float
                    What are the y tick padding of labels for axis.
                    
        grid : bool
             Do we show grid?
        """

        SimplePlot.__init__(self,**param)
        fontsize = 15
        self.parameters = {
            "fontsize": fontsize,
            "x_tick_style": 'Min',
            "y_tick_style": 'Min',
            "x_axis": True,
            "y_axis": True,
            "x_label": None,
            "y_label": None,
            "top_right_border": False,
            "left_border": True,
            "bottom_border": True,
            "title": None,
            "x_scale": 'linear',
            "x_scale_base": None,
            "y_scale": 'linear',
            "y_scale_base": None,
            "x_lim": None,
            "y_lim": None,
            "x_ticks": None,
            "y_ticks": None,
            "x_tick_labels": None,
            "y_tick_labels": None,
            "x_tick_pad": fontsize - 5,
            "y_tick_pad": fontsize - 5,
            "grid" : False,
        }
        
        self.color_cycle = {
                            'Bl':(0,0,0),
                            'Or':(.9,.6,0),
                            'SB':(.35,.7,.9),
                            'bG':(0,.6,.5),
                            'Ye':(.95,.9,.25),
                            'Bu':(0,.45,.7),
                            'Ve':(.8,.4,0),
                            'rP':(.8,.6,.7),
                        }
        import matplotlib.colors
        
        self.colormap = matplotlib.colors.LinearSegmentedColormap.from_list('CMcmSBVe',[self.color_cycle['SB'],self.color_cycle['Ve']])

    def pre_axis_plot(self):
        pylab.rc('axes', linewidth=3)
        self.xtick_pad_backup = pylab.rcParams['xtick.major.pad']
        pylab.rcParams['xtick.major.pad'] = self.x_tick_pad
        self.ytick_pad_backup = pylab.rcParams['ytick.major.pad']
        pylab.rcParams['ytick.major.pad'] = self.y_tick_pad
        self.colormap_backup = pylab.rcParams['axes.color_cycle']
        pylab.rcParams['axes.color_cycle'] = [self.color_cycle[c] for c in sorted(self.color_cycle.keys())]


    def pre_plot(self):
        pass

    def post_plot(self):
       
        if self.title != None:
            pylab.title(self.title, fontsize=self.fontsize)
        
        
        if self.x_scale:
            pylab.xscale(self.x_scale)
            if self.x_scale_base:
                pylab.xscale(self.x_scale, basex=self.x_scale_base)
        
        if self.y_scale:
            pylab.yscale(self.y_scale)
            if self.y_scale_base:
                pylab.yscale(self.y_scale, basey=self.y_scale_base)

        if not self.x_axis:
            phf.disable_xticks(self.axis)
            phf.remove_x_tick_labels()
        if not self.y_axis:
            phf.disable_yticks(self.axis)
            phf.remove_y_tick_labels()

        if self.x_lim:
            pylab.xlim(self.x_lim)
        if self.y_lim:
            pylab.ylim(self.y_lim)

        self._ticks()


        if self.y_label and self.y_axis:
            pylab.ylabel(self.y_label,multialignment='center',fontsize=self.fontsize)
        if self.x_label and self.x_axis:
            pylab.xlabel(self.x_label,multialignment='center',fontsize=self.fontsize)
        if not self.top_right_border:
            phf.disable_top_right_axis(self.axis)
        if not self.left_border:
            phf.disable_left_axis(self.axis)
        if not self.bottom_border:
            phf.disable_bottom_axis(self.axis)

        if self.grid:
            self.axis.grid(True)

        
        pylab.rc('axes', linewidth=1)
        pylab.rcParams['xtick.major.pad'] = self.xtick_pad_backup
        pylab.rcParams['ytick.major.pad'] = self.ytick_pad_backup
        pylab.rcParams['axes.color_cycle'] =self.colormap_backup

    def _ticks(self):
        if self.x_axis:
            if self.x_ticks != None and self.x_tick_style == 'Custom':
                if self.x_tick_labels != None:
                    assert len(self.x_ticks) == len(self.x_tick_labels)
                    pylab.xticks(self.x_ticks, self.x_tick_labels)
                else:
                    pylab.xticks(self.x_ticks)
                    phf.remove_x_tick_labels()
            elif self.x_ticks != None:    
                 pylab.xticks(self.x_ticks)
                 phf.short_tick_labels_axis(self.axis.xaxis)
            else:
                if self.x_tick_style == 'Min':
                    phf.three_tick_axis(self.axis.xaxis,log=(self.x_scale!='linear'))
                elif self.x_tick_style == 'Custom':                
                   phf.disable_xticks(self.axis)
                   phf.remove_x_tick_labels()
                else:
                    raise ValueError('Unknown x tick style %s', self.x_tick_style)

        if self.y_axis:
            if self.y_ticks != None and self.x_tick_style == 'Custom':
                if self.y_tick_labels != None:
                    assert len(self.y_ticks) == len(self.y_tick_labels)
                    pylab.yticks(self.y_ticks, self.y_tick_labels)
                else:
                    pylab.yticks(self.y_ticks)
                    phf.remove_y_tick_labels()
            elif self.y_ticks != None:    
                 pylab.yticks(self.y_ticks)
                 phf.short_tick_labels_axis(self.axis.yaxis)
            else:
                if self.y_tick_style == 'Min':
                    phf.three_tick_axis(self.axis.yaxis,log=(self.y_scale!='linear'))
                elif self.y_tick_style == 'Custom':                
                   phf.disable_yticks(self.axis)
                   phf.remove_y_tick_labels()
                else:
                    raise ValueError('Unknow y tick style %s', self.y_tick_style)

        for label in self.axis.get_xticklabels() + self.axis.get_yticklabels():
            label.set_fontsize(self.fontsize)


class SpikeRasterPlot(StandardStyle):
    """
    This function plots the raster plot of spikes in the `spike_lists` argument.

    Parameters
    ----------
    spike_lists : list
                The `spike_lists` argument is a list of list of SpikeList objects. The top
                level list corresponds to different sets of spikes that should be plotted
                over each other. They have to contain the same number of neurons. Each set
                of spikes will be colored by the color on corresponding postion of the
                colors parameter (matplotlib readable color formats accapted). If None all
                colors will be set to '#848484' (gray). The second level list contains 
                different trials of the populations responses stored in the individual SpikeList objects.

    Other parameters
    ----------------
    group_trials : bool
                 If is set to true - trials will be concatenated and plotted on the same line.
    
    colors : list
           The colors to assign to the different sets of spikes. 

    Notes
    -----
    
    Each trial will be plotted as a sepparate line of spikes, and these will be
    grouped by the neurons. Only neurons in the neurons parameter will be
    plotted. If neurons are None, (up to) first 10 neurons will be plotted.
    """
    def __init__(self, spike_lists,**param):
        StandardStyle.__init__(self,**param)
        self.sps = spike_lists
        self.parameters["colors"] = None
        self.parameters["group_trials"] = False

    def plot(self):
        if self.parameters["colors"] == None:
            colors = ['#000000' for i in xrange(0, len(self.sps))]
        else:
            colors = self.colors

        neurons = [i for i in xrange(0, len(self.sps[0][0]))]
        
        t_start = float(self.sps[0][0][0].t_start)
        t_stop = float(self.sps[0][0][0].t_stop)
        
        
        num_n = len(neurons)  # number of neurons
        num_t = len(self.sps[0])  # number of trials

        for k, sp in enumerate(self.sps):
            for j, n in enumerate(neurons):
                if self.group_trials:
                   train = []
                   for i, spike_list in enumerate(sp):
                       train.extend(spike_list[n])
                   self.axis.plot(train,[j for x in xrange(0, len(train))],'|',color=colors[k])
                else:
                    for i, spike_list in enumerate(sp):
                        spike_train = spike_list[n]
                        self.axis.plot(spike_train,
                                       [j * (num_t + 1) + i + 1
                                          for x in xrange(0, len(spike_train))],
                                       '|',
                                       color=colors[k])
            if not self.group_trials:
                for j in xrange(0, num_n - 1):
                    self.axis.axhline(j * (num_t + 1) + num_t + 1, c='k')
                    
        if not self.group_trials:
            self.y_lim = (0, num_n * (num_t + 1))
        else:
            self.y_lim = (0, num_n)
        
        self.x_ticks = [t_start, (t_stop-t_start)/2, t_stop]
        self.x_tick_style = 'Custom'
        self.x_lim = (t_start, t_stop)
        self.x_label = 'time (ms)'
        if not self.group_trials:
            if num_n == 1:
                self.y_label = 'Trial'
            else:
                self.y_label = 'Neuron/Trial'
        else:
            self.y_label = 'Neuron' 
        self.y_ticks = []

class SpikeHistogramPlot(SpikeRasterPlot):
    """
    This function plots the raster plot of spikes in the spike_list argument.
    
    Parameters
    ----------
    spike_list : list
               List of list of SpikeList objects.
               The top level list corresponds to different sets of spikes that should be
               plotted over each other. They have to contain the same number of neurons.
               Each set of spikes will be colored by the color on corresponding postion of
               the colors parameter. If None all colors will be set to '#848484' (gray).
               The second level list contains different trials of the populations responses
               stored in the individual SpikeList objects.

    
    Other parameters
    ----------------
    bin_width : bool
                 The with of the bins into which to bin the spikes.
    
    colors : list
           The colors to assign to the different sets of spikes. 


    Notes
    -----
    
    Each trial will be plotted as a sepparate line of spikes, and these will be
    grouped by the neurons. Only neurons in the neurons parameter will be
    plotted. If neurons are None, the first neuron will be plotted. 
    """

    def __init__(self, spike_lists,**param):
        SpikeRasterPlot.__init__(self, spike_lists,**param)
        self.parameters["bin_width"] = 5.0
        self.parameters["colors"] = ['#000000' for i in xrange(0, len(self.sps))]
    def plot(self):
        self.neurons = [i for i in xrange(0, min(10, len(self.sps[0][0])))]

        t_stop = float(self.sps[0][0][0].t_stop)
        t_start = float(self.sps[0][0][0].t_start)

        all_spikes = []
        for k, sp in enumerate(self.sps):
            tmp = []
            for i, spike_list in enumerate(sp):
                for j in self.neurons:
                    spike_train = spike_list[j]
                    tmp.extend(spike_train.magnitude)
            all_spikes.append(tmp)

        if all_spikes != []:
            self.axis.hist(all_spikes,
                           bins=numpy.arange(0, t_stop, self.bin_width),
                           color=self.colors,
                           edgecolor='none')

        self.y_label = '(spk/ms)'
        self.x_tick_style = 'Custom'
        self.x_ticks = [t_start, (t_stop-t_start)/2, t_stop]
        self.x_lim = (t_start, t_stop)
        self.x_label = 'time (ms)'




class StandardStyleAnimatedPlot(StandardStyle):
    """
    This is an abstract class helping construction of animated graphs.

    Each class subclassing from this class should implement the `SimplePlot`
    `plot()` function in which it should draw the first frame of the animation
    with all the corresponding decorations.

    Next it needs to implement the `plot_next_frame` function which should
    replot the data in the plot corresponding to the next frame. This function
    needs to keep track of the frame number itself if it needs it. Note that
    advanced usage of matplotlib is recomanded here so that not the whole graph
    is always replotted but only new data are set to the graph.

    The `StandardStyleAnimatedPlot` will take care of the updating of the
    figure, so now `draw()` or `show()` or event commands should be issued.

    Parameters
    ----------
    
    frame_duration : double
                   Duration of single frame. 
    """

    def plot_next_frame(self):
        """
        The function that each instance of `StandardStyleAnimatedPlot` has to implement, in which it updated
        the data in the plot to the next frame.
        """
        raise NotImplementedError

    @staticmethod  # hack to make it compatible with FuncAnimation - we have to make it static
    def _plot_next_frame(self):
        a = self.plot_next_frame()
        return a,

    def post_plot(self):
        StandardStyle.post_plot(self)
        self.plotting_parent.register_animation_update_function(StandardStyleAnimatedPlot._plot_next_frame,self)

class PixelMovie(StandardStyleAnimatedPlot):
    """
    An instatiation of StandardStyleAnimatedPlot that works with regularly sampled data - i.e. 3D matricies
    where axis are interpreted as x,y,t.
    
    Parameters
    ----------
    
    movie : ndarray
          3D array with axis (t,x,y) holding the data.
          
    """
    
    def __init__(self, movie,background_luminance,**param):
        StandardStyleAnimatedPlot.__init__(self,**param)
        self.background_luminance = background_luminance
        self.movie = movie
        self.l = len(movie)
        self.i = 0
        self.parameters["left_border"] = False
        self.parameters["bottom_border"] = False

    def plot_next_frame(self):
        self.im.set_array(self.movie[self.i])
        self.i = self.i + 1
        if self.i == self.l:
            self.i = 0
        return self.im

    def plot(self):
        self.im = self.axis.imshow(self.movie[0],interpolation='nearest',vmin=0,vmax=self.background_luminance*2,cmap='gray')


class ScatterPlotMovie(StandardStyleAnimatedPlot):
    """
    An instatiation of StandardStyleAnimatedPlot that works with irregularly sampled data. That data
    are assumed to have constant positions throught the time.
    
    Parameters
    ----------
    
    x : ndarray
        1D array containing the x coordinate of the values to be displayed.

    y : ndarray
        1D array containing the y coordinate of the values to be displayed.
    
    z : ndarray
        2D array containing the values to be displayed (t,values)
    """
    
    def __init__(self, x, y, z,**param):
        StandardStyleAnimatedPlot.__init__(self,**param)
        self.z = z
        self.x = x
        self.y = y
        self.l = len(z)
        self.i = 0
        self.parameters["dot_size"] = 20
        self.parameters["marker"] = 'o'
        self.parameters["left_border"] = False
        self.parameters["bottom_border"] = False

    def plot_next_frame(self):
        #vmax = numpy.max(self.z)/2.0
        #d= numpy.array([[1.0,1.0,1.0,x/vmax] for x in self.z[self.i, :].flatten()])
        self.scatter.set_array(self.z[self.i, :].flatten())
        self.i = self.i + 1
        if self.i == self.l:
            self.i = 0
        print self.i
        return self.scatter

    def plot(self):
        vmax = numpy.max(self.z)/4.0
        #d= numpy.array([[1,1,1,x/vmax] for x in self.z[0, :].flatten()])
        #print numpy.shape(d)
        self.scatter = self.axis.scatter(self.x.flatten(), self.y.flatten(),
                                         c=self.z[0, :].flatten(),
                                         s=self.parameters["dot_size"],
                                         marker=self.parameters["marker"],
                                         lw=0,
                                         vmax=vmax,
                                         alpha=0.4,
                                         cmap='gray')
        pylab.axis('equal')
        pylab.gca().set_axis_bgcolor('black')

class ScatterPlot(StandardStyle):
    """
    Simple scatter plot

    Parameters
    ----------

    z : ndarray
      Array of the values to be plotted.
    
    x : ndarray
      Array with the y positions of the neurons.
      
    y : ndarray
      Array with the y positions of the neurons.
      
      
    periodic : bool
             If the z is a periodic value.
             
    period : double
           If periodic is True the z should come from (0,period).
    
    
    Other parameters
    ----------------
    
    identity_line : bool
                  Should identity line be show?  
      
    colorbar : bool
             Should there be a colorbar?
             
    cmp : colormap
             The colormap to use.
    
    dot_size : double
             The size of the markers in the scatter plot.
             
    marker : str
           The type of marker to use in the scatter plot.
    
    equal_aspect_ratio : bool
                       If to enforce equal aspect ratio. 
        
    mark_means : bool
               Whether to mark the means of each axis.

    colorbar_label : label
               The label  that will be put on the colorbar.
    """

    def __init__(self, x, y, z='b', periodic=False, period=None,**param):
        StandardStyle.__init__(self,**param)
        self.z = z
        self.x = x
        self.y = y
        self.periodic = periodic
        self.period = period
        if self.periodic:
            self.parameters["cmp"] = 'hsv'
        else:
            self.parameters["cmp"] = self.colormap
        self.parameters["dot_size"] = 20
        self.parameters["marker"] = 'o'
        self.parameters["equal_aspect_ratio"] = False
        self.parameters["top_right_border"]=True
        self.parameters["colorbar"] = False
        self.parameters["mark_means"] = False
        self.parameters["identity_line"] = False
        self.parameters["colorbar_label"] = None

    def plot(self):
        if not self.periodic and self.z !='b':
            vmax = numpy.max(self.z)
            vmin = numpy.min(self.z)
        else:
            vmax = self.period
            vmin = 0
        ax = self.axis.scatter(self.x, self.y, c=self.z,
                               s=self.dot_size,
                               marker=self.marker,
                               lw=0,
                               cmap=self.cmp,
                               color='k',
                               vmin=vmin,
                               vmax=vmax)
        if self.equal_aspect_ratio:
            self.axis.set_aspect(aspect=1.0, adjustable='box')
        self.x_lim = (numpy.min(self.x),numpy.max(self.x))
        self.y_lim = (numpy.min(self.y),numpy.max(self.y))

        #self.x_ticks = [1.1*numpy.min(self.x), 1.1*numpy.max(self.x)]
        #self.y_ticks = [1.1*numpy.min(self.y), 1.1*numpy.max(self.y)]

        if self.identity_line:
           pylab.plot([-1e10,1e10],[-1e10,1e10],'k',linewidth=2) 

        if self.mark_means:
           self.axis.plot([numpy.mean(self.x),numpy.mean(self.x)],[-1e10,1e10],'r--')
           self.axis.plot([-1e10,1e10],[numpy.mean(self.y),numpy.mean(self.y)],'r--')
           
           
        if self.colorbar:
            cb = pylab.colorbar(ax, ticks=[vmin, vmax], use_gridspec=True)
            cb.set_label(self.colorbar_label)
            cb.set_ticklabels(["%.2g" % vmin, "%.2g" % vmax])


class StandardStyleLinePlot(StandardStyle):
    """
    This function plots vector data in simple line plots.
    
    Parameters
    ----------
    x, y : list
         lists each containing corresponding vectors of x and y axis values to
         be plotted.

    labels : list, optional
           Can contain the labels to be given to the individual line plots.

    
    Other parameters
    ----------------
    
    colors : int or list or dict
           The colors of the plots. If it is one color all plots will have that same color. If it is a list its
           length should correspond to length of x and y and the corresponding colors will be assigned to the individual graphs.
           If dict, the keys should be labels, and values will be the colors that will be assigned to the lines that correspond to the labels.

    mean : bool
         If the mean of the vectors should be plotted as well.
         
    fill : bool
         If true the graphs bellow the lines will be filled, with the same color as the one assigned to the line but with 50% transparency
    
    legend : bool
           If true the legend will be shown

    """

    def __init__(self, x, y, labels=None,**param):
        StandardStyle.__init__(self,**param)
        self.x = x
        self.y = y
        self.parameters["labels"] = labels
        self.parameters["colors"] = None
        self.parameters["mean"] = False
        self.parameters["fill"] = False
        self.parameters["legend"] = False
        self.parameters["linewidth"] = 1

        assert len(x) == len(y)
        if labels != None:
            assert len(x) == len(labels)
        
        if self.mean:
            for i in xrange(0, len(x)):
                if not numpy.all(x[i] == x[0]):
                    raise ValueError("Mean cannot be calculated from data not containing identical x axis values")

    def plot(self):
        
        if type(self.colors) == dict:
           assert self.labels != None
           assert len(self.colors.keys()) == len(self.labels)
        
        tmin = 10**10
        tmax = -10**10
        for i in xrange(0, len(self.x)):
            if self.mean:
                if i == 0:
                    m = self.y[i]
                else:
                    m = m + self.y[i]

            if self.colors == None:
                color = 'k'
            elif type(self.colors) == list:
                color = self.colors[i]
            elif type(self.colors) == dict:
                assert self.labels[i] in self.colors.keys(), "Cannot find curve named %s" % (self.labels[i])
                color = self.colors[self.labels[i]]
            else:
                color = self.colors

            if self.labels != None:
                self.axis.plot(self.x[i], self.y[i],
                               linewidth=self.linewidth,
                               label=self.labels[i],
                               color=color)
            else:
                self.axis.plot(self.x[i], self.y[i],
                               linewidth=self.linewidth,
                               color=color)
            
            if self.fill:
               d = numpy.zeros(len(self.y[i]))
               self.axis.fill_between(self.x[i],self.y[i],where=self.y[i]>=d, color=color, alpha=0.2,linewidth=0)
               self.axis.fill_between(self.x[i],self.y[i],where=self.y[i]<=d, color=color, alpha=0.2,linewidth=0)
                
            pylab.hold('on')

            tmin = min(tmin, self.x[i][0])
            tmax = max(tmax, self.x[i][-1])

        if self.mean:
            m = m / len(self.x)
            self.axis.plot(self.x[0], m, color='k', linewidth=2*self.linewidth)

        if self.legend:
            self.axis.legend()
        self.x_lim = (tmin, tmax)

class ConductancesPlot(StandardStyle):
    """
    Plots conductances.

    Parameters
    ----------
    
    exc : list
         List of excitatory conductances (AnalogSignal type).
         
    inh : list
        List of inhibitory conductances (AnalogSignal type).

    
    Other parameters
    ----------------

    legend : bool
           Whether legend should be displayed.
    """

    def __init__(self, exc, inh,**param):
        StandardStyle.__init__(self,**param)
        self.gsyn_es = exc
        self.gsyn_is = inh
        self.parameters["legend"] = False

    def plot(self):
        mean_gsyn_e = numpy.zeros(numpy.shape(self.gsyn_es[0]))
        mean_gsyn_i = numpy.zeros(numpy.shape(self.gsyn_is[0]))
        sampling_period = self.gsyn_es[0].sampling_period
        t_stop = float(self.gsyn_es[0].t_stop - sampling_period)
        t_start = float(self.gsyn_es[0].t_start)
        time_axis = numpy.arange(0, len(self.gsyn_es[0]), 1) / float(len(self.gsyn_es[0])) * abs(t_start-t_stop) + t_start
    
        for e, i in zip(self.gsyn_es, self.gsyn_is):
            e = e.rescale(mozaik.tools.units.nS) #e=e*1000
            i = i.rescale(mozaik.tools.units.nS) #i=i*1000
            self.axis.plot(time_axis, e.tolist(), color='#F5A9A9')
            self.axis.plot(time_axis, i.tolist(), color='#A9BCF5')
            mean_gsyn_e = mean_gsyn_e + numpy.array(e.tolist())
            mean_gsyn_i = mean_gsyn_i + numpy.array(i.tolist())

        mean_gsyn_i = mean_gsyn_i / len(self.gsyn_is)
        mean_gsyn_e = mean_gsyn_e / len(self.gsyn_es)

        p1, = self.axis.plot(time_axis, mean_gsyn_e.tolist(), color='r', linewidth=1)
        p2, = self.axis.plot(time_axis, mean_gsyn_i.tolist(), color='b', linewidth=1)

        if self.legend:
            self.axis.legend([p1, p2], ['exc', 'inh'])

        self.x_lim = (t_start, t_stop)
        #self.x_ticks = [t_start, (t_stop - t_start)/2, t_stop]
        self.x_label = 'time(' + self.gsyn_es[0].t_start.dimensionality.latex + ')'
        self.y_label = 'g(' + mozaik.tools.units.nS.dimensionality.latex + ')'


class ConnectionPlot(StandardStyle):
    """
    A plot that will display connections.

    Parameters
    ----------

    pos_x : ndarray
            Array with x position of the target neurons.
    
    pos_y : ndarray
            Array with y position of the target neurons.

    source_x : double
             x position of the source neuron.

    source_y : double
             y position of the source neuron.

    weights : list
            Array of weights from source neuron to target neurons.


    colors : list, optional
           The extra information per neuron that will be displayed as colors. None if no info.

    period : float, optional
           If period is not None the colors should come from (0,period)


    Other parameters
    ----------------
    
    line : bool
         (TODO) True means the existing connections will be displayed as lines, with their thickness indicating the strength.
         False means the strength of connection will be indicated by the size of the circle representing the corresponding target neuron. 

    colorbar : bool
             Should there be a colorbar ?

    cmp : colormap
             The colormap to use.
    
    colorbar_label : label
               The label  that will be put on the colorbar.             
    """

    def __init__(self, pos_x, pos_y, source_x, source_y, weights,colors=None,period=None,**param):
        StandardStyle.__init__(self,**param)
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.source_x = source_x
        self.source_y = source_y
        self.weights = weights
        self.colors = colors
        
        self.period = period

        if self.period != None:
            self.parameters["cmp"] = 'hsv'
        else:
            self.parameters["cmp"] = self.colormap
        
        self.parameters["top_right_border"] = True
        self.parameters["colorbar"] = False
        self.parameters["colorbar_label"] = None
        self.parameters["line"] = False

    def plot(self):
        self.x_lim = [numpy.min(self.pos_x),numpy.max(self.pos_x)]
        self.y_lim = [numpy.min(self.pos_y),numpy.max(self.pos_y)]
        if len(numpy.nonzero(self.weights)[0]) == 0:
            return 
        
        
        self.pos_x = self.pos_x[numpy.nonzero(self.weights)[0]]
        self.pos_y = self.pos_y[numpy.nonzero(self.weights)[0]]

        
        if self.colors != None:
            self.colors = numpy.array(self.colors)
            self.colors = self.colors[numpy.nonzero(self.weights)[0]] 
        self.weights = self.weights[numpy.nonzero(self.weights)[0]]

        if self.colors == None:
            if numpy.max(self.weights) > 0:
                s = self.weights / numpy.max(self.weights) * 200
            else:
                s = 0
            ax = self.axis.scatter(self.pos_x, self.pos_y, c='black', s=s, lw=0)
        else:
            if self.period == None:
                vmax = numpy.max(self.colors)
                vmin = numpy.min(self.colors)
            else:
                vmax = self.period
                vmin = 0
                
            ax = self.axis.scatter(self.pos_x, self.pos_y, c=numpy.array(self.colors),edgecolors=None,
                                   s=self.weights/numpy.max(self.weights)*100,
                                   lw=1, cmap=self.cmp,
                                   vmin=vmin, vmax=vmax)
            if self.colorbar:
                cb = pylab.colorbar(ax, ticks=[vmin, vmax], use_gridspec=True)
                cb.set_label(self.colorbar_label)
                cb.set_ticklabels(["%.3g" % vmin, "%.3g" % vmax])

        self.axis.set_aspect(aspect=1.0, adjustable='box')

        self.x_label = 'x'
        self.y_label = 'y'


class HistogramPlot(StandardStyle):
    """
    This function plots the histogram of list of value lists, coloring each independently.
    
    Parameters
    ----------
    values : list
               List of numpy arrays objects.
               The top level list corresponds to different sets of values that will be 
               plotted together.
               
               Each set will be colored by the color on corresponding postion of
               the colors parameter. If None all colors will be set to '#848484' (gray).
    
    Other parameters
    ----------------
    num_bins : int
                 The with of the bins into which to bin the spikes.
    
    colors : list
           The colors to assign to the different sets of spikes. 
    """

    def __init__(self, values,**param):
        StandardStyle.__init__(self,**param)
        self.values = values
        self.parameters["num_bins"] = 15.0
        self.parameters["log"] = False

    def plot(self):
        
        if self.parameters["log"]:
            mmin  = numpy.floor(numpy.log10(numpy.min(numpy.array(self.values)[numpy.nonzero(self.values)])))
            bins = numpy.logspace(mmin,numpy.ceil(numpy.log10(numpy.max(self.values))),self.num_bins)
            
            if numpy.min(self.values) == 0.0:
               bins = numpy.concatenate([[0.0],bins])
        else:
            bins = self.num_bins   
            
        self.axis.hist(self.values,bins=bins,range=self.x_lim,edgecolor='none')
        self.y_label = '#'
        
