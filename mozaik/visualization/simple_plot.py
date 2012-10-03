"""
See visualization.plotting for documentation.
"""

import mozaik.visualization.plotting_helper_functions as phf
import pylab
import numpy
import mozaik

logger = mozaik.getMozaikLogger("Mozaik")


class SimplePlot(object):
    """
    One general policy is that any overwriting of the standard style default
    values by the `Plotting` mechanisms should take precedence over those done
    by specific instances of `SimplePlot`. In order to enforce this precedence,
    the modifiable parameters of the classes should be stored in the common
    `dictionary` parameters. Each class derived from `SimplePlot` should add its
    modifiable parameters into the `parameters` dictionary in its constructor.

    The kwargs passed to the instances of `SimplePlot` from the `Plotting`
    mechanisms will be stored. At the beginning of the __call__ the dictionary
    will be updated with these stored kwargs, and the updated parameters (note
    not all have to be present) will be marked. These marked parameters will not
    be then modifiable any further. In order to do so, the `parameters`
    dictionary is accessible via the __getattr__ and __setattr__ functions.

    *Note, for this reason all `SimplePlot` classes need to take care that none
    of the modifiable attributes is also defined as a class attribute.*
    """

    def pre_plot(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError

    def post_plot(self):
        raise NotImplementedError

    def pre_axis_plot(self):
        raise NotImplementedError

    def __init__(self, **kwargs):
        self.parameters = {}  # the common modifiable parameter dictionary
        self.to_be_modified_parameters = kwargs

    def __getattr__(self, name):
        if name == 'parameters':
            return self.__dict__[name]
        if name in self.__dict__['parameters'] and name in self.__dict__:
            raise AttributeError("Error, attribute both in __dict__ and self.parameters")
        elif name in self.__dict__['parameters']:
            return self.__dict__['parameters'][name]
        elif name in self.__dict__:
            return self.__dict__[name]
        else:
            raise AttributeError

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

    def __call__(self, gs):
        """
        Calls all the plotting styling and execution functions in the right
        order
        """
        self.update_params()
        self.pre_axis_plot()
        self.axis = pylab.subplot(gs)
        self.pre_plot()
        self.plot()
        self.post_plot()

    def update_params(self):
        """
        Updates the modifiable parameters and sets them to be further
        un-modifiable
        """
        for key in self.to_be_modified_parameters:
            if not key in self.__dict__['parameters']:
                raise AttributeError("Error, unknown parameter supplied %s, known parameters: %s" % (key,
                                                                                                     self.__dict__['parameters'].keys()))
        self.__dict__['parameters'].update(self.to_be_modified_parameters)
        self.modified = {}
        for k in self.__dict__['parameters']:
            self.modified[k] = False
        for k in self.to_be_modified_parameters:
            self.modified[k] = True


class StandardStyle(SimplePlot):

    def __init__(self, **kwargs):
        """
        fontsize            Font size to be used for tick labels and axis labels
        ?_tick_style        The style of ticks to be plotted. Note that the style interacts with the x_ticks and y_ticks commands in that it formats the ticks set by these
                            variables, or by the default ticks set by matplotlib
                            Available styles are:
                               Min - plots three ticks, 2 on sides one in the middle (if even number of xticks supplied only the side ones will be plotted)
                               Custom - will plot tikcs as defined by x/yticks arguments
        x_axis              whether to plot the x_axis (and  the ticks)
        y_axis              whether to plot the y_axis (and  the ticks)
        x_label             what is the xlabel (None means no label will be plotted)
        y_label             what is the ylabel (None means no label will be plotted)
        top_right_border    Whether to plot the top and right border of the axis
        left_border         Whether to plot the left border of the axis
        bottom_border       Whether to plot the right border of the axis
        title               what is the title (None means no label will be plotted)
        x_lim               what are the xlims (None means matplotlib will infer from data)
        y_lim               what are the ylims (None means matplotlib will infer from data)
        x_ticks             what are the xticks (note that the tick style, and x_axis can override/modify these)
        y_ticks             what are the yticks (note that the tick style, and y_axis can override/modify these)
        x_tick_labels       what are the x tick lables (note that the tick style, and x_axis can override/modify these, and x_tick_labels have to match x_ticks)
        y_tick_labels       what are the y tick lables (note that the tick style, and y_axis can override/modify these, and y_tick_labels have to match y_ticks)
        x_tick_pad          what are the x tick padding of labels for axis
        y_tick_pad          what are the y tick padding of labels for axis
        """

        SimplePlot.__init__(self, **kwargs)
        fontsize = 12
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
            "x_lim": None,
            "y_lim": None,
            "x_ticks": None,
            "y_ticks": None,
            "x_tick_labels": None,
            "y_tick_labels": None,
            "x_tick_pad": fontsize - 5,
            "y_tick_pad": fontsize - 5,
        }

    def pre_axis_plot(self):
        pylab.rc('axes', linewidth=3)
        self.xtick_pad_backup = pylab.rcParams['xtick.major.pad']
        pylab.rcParams['xtick.major.pad'] = self.x_tick_pad
        self.ytick_pad_backup = pylab.rcParams['ytick.major.pad']
        pylab.rcParams['ytick.major.pad'] = self.y_tick_pad

    def pre_plot(self):
        pass

    def post_plot(self):
        if self.title != None:
            pylab.title(self.title, fontsize='x-small')

        if self.x_lim:
            pylab.xlim(self.x_lim)
        if self.y_lim:
            pylab.ylim(self.y_lim)
        if not self.x_axis:
            phf.disable_xticks(self.axis)
            phf.remove_x_tick_labels()
        if not self.y_axis:
            phf.disable_yticks(self.axis)
            phf.remove_y_tick_labels()

        self.ticks()

        if self.y_label:
            pylab.ylabel(self.y_label)
        if self.x_label:
            pylab.xlabel(self.x_label)
        if not self.top_right_border:
            phf.disable_top_right_axis(self.axis)
        if not self.left_border:
            phf.disable_left_axis(self.axis)
        if not self.bottom_border:
            phf.disable_bottom_axis(self.axis)

        pylab.rc('axes', linewidth=1)
        pylab.rcParams['xtick.major.pad'] = self.xtick_pad_backup
        pylab.rcParams['ytick.major.pad'] = self.ytick_pad_backup

    def ticks(self):

        if self.x_ticks != None:
            if self.x_tick_labels != None and (len(self.x_ticks) == len(self.x_tick_labels)):
                pylab.xticks(self.x_ticks, self.x_tick_labels)
            else:
                pylab.xticks(self.x_ticks)

        if self.y_ticks != None:
            if self.y_tick_labels != None and (len(self.y_ticks) == len(self.y_tick_labels)):
                pylab.yticks(self.y_ticks, self.y_tick_labels)
            else:
                pylab.yticks(self.y_ticks)

        if self.x_tick_style == 'Min':
            phf.three_tick_axis(self.axis.xaxis)
        elif self.x_tick_style == 'Custom':
            pass
        else:
            raise ValueError('Unknown x tick style %s', self.x_tick_style)

        if self.y_tick_style == 'Min':
            phf.three_tick_axis(self.axis.yaxis)
        elif self.y_tick_style == 'Custom':
            pass
        else:
            raise ValueError('Unknow y tick style %s', self.y_tick_style)

        for label in self.axis.get_xticklabels() + self.axis.get_yticklabels():
            label.set_fontsize(self.fontsize)


class SpikeRasterPlot(StandardStyle):
    """
    This function plots the raster plot of spikes in the `spike_lists` argument.

    The `spike_lists` argument is a list of list of SpikeList objects. The top
    level list corresponds to different sets of spikes that should be plotted
    over each other. They have to contain the same number of neurons. Each set
    of spikes will be colored by the color on corresponding postion of the
    colors parameter (matplotlib readable color formats accapted). If None all
    colors will be set to '#848484' (gray).

    The second level list contains different trials of the populations responses
    stored in the individual SpikeList objects.

    Each trial will be plotted as a sepparate line of spikes, and these will be
    grouped by the neurons. Only neurons in the neurons parameter will be
    plotted. If neurons are None, (up to) first 10 neurons will be plotted.
    """

    def __init__(self, spike_lists, **kwargs):
        StandardStyle.__init__(self, **kwargs)
        self.sps = spike_lists
        self.parameters["colors"] = None
        self.parameters["neurons"] = None

    def plot(self):
        if self.parameters["colors"] == None:
            colors = ['#000000' for i in xrange(0, len(self.sps))]
        else:
            colors = self.colors

        if self.parameters["neurons"] == None:
            neurons = [i for i in xrange(0, min(10, len(self.sps[0][0])))]
        else:
            neurons = self.neurons

        t_stop = float(self.sps[0][0][0].t_stop)
        num_n = len(neurons)  # number of neurons
        num_t = len(self.sps[0])  # number of trials

        for k, sp in enumerate(self.sps):
            for i, spike_list in enumerate(sp):
                for j, n in enumerate(neurons):
                    spike_train = spike_list[n]
                    self.axis.plot(spike_train,
                                   [j * (num_t + 1) + i + 1
                                      for x in xrange(0, len(spike_train))],
                                   '|',
                                   color=colors[k])

            for j in xrange(0, num_n - 1):
                self.axis.axhline(j * (num_t + 1) + num_t + 1, c='k')

        self.y_lim = (0, num_n * (num_t + 1))

        self.x_ticks = [0, t_stop/2, t_stop]
        self.x_tick_style = 'Custom'
        self.x_lim = (0, t_stop)
        self.x_label = 'time (ms)'
        if num_n == 1:
            self.y_label = 'Trial'
        else:
            self.y_label = 'Neuron/Trial'
        self.y_ticks = []


class SpikeHistogramPlot(SpikeRasterPlot):
    """
    This function plots the raster plot of spikes in the spike_list argument.

    The spike list argument is a list of list of SpikeList objects.
    The top level list corresponds to different sets of spikes that should be
    plotted over each other. They have to contain the same number of neurons.
    Each set of spikes will be colored by the color on corresponding postion of
    the colors parameter. If None all colors will be set to '#848484' (gray).

    The second level list contains different trials of the populations responses
    stored in the individual SpikeList objects.

    Each trial will be plotted as a sepparate line of spikes, and these will be
    grouped by the neurons. Only neurons in the neurons parameter will be
    plotted. If neurons are None, the first neuron will be plotted. """

    def __init__(self, spike_lists, **kwargs):
        SpikeRasterPlot.__init__(self, spike_lists, **kwargs)
        self.parameters["bin_width"] = 5.0

    def plot(self):
        self.colors = ['#000000' for i in xrange(0, len(self.sps))]
        self.neurons = [i for i in xrange(0, min(10, len(self.sps[0][0])))]

        t_stop = float(self.sps[0][0][0].t_stop)

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
        self.x_ticks = [0, t_stop/2, t_stop]
        self.x_lim = (0, t_stop)
        self.x_label = 'time (ms)'


animation_list = []


class StandardStyleAnimatedPlot(StandardStyle):
    """
    This is an abstract class helping construction of animated graphs.

    Each class subclassing from this class should implement the `SimplePlot`
    `plot()` function in which it should draw the first frame of the animation
    with all the corresponding decorations.

    Second it needs to implement the `plot_next_frame` function which should
    replot the data in the plot corresponding to the next frame. This function
    needs to keep track of the frame number itself if it needs it. Note that
    advanced usage of matplotlib is recomanded here so that not the whole graph
    is always replotted but only new data are set to the graph.

    The `StandardStyleAnimatedPlot` will take care of the updating of the
    figure, so now `draw()` or `show()` or event commands should be issued.
    """

    def __init__(self, frame_duration, **kwargs):
        StandardStyle.__init__(self, **kwargs)
        self.lock=False
        self.frame_duration = frame_duration
        self.artists = []

    def plot_next_frame(self):
        raise NotImplementedError

    @staticmethod  # hack to make it compatible with FuncAnimation - we have to make it static
    def _plot_next_frame(b, self):
        a = self.plot_next_frame()
        return a,

    def post_plot(self):
        StandardStyle.post_plot(self)
        import matplotlib.animation as animation
        ani = animation.FuncAnimation(self.axis.figure,
                                      StandardStyleAnimatedPlot._plot_next_frame,
                                      fargs=(self,),
                                      interval=self.frame_duration,
                                      blit=True)
        global animation_list
        animation_list.append(ani)


class PixelMovie(StandardStyleAnimatedPlot):

    def __init__(self, movie, frame_duration, **kwargs):
        StandardStyleAnimatedPlot.__init__(self, frame_duration, **kwargs)
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
        self.im = self.axis.imshow(self.movie[0],
                                   interpolation='nearest',
                                   cmap='gray')


class ScatterPlotMovie(StandardStyleAnimatedPlot):

    def __init__(self, x, y, z, frame_duration, **kwargs):
        StandardStyleAnimatedPlot.__init__(self, frame_duration, **kwargs)
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
        self.scatter.set_array(self.z[self.i, :].flatten())
        self.i = self.i + 1
        if self.i == self.l:
            self.i = 0
        return self.scatter

    def plot(self):
        vmin = 0
        vmax = numpy.max(self.z)
        self.scatter = self.axis.scatter(self.x.flatten(), self.y.flatten(),
                                         c=self.z[0, :].flatten(),
                                         s=self.parameters["dot_size"],
                                         marker=self.parameters["marker"],
                                         lw=1,
                                         cmap='gray',
                                         vmin=vmin,
                                         vmax=vmax)
        pylab.axis('equal')


class ScatterPlot(StandardStyle):
    """
    Simple scatter plot

    The inputs are:

              z -
                      array of the values to be plotted
              x -
                      array with the y positions of the neurons
              y -
                      array with the y positions of the neurons
              periodic -
                      if the z is a periodic value
              period -
                      if periodic is True the z should come from (0,period)
              colorbar -
                      should there be a colorbar ?
    """

    def __init__(self, x, y, z, periodic=False, period=None, **kwargs):
        StandardStyle.__init__(self, **kwargs)
        self.z = z
        self.x = x
        self.y = y
        self.periodic = periodic
        self.period = period
        if self.periodic:
            self.parameters["colormap"] = 'hsv'
        else:
            self.parameters["colormap"] = 'gray'
        self.parameters["dot_size"] = 20
        self.parameters["marker"] = 'o'
        self.parameters["top_right_border"]=True
        self.parameters["colorbar"] = False
        self.parameters["colorbar_label"] = None

    def plot(self):
        if not self.periodic:
            vmax = numpy.max(self.z)
            vmin = numpy.min(self.z)
        else:
            vmax = self.period
            vmin = 0

        ax = self.axis.scatter(self.x, self.y, c=self.z,
                               s=self.dot_size,
                               marker=self.marker,
                               lw=0,
                               cmap=self.colormap,
                               vmin=vmin,
                               vmax=vmax)
        self.axis.set_aspect(aspect=1.0, adjustable='box')
        #self.x_lim = (1.1*numpy.min(self.x), 1.1*numpy.max(self.x))
        #self.y_lim = (1.1*numpy.min(self.y), 1.1*numpy.max(self.y))

        #self.x_ticks = [1.1*numpy.min(self.x), 1.1*numpy.max(self.x)]
        #self.y_ticks = [1.1*numpy.min(self.y), 1.1*numpy.max(self.y)]

        if self.colorbar:
            cb = pylab.colorbar(ax, ticks=[vmin, vmax], use_gridspec=True)
            cb.set_label(self.colorbar_label)
            cb.set_ticklabels(["%.3g" % vmin, "%.3g" % vmax])


class StandardStyleLinePlot(StandardStyle):
    """
    This function plots vector data in simple line plots.

    x, y - lists each containing corresponding vectors of x and y axis values to
           be plotted.

    labels - can contain the labels to be given to the individual
             line plots.

    colors - parameter will determine the colors of the plots. If it is one
             colour all plots will have that same color. If it is a list its
             length should correspond to length of x and y and the
             corresponding colors will be assigned to the individual graphs.

    mean - if the mean of the vectors should be plotted as well.

    """

    def __init__(self, x, y, labels=None, **kwargs):
        StandardStyle.__init__(self, **kwargs)
        self.x = x
        self.y = y
        self.parameters["labels"] = labels
        self.parameters["colors"] = None
        self.parameters["mean"] = False
        self.parameters["linewidth"] = 1

        if self.mean:
            for i in xrange(0, len(x)):
                if not numpy.all(x[i] == x[0]):
                    raise ValueError("Mean cannot be calculated from data not containing identical x axis values")

    def plot(self):
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
            pylab.hold('on')

            tmin = min(tmin, self.x[i][0])
            tmax = min(tmax, self.x[i][-1])

        if self.mean:
            m = m / len(self.x)
            self.axis.plot(self.x[0], m, color='k', linewidth=2*self.linewidth)

        if self.labels!=None:
            self.axis.legend()

        self.xlim = (tmin, tmax)


class ConductancesPlot(StandardStyle):
    """
    Plots conductances.

    exc - list of excitatory conductances (AnalogSignal type)
    inh - list of inhibitory conductances (AnalogSignal type)

    The legend=(True/False) parameter says whether legend should be displayed.
    """

    def __init__(self, exc, inh, **kwargs):
        StandardStyle.__init__(self, **kwargs)
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
            e = e * 1000
            i = i * 1000
            self.axis.plot(time_axis, e.tolist(), color='#F5A9A9')
            self.axis.plot(time_axis, i.tolist(), color='#A9BCF5')
            mean_gsyn_e = mean_gsyn_e + numpy.array(e.tolist())
            mean_gsyn_i = mean_gsyn_i + numpy.array(i.tolist())

        mean_gsyn_i = mean_gsyn_i / len(self.gsyn_is)
        mean_gsyn_e = mean_gsyn_e / len(self.gsyn_es)

        p1, = self.axis.plot(time_axis, mean_gsyn_e.tolist(), color='r', linewidth=2)
        p2, = self.axis.plot(time_axis, mean_gsyn_i.tolist(), color='b', linewidth=2)

        if self.legend:
            self.axis.legend([p1, p2], ['exc', 'inh'])

        self.x_lim = (t_start, t_stop)
        self.x_ticks = [t_start, (t_stop - t_start)/2, t_stop]
        self.x_label = 'time(' + self.gsyn_es[0].t_start.dimensionality.latex + ')'
        self.y_label = 'g(1000' + self.gsyn_es[0].dimensionality.latex + ')'


class ConnectionPlot(StandardStyle):
    """
    Simple scatter plot

    The inputs are:

        pos_x -
                array with x position of the target neurons
        pos_y -
                array with y position of the target neurons

        source_x -
                x position of the source neuron

        source_y -
                y position of the source neuron

        weights -
                array of weights from source neuron to target neurons

        line - True means the existing connections will be displayed as lines, with their thickness indicating the strength.
               False means the strength of connection will be indicated by the size of the circle representing the corresponding target neuron.


        colors - the extra information per neuron that will be displayed as colors. None if no info

        period -
                if period is not None the colors should come from (0,period)

        colorbar -
                should there be a colorbar ?

    """

    def __init__(self, pos_x, pos_y, source_x, source_y, weights, colors=None,
                 period=None, **kwargs):
        StandardStyle.__init__(self, **kwargs)
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.source_x = source_x
        self.source_y = source_y
        self.weights = weights
        self.colors = colors
        self.period = period

        if self.period:
            self.parameters["colormap"] = 'hsv'
        else:
            self.parameters["colormap"] = 'gray'
        self.parameters["top_right_border"] = True
        self.parameters["colorbar"] = False
        self.parameters["colorbar_label"] = None
        self.parameters["line"] = False

    def plot(self):
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
            ax = self.axis.scatter(self.pos_x, self.pos_y, c=self.colors,
                                   s=self.weights/numpy.max(self.weights)*30,
                                   lw=0, cmap=self.colormap,
                                   vmin=vmin, vmax=vmax)
            if self.colorbar:
                cb = pylab.colorbar(ax, ticks=[vmin, vmax], use_gridspec=True)
                cb.set_label(self.colorbar_label)
                cb.set_ticklabels(["%.3g" % vmin, "%.3g" % vmax])

        self.x_label = 'x'
        self.y_label = 'y'
