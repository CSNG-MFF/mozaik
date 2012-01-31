from MozaikLite.visualization.plotting_helper_functions import *
import pylab
import numpy

"""
See visualization.plotting for documentation.
"""

class SimplePlot(object):
        """
        One general policy is that any overwriting of the standard style default values
        by the `Plotting` mechanisms should take precedence over those done by specific 
        instances of `SimplePlot`. In order to enforce this precedence, the modifiable
        parameters of the classes should be stored in the common `dictionary` parameters.
        Each class derived from `SimplePlot` should add its modifiable parameters into the 
        `parameters` dictionary in its constructor.
        
        The kwargs passed to the instances of `SimplePlot` from the `Plotting` mechanisms
        will be stored. At the beginning of the __call__ the dictionary will be updated 
        with these stored kwargs, and the updated parameters (note not all have to be present)
        will be marked. These marked parameters will not be then modifiable any further. 
        In order to do so, the `parameters` dictionary is accessible via the __getattr__
        and __setattr__ functions. 
        
        *Note, for this reason all `SimplePlot` classes need to 
        take care that none of the modifiable attributes is also defined as a class attribute.*
        """
        def pre_plot(self):      
              raise NotImplementedError
              pass

        def plot(self):
              raise NotImplementedError
              pass
          
        def post_plot(self):
              raise NotImplementedError
              pass

        def pre_axis_plot(self):
              raise NotImplementedError
              pass

        def __init__(self,**kwargs):
            self.parameters = {} # the common modifiable parameter dictionary 
            self.to_be_modified_parameters = kwargs
              
          
        def __getattr__(self,name):
                if name == 'parameters':
                   return self.__dict__[name] 

                if self.__dict__['parameters'].has_key(name) and self.__dict__.has_key(name):
                    raise AttributeError , "Error, attribute both in __dict__ and self.parameters" 
                elif self.__dict__['parameters'].has_key(name):
                    return self.__dict__['parameters'][name]
                elif self.__dict__.has_key(name):
                    return self.__dict__[name]
                else:
                    raise AttributeError    
                      
        
        def __setattr__(self,name,value):
                if name == 'parameters':
                   self.__dict__[name] = value
                   return

                if self.__dict__['parameters'].has_key(name) and self.__dict__.has_key(name):
                     raise AttributeError , "Error, attribute both in __dict__ and self.parameters" 
                elif self.__dict__['parameters'].has_key(name):       
                     if not self.__dict__['modified'][name]:
                        self.__dict__['parameters'][name] = value 
                else:       
                     self.__dict__[name] = value
          
        def __call__(self,gs):
              """Calls all the plotting styling and execution functions in the right order"""
              self.update_params()
              self.pre_axis_plot()
              self.axis = pylab.subplot(gs)     
              self.pre_plot()
              self.plot()
              self.post_plot()
        
        def update_params(self):
              """Updates the modifiable parameters and sets them to be further un-modifiable"""
              
              for key in self.to_be_modified_parameters:
                  if not self.__dict__['parameters'].has_key(key):
                     raise AttributeError , ("Error, unknow parameter supplied %s, know parameters: %s" % (key,  self.__dict__['parameters'].keys())) 
              
              self.__dict__['parameters'].update(self.to_be_modified_parameters)   
              self.modified = {}
              for k in self.__dict__['parameters'].keys():
                  self.modified[k] = False
              
              for k in self.to_be_modified_parameters.keys():    
                  self.modified[k] = True  
          
class StandardStyle(SimplePlot):
       
       def __init__(self,**kwargs):
              """
              fontsize            Font size to be used for tick labels and axis labels
              ?_tick_style        The stile of ticks to be plotted 
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
              xlim                what are the xlims (None means matplotlib will infer from data)
              ylim                what are the ylims (None means matplotlib will infer from data)
              xticks              what are the xtikcs (note that the tick style, and x_axis can override/modify these)
              yticks              what are the ytikcs (note that the tick style, and y_axis can override/modify these)
              """
              SimplePlot.__init__(self,**kwargs)
              self.parameters["fontsize"] = 7 # Font size to be used for tick labels and axis labels
              self.parameters["x_tick_style"] = 'Min' # The style of ticks to be plotted 
              self.parameters["y_tick_style"] = 'Min' # The style of ticks to be plotted 
              self.parameters["x_axis"] = True # whether to plot the x_axis (and  the ticks)
              self.parameters["y_axis"] = True # whether to plot the y_axis (and  the ticks)
              self.parameters["xlabel"] = None # what is the xlabel (None means no label will be plotted)
              self.parameters["ylabel"] = None # what is the ylabel (None means no label will be plotted)
              self.parameters["top_right_border"] = False # Whether to plot the top and right border of the axis
              self.parameters["left_border"] = True # Whether to plot the left border of the axis
              self.parameters["bottom_border"] = True # Whether to plot the bottom border of the axis
              self.parameters["title"] = None  # what is the title (None means no label will be plotted)
              self.parameters["xlim"] = None   # what are the xlims (None means matplotlib will infer from data)
              self.parameters["ylim"] = None   # what are the ylims (None means matplotlib will infer from data)
              self.parameters["xticks"] = None # what are the xtikcs (note that the tick style, and x_axis can override/modify these)
              self.parameters["yticks"] = None # what are the ytikcs (note that the tick style, and y_axis can override/modify these)

       def pre_axis_plot(self):
           pylab.rc('axes', linewidth=3)

       def pre_plot(self):
           pass
            
       def post_plot(self):
           
           if self.title:
              pylab.title(title)
           
           if self.xlim:
              pylab.xlim(self.xlim)

           if self.ylim:
              pylab.ylim(self.ylim)
                
           if not self.x_axis:
               disable_xticks(self.axis)
               remove_x_tick_labels()

           if not self.y_axis:
               disable_yticks(self.axis)
               remove_y_tick_labels()

           self.ticks(self.x_tick_style,self.y_tick_style,self.xticks,self.yticks) 

           if self.ylabel:
              pylab.ylabel(self.ylabel)
        
           if self.xlabel:
              pylab.xlabel(self.xlabel)

           if not self.top_right_border:
              disable_top_right_axis(self.axis)

           if not self.left_border:
              disable_left_axis(self.axis)

           if not self.bottom_border:
              disable_bottom_axis(self.axis)

           pylab.rc('axes', linewidth=1)         
        
       
       def ticks(self,x_tick_style,y_tick_style,xticks,yticks):
           if self.xticks != None:
              pylab.xticks(xticks)

           if self.yticks != None:
              pylab.yticks(yticks)
          
           if self.x_tick_style=='Min':
              three_tick_axis(self.axis.xaxis) 
           elif self.x_tick_style=='Custom':
              pass
           else:
              raise ValueError('Unknow x tick style %s', self.x_tick_style)

           if self.y_tick_style=='Min':
              three_tick_axis(self.axis.yaxis) 
           elif self.y_tick_style=='Custom':
              pass
           else:
              raise ValueError('Unknow y tick style %s', self.x_tick_style)

class SpikeRasterPlot(StandardStyle):         
      """
      This function plots the raster plot of spikes in the `spike_lists` argument. 
      
      The `spike_lists` argument is a list of list of SpikeList objects.
      The top level list corresponds to different sets of spikes that should 
      be plotted over each other. They have to contain the same number of neurons. 
      Each set of spikes will be colored by the color on corresponding postion of the 
      colors parameter (matplotlib readable color formats accapted). 
      If None all colors will be set to '#848484' (gray).
      
      The second level list contains different trials of the populations responses 
      stored in the individual SpikeList objects. 

      Each trial will be plotted as a sepparate line of spikes, and these will be
      grouped by the neurons. Only neurons in the neurons parameter will be plotted.
      If neurons are None, (up to) first 10 neurons will be plotted. 
      """
    

      def __init__(self,spike_lists,**kwargs):
          StandardStyle.__init__(self,**kwargs)
          self.sps = spike_lists
          self.parameters["colors"] = None
          self.parameters["neurons"] = None

      def plot(self):  
         if self.parameters["colors"] == None:
            colors = ['#000000' for i in xrange(0,len(self.sps))]
         else:
           colors = self.colors
        
         if self.parameters["neurons"] == None:
            neurons = [i for i in xrange(0,min(10,len(self.sps[0][0])))]
         else:
           neurons = self.neurons
          
         t_stop = float(self.sps[0][0][0].t_stop)
         num_n = len(neurons) # number of neurons
         num_t = len(self.sps[0]) # number of trials
         
         for k, sp in enumerate(self.sps):
             for i,spike_list in enumerate(sp):
                for j,n in enumerate(neurons):
                    spike_train = spike_list[n]
                    self.axis.plot(spike_train,[j*(num_t+1) + i + 1 for x in xrange(0,len(spike_train))],',',color=colors[k])
                 
             for j in xrange(0,num_n-1):   
                self.axis.axhline(j*(num_t+1)+num_t+1,c='k')
         
         self.ylim = (0,num_n * (num_t+1) )
            
         self.xticks = [0,t_stop/2,t_stop]
         self.x_tick_style = 'Custom'
         self.xlim = (0,t_stop)    
         self.xlabel = 'time (ms)'
         if num_n == 1:
            self.ylabel = 'Trial' 
         else:
            self.ylabel = 'Neuron/Trial'
         self.yticks=[]

         
class SpikeHistogramPlot(SpikeRasterPlot):         
    """
    This function plots the raster plot of spikes in the spike_list argument. 
    
    The spike list argument is a list of list of SpikeList objects.
    The top level list corresponds to different sets of spikes that should 
    be plotted over each other. They have to contain the same number of neurons. 
    Each set of spikes will be colored by the color on corresponding postion of the 
    colors parameter. If None all colors will be set to '#848484' (gray).
    
    The second level list contains different trials of the populations responses 
    stored in the individual SpikeList objects. 
    
    Each trial will be plotted as a sepparate line of spikes, and these will be
    grouped by the neurons. Only neurons in the neurons parameter will be plotted.
    If neurons are None, the first neuron will be plotted. 
    """
    
    def plot(self):  
        if self.parameters["colors"] == None:
           colors = ['#000000' for i in xrange(0,len(self.sps))]
        else:
           colors = self.colors
        
        if self.parameters["neurons"] == None:
           neurons = [i for i in xrange(0,min(10,len(self.sps[0][0])))]
        else:
           neurons = self.neurons
        
        t_stop = float(self.sps[0][0][0].t_stop)
        num_n = len(neurons) # number of neurons
        num_t = len(self.sps[0]) # number of trials
        
        all_spikes = []
        for k, sp in enumerate(self.sps):
            tmp = []
            for i,spike_list in enumerate(sp):
                for j in neurons:
                    spike_train = spike_list[j]
                    tmp.extend(spike_train)
            all_spikes.append(tmp)
            
        if all_spikes != []:
           self.axis.hist(all_spikes,bins=numpy.arange(0,t_stop,1),color=colors,edgecolor='none')
        
        self.ylabel = '(spk/ms)'
        self.x_tick_style = 'Custom'
        self.xticks = [0,t_stop/2,t_stop]
        self.xlim = (0,t_stop)    
        self.xlabel = 'time (ms)'

animation_list = []

class StandardStyleAnimatedPlot(StandardStyle):
      """
      This is an abstract class helping construction of animated graphs.
      
      Each class subclassing from this class should implement the `SimplePlot`
      `plot()` function in which it should draw the first frame of the animation
      with all the corresponding decorations.
      
      Second it needs to implement the `plot_next_frame` function which should replot the data
      in the plot corresponding to the next frame. 
      This function needs to keep track of the frame number itself if it needs it.
      Note that advanced usage of matplotlib is recomanded here so that not the 
      whole graph is always replotted but only new data are set to the graph.
      
      The `StandardStyleAnimatedPlot` will take care of the updating of the figure,
      so now `draw()` or `show()` or event commands should be issued.
      """
      def __init__(self,frame_duration,**kwargs):
          StandardStyle.__init__(self,**kwargs)
          self.lock=False
          self.frame_duration = frame_duration
          self.artists = []
          
      def plot_next_frame(self):
          raise NotImplementedError
          pass
      
      @staticmethod # hack to make it compatible with FuncAnimation - we have to make it static
      def _plot_next_frame(b,self):  
          a = self.plot_next_frame()
          return a,

      def post_plot(self):                
          StandardStyle.post_plot(self)
          import matplotlib.animation as animation
          ani = animation.FuncAnimation(self.axis.figure, StandardStyleAnimatedPlot._plot_next_frame, fargs=(self,),interval=self.frame_duration, blit=True)
          global animation_list
          animation_list.append(ani)
          
          
          
  
class PixelMovie(StandardStyleAnimatedPlot):
      def __init__(self,movie,frame_duration,**kwargs):
          StandardStyleAnimatedPlot.__init__(self,frame_duration,**kwargs)
          self.movie = movie
          self.l = len(movie)
          self.i = 0
          self.parameters["left_border"] = False
          self.parameters["bottom_border"] = False
          
      def plot_next_frame(self):
            self.im.set_array(self.movie[self.i])
            self.i=self.i+1
            if self.i == self.l:
               self.i = 0
            return self.im
            
      def plot(self):  
          self.im  = self.axis.imshow(self.movie[0],interpolation='nearest',cmap='gray')
            

