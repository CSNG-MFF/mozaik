"""
This plot contains classes that assist with construction of complicated arrangements of plots.
The typical example is class that help with creating a line of plots with common y axis.
"""
import param
from param.parameterized import Parameterized
from mozaik.storage.queries import *
import matplotlib.gridspec as gridspec



class LinePlot(Parameterized):          
        """
        Plot multiple plots with common x or y axis in a row or column.
        """ 
        horizontal = param.Boolean(default=True,instantiate=True,doc="Should the line of plots be horizontal or vertical")
        shared_axis = param.Boolean(default=True,instantiate=True,doc="should the x axis or y axis (depending on the horizontal flag) considered shared") 
        length = param.Integer(default=0,instantiate=True,doc="how many plots will there be")
        function = param.Callable(doc="The function that should be called to plot individual plots. It should accept three parameters: self,index in the line, gridspec object into which to plot the plot, the simple_plot parameters")
        
        
        def make_line_plot(self,subplotspec,params):
            """
            Call to execute the line plot.
            
            funtion - is the function that plots the individual plots. Function has to accept idx
            
            subplotspec - is the subplotspec into which the whole lineplot is to be plotted
            params - are the simple plot parameters that will be modified and passed to each of the subpots
            """
            if not self.length:
               print 'Length not specified'
               return
            
            if self.horizontal:
                gs = gridspec.GridSpecFromSubplotSpec(1, self.length, subplot_spec=subplotspec)
            else:
                gs = gridspec.GridSpecFromSubplotSpec(self.length ,1 , subplot_spec=subplotspec)
            
            for idx in xrange(0,self.length):
              p = params.copy()
              if idx > 0 and self.shared_axis and self.horizontal:
                  p.setdefault("y_label",None)
              if (idx < self.length-1) and self.shared_axis and not self.horizontal:
                  p.setdefault("x_label",None)
                          
              if self.horizontal:
                  self._single_plot(idx,gs[0,idx],p)
              else:
                  self._single_plot(idx,gs[idx,0],p)
        
        def _single_plot(self,idx,gs,p):
            self.function(idx,gs,p)


class PerDSVPlot(LinePlot):          
      """
      This is a LinePlot that automatically partitions the datastore view based on some rules. And than
      executes the individual plots on the line over the individual DSVs.
      
      Unlike LinePlot the function parameter should accept as the first arguemnt DSV not the index of the plot on the line.
      
      The partition dsvs function should perform the partitioning.
      """
      
      function = param.Callable(instantiate=True,doc="The function that should be called to plot individual plots. It should accept three parameters: self,the DSV from which to generate the plot, gridspec object into which to plot the plot, the simple_plot parameters")

      def  __init__(self,datastore,**params):
           LinePlot.__init__(self,**params)
           self.datastore = datastore
           self.dsvs = self.partiotion_dsvs()
           self.length = len(self.dsvs)
      
      def partiotion_dsvs(self):
          raise NotImplementedError
          pass
        
      def _single_plot(self,idx,gs,p):
          f = self.function
          self.function(self.dsvs[idx],gs,p)

class PerStimulusPlot(PerDSVPlot):
    """
    Line plot where each plot corresponds to stimulus with the same parameter except trials.
    
    The self.dsvs will contain the datastores you want to plot in each of the subplots - i.e. all recordings
    in the given datastore come from the same stimulus of the same parameters except for the trial parameter.
    """
    def partiotion_dsvs(self):       
        return partition_by_stimulus_paramter_query(self.datastore,8)

    def _single_plot(self,idx,gs,p):
            stimulus = self.dsvs[idx].get_stimuli()[0]
            p.setdefault("title",str(stimulus))
            PerDSVPlot._single_plot(self,idx,gs,p)
            
            
            
            