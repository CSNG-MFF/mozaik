"""
This module contains definition of the PopulationSelector API.
It is used as mechanism for selecting subpopulations of neurons within
Sheets. The most typical use is for selecting neurons for recordings, where 
a PopulationSelector can for example simulate the sampling of neurons 
when using a multi-electrode array of some specific spatial configuration.
"""

from mozaik.core import ParametrizedObject
from mozaik.tools.circ_stat import circular_dist
from parameters import ParameterSet
import math
import numpy
import mozaik

logger = mozaik.getMozaikLogger()

class PopulationSelector(ParametrizedObject):
    """
    The PopulationSelector specifies which cells should be selected from population. 
    
    It defines only one function: generate_idd_list_of_neurons that should 
    return the list of selected neurons ids, based on the provided sheet and parameters.
    
    Parameters
    ----------
    parameters : ParameterSet
               The dictionary of required parameters.
                
    sheet : Sheet
          The sheet from which to pick the neurons
    """
          
    def __init__(self, sheet, parameters):
        ParametrizedObject.__init__(self, parameters)
        self.sheet = sheet  

    def generate_idd_list_of_neurons(self):
        """
        The abastract function that has to be implemented by each `.PopulationSelector` 
        and has to return the list of selected neurons.
        
        Returns
        -------
        ids : list
            List of selected ids.
        """
        raise NotImplemented 

class RCAll(PopulationSelector):
      """
      This PopulationSelector selects all neurons in the sheet.
      """
      def generate_idd_list_of_neurons(self):
          return self.sheet.pop.all_cells.astype(int)

class RCRandomN(PopulationSelector):
      """
      Select random neurons.  

      This PopulationSelector selects *num_of_cells* random neurons from the given population.

      Other parameters
      ----------------
      num_of_cells : int
                   The number of cells to be selected.
      """
      
      required_parameters = ParameterSet({
        'num_of_cells': int,  # The number of cells to be selected
      })  
        
      def generate_idd_list_of_neurons(self):
          z = self.sheet.pop.all_cells.astype(int)
          logger.info("R1: " + str(z[:10]))
          mozaik.rng.shuffle(z)
          logger.info("R1: " + str(z[:10]))
          return z[:self.parameters.num_of_cells]

class RCRandomPercentage(PopulationSelector):
      """
      Select random neurons.

      This PopulationSelector selects *percentage* of randomly chosen neurons from the given population.


      Other parameters
      ----------------
      percentage : float
                   The percentage of neurons to select.

      """
      required_parameters = ParameterSet({
        'percentage': float,  # the cell type of the sheet
      })  
        
      def generate_idd_list_of_neurons(self):
          z = self.sheet.pop.all_cells.astype(int)
          mozaik.rng.shuffle(z)
          return z[:int(len(z)*self.parameters.percentage/100)]

          
class RCGrid(PopulationSelector):
      """
      Select neurons on a grid.

      This PopulationSelector assumes a grid of points ('electrodes') with a 
      given *spacing* and *size*, centered on (*offset_x*,*offset_x*) coordinates.
      It then finds the closest neuron to each point in the grid to be
      inserted into the list of selected neurons .
      
      Other parameters
      ----------------

      size : float (micro meters of cortical space)
           The size of the grid (it is assumed to be square) - it has to be multiple of spacing 
      
      spacing : float (micro meters of cortical space)
           The space between two neighboring electrodes.

      offset_x : float (micro meters of cortical space)
           The x axis offset from the center of the sheet.

      offset_y : float (micro meters of cortical space)
           The y axis offset from the center of the sheet.
      """
      
      required_parameters = ParameterSet({
        'size': float,  # the size of the grid (it is assumed to be square) - it has to be multiple of spacing (micro meters)
        'spacing' : float, #the space between two electrodes (micro meters)
        'offset_x' : float, # the x axis offset from the center of the sheet (micro meters)
        'offset_y' : float, # the y axis offset from the center of the sheet (micro meters)
      })  
      
      def generate_idd_list_of_neurons(self):
          assert math.fmod(self.parameters.size,self.parameters.spacing) < 0.000000001 , "Error the size has to be multiple of spacing!"
          
          picked = []
          z = self.sheet.pop.all_cells.astype(int)
          for x in self.parameters.offset_x + numpy.arange(0,self.parameters.size,self.parameters.spacing) - self.parameters.size/2.0:
              for y in self.parameters.offset_y + numpy.arange(0,self.parameters.size,self.parameters.spacing) - self.parameters.size/2.0:
                  xx,yy = self.sheet.cs_2_vf(x,y)
                  picked.append(z[numpy.argmin(numpy.power(self.sheet.pop.positions[0] - xx,2) +  numpy.power(self.sheet.pop.positions[1] - yy,2))])
          
          logger.info("RCGrid> picked neurons: " + str(picked))
          return list(set(picked))

class SimilarAnnotationSelector(PopulationSelector):
      """
      Choose neurons based on annotations info.

      This PopulationSelector picks random *num_of_cells* neurons whose 
      *annotation* value is closer than *distance* from pre-specified *value* 
      (based on Euclidian norm).
      
      Other parameters
      ----------------
      annotation : str
                 The name of the annotation value. It has to be defined in the given population for all neurons.
      
      distance : float 
		 The the upper limit on distance between the given neurons annotation value and the specified value that permits inclusion.
      
      value : float
	    The value from which to calculate distance.
      
      num_of_cells : int
                   The number of cells to be selected.

      period : float
		The period of the annotation value (0 if none)
      """
      
      required_parameters = ParameterSet({
        'annotation' : str,
        'distance' : float,
        'value': float,
        'num_of_cells': int,  # The number of cells to be selected
        'period' :  float, # if the value is periodic this should be set to the period, oterwise it should be set to 0.
      })  
      def pick_close_to_annotation(self):
          picked = []
          z = self.sheet.pop.all_cells.astype(int)
          vals = [self.sheet.get_neuron_annotation(i,self.parameters.annotation) for i in range(0,len(z))]
          if self.parameters.period != 0:
            picked = numpy.array([i for i in range(0,len(z)) if abs(vals[i]-self.parameters.value) < self.parameters.distance])
          else:
            picked = numpy.array([i for i in range(0,len(z)) if circular_dist(vals[i],self.parameters.value,self.parameters.period) < self.parameters.distance])  
          
          return picked
      
      def generate_idd_list_of_neurons(self):
          picked = self.pick_close_to_annotation()
          mozaik.rng.shuffle(picked)
          return z[picked[:self.parameters.num_of_cells]]
          
          
          
class SimilarAnnotationSelectorRegion(SimilarAnnotationSelector):
      """
      Choose neurons based on annotations info.

      This PopulationSelector picks random *num_of_cells* neurons whose 
      *annotation* value is closer than *distance* from pre-specified *value* 
      (based on Euclidian norm). Furthermore, all selected neurons have to
      sit within a region defined by *size* centered  on (*offset_x*,*offset_x*) coordinates
      (in degrees of visual field).

      Other parameters
      ----------------
      annotation : str
                 The name of the annotation value. It has to be defined in the given population for all neurons.
      
      distance : The the upper limit on distance between the given neurons annotation value and the specified value that permits inclusion.
      
      value : The value from which to calculate distance.
      
      num_of_cells : int
                   The number of cells to be selected.

      size : float (micro meters of cortical space)
           The size of the grid (it is assumed to be square) - it has to be multiple of spacing 
      
      offset_x : float (micro meters of cortical space)
           The x axis offset from the center of the sheet.

      offset_y : float (micro meters of cortical space)
           The y axis offset from the center of the sheet.
      """
      
      required_parameters = ParameterSet({
        'size': float,  # the size of the grid (it is assumed to be square) - it has to be multiple of spacing (micro meters)
        'offset_x' : float, # the x axis offset from the center of the sheet (micro meters)
        'offset_y' : float, # the y axis offset from the center of the sheet (micro meters)
      })  


      
      def generate_idd_list_of_neurons(self):
          picked_or = set(self.pick_close_to_annotation())
          xx,yy = self.sheet.cs_2_vf(self.sheet.pop.positions[0],self.sheet.pop.positions[1])
          picked_region = set(numpy.arange(0,len(xx))[numpy.logical_and(
                                                                         abs(numpy.array(xx - self.parameters.offset_x)) < self.parameters.size/2.0,
                                                                         abs(numpy.array(yy - self.parameters.offset_y)) < self.parameters.size/2.0
                                                      )])
          picked = list(picked_or & picked_region)  
          mozaik.rng.shuffle(picked)
          z = self.sheet.pop.all_cells.astype(int)
          return z[picked[:self.parameters.num_of_cells]]
           
