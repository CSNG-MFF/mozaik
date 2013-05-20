from mozaik.framework.interfaces import MozaikParametrizeObject
from parameters import ParameterSet
import math
import numpy

class PopulationSelector(MozaikParametrizeObject):
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
        MozaikParametrizeObject.__init__(self, parameters)
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
      This PopulationSelector selects random specified number of neurons.
      
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
          numpy.random.shuffle(z)
          return z[:self.parameters.num_of_cells]

class RCRandomPercentage(PopulationSelector):
      """
      This PopulationSelector select random percentage of the population.
      
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
          numpy.random.shuffle(z)
          return z[:int(len(z)*self.parameters.percentage/100)]

          
class RCGrid(PopulationSelector):
      """
      This PopulationSelector assumes a grid of points ('electrodes') and includes the closest neuron to each point to the selected list.
      
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
          print math.fmod(self.parameters.size,self.parameters.spacing)
          assert math.fmod(self.parameters.size,self.parameters.spacing) < 0.000000001 , "Error the size has to be multiple of spacing!"
          
          picked = []
          z = self.sheet.pop.all_cells.astype(int)
          for x in self.parameters.offset_x + numpy.arange(0,self.parameters.size,self.parameters.spacing) - self.parameters.size/2.0:
              for y in self.parameters.offset_y + numpy.arange(0,self.parameters.size,self.parameters.spacing) - self.parameters.size/2.0:
                  xx,yy = self.sheet.cs_2_vf(x,y)
                  picked.append(z[numpy.argmin(numpy.power(self.sheet.pop.positions[0] - xx,2) +  numpy.power(self.sheet.pop.positions[1] - yy,2))])
          
          return list(set(picked))
