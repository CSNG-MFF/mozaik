from mozaik.framework.interfaces import MozaikParametrizeObject
from parameters import ParameterSet
import math
import numpy

class PopulationSelector(MozaikParametrizeObject):
    """
    The PopulationSelector specifies which cells should be selected from population. 
    
    It defines only one function: generate_idd_list_of_neurons that should 
    return the list of selected neurons ids, based on the provided sheet and parameters.
    """
          
    def __init__(self, sheet, parameters):
        MozaikParametrizeObject.__init__(self, parameters)
        self.sheet = sheet  

    def generate_idd_list_of_neurons(self):
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
      """
      required_parameters = ParameterSet({
        'num_of_cells': int,  # the cell type of the sheet
      })  
        
      def generate_idd_list_of_neurons(self):
          z = self.sheet.pop.all_cells.astype(int)
          numpy.random.shuffle(z)
          return z[:self.parameters.num_of_cells]

class RCRandomPercentage(PopulationSelector):
      """
      This PopulationSelector select random percentage of the population.
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
