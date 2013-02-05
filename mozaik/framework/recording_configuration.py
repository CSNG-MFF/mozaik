from mozaik.framework.interfaces import MozaikParametrizeObject
from NeuroTools.parameters import ParameterSet
import numpy

class RecordingConfiguration(MozaikParametrizeObject):
    """
    The RecordingConfiguration specifies which cells should be recorded from. 
    
    It defines only one function: generate_idd_list_of_neurons_to_record that should 
    return the list of neurons ids to record, based on the provided sheet and parameters.
    """
          
    def __init__(self, sheet, parameters):
        MozaikParametrizeObject.__init__(self, parameters)
        self.sheet = sheet  

    def generate_idd_list_of_neurons_to_record(self):
        raise NotImplemented 

class RCAll(RecordingConfiguration):
      """
      This RecordingConfiguration records all neurons in the sheet.
      """
      def generate_idd_list_of_neurons_to_record(self):
          return [a for a in self.sheet.pop.all()]

class RCRandomN(RecordingConfiguration):
      """
      This RecordingConfiguration records random specified number of neurons.
      """
      required_parameters = ParameterSet({
        'num_of_cells': int,  # the cell type of the sheet
      })  
        
      def generate_idd_list_of_neurons_to_record(self):
          return numpy.random.permutation([a for a in self.sheet.pop.all()])[:self.parameters.num_of_cells]

class RCGrid(RecordingConfiguration):
      """
      This RecordingConfiguration assumes a grid of 'electrodes' and includes the closest neuron to each 'electrode' to the recording list.
      """
      required_parameters = ParameterSet({
        'size': float,  # the size of the electrode array (it is assumed to be square) - it has to be multiple of spacing
        'spacing' : float, #the space between two electrodes 
        'offset_x' : float, # the x axis offset from the lower,left corner of the sheet
        'offset_Y' : float, # the y axis offset from the lower,left corner of the sheet
      })  
      
      def generate_idd_list_of_neurons_to_record(self):
          assert fmod(self.parameters.size,self.parameters.spacing) == 0 , "Error the size has to be multiple of spacing!"
          
          picked = []
          
          for x in self.parameters.offset_x + numpy.arange(0,self.parameters.size,self.parameters.spacing):
              for y in self.parameters.offset_y + numpy.arange(0,self.parameters.size,self.parameters.spacing):
                  picked.append(self.sheet.pop.all()[numpy.argmin(numpy.power(self.sheet.positions[0] - x,2) +  numpy.power(self.sheet.positions[0] - x,2))])
          
          return list(set(picked))
