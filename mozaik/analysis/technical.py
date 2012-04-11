from mozaik.analysis.analysis_data_structures import PerNeuronValue
from mozaik.analysis.analysis import Analysis
from sets import Set
import quantities as qt
import numpy

class NeuronAnnotationsToPerNeuronValues(Analysis):
      """
      Creates a PerNeuronValues analysis data structure per each neuron annotation 
      that is defined for all neurons in a given sheet.
      
      This method is aware of several specific annotations and adds additional appropriate information 
      to the PerNeuronValue ADS (i.e. setting period to numpy.pi of orientation preference of initial connection fields).
      Users are expected to modify this class to add additional information for their new annotations if required.
      It is assumed that in future the handling of parameters around Mozaik might be enhanced and unified further to avoid
      extension of this class.
      """
      
      def analyse(self):
        print 'Starting NeuronAnnotationsToPerNeuronValues Analysis'
        anns = self.datastore.get_neuron_annotations()
        
        for sheet in self.datastore.sheets():
            keys = Set([])

            for n in xrange(0,len(anns[sheet])):
                keys = keys.union(anns[sheet][n].keys())
            
            for k in keys:
                # first check if the key is defined for all neurons
                key_ok = True
                
                for n in xrange(0,len(anns[sheet])):
                    if not anns[sheet][n].has_key(k):
                       key_ok = False
                       break
                
                if key_ok:
                   values = []
                   for n in xrange(0,len(anns[sheet])):
                       values.append(anns[sheet][n][k])
                   print 'Adding PerNeuronValue: ' , k
                   
                   period=None
                   if k == 'LGNAfferentOrientation':
                      period = numpy.pi
                   if k == 'LGNAfferentPhase':
                      period = 2*numpy.pi  
                   
                   self.datastore.full_datastore.add_analysis_result(PerNeuronValue(values,qt.dimensionless,period=period,value_name=k,sheet_name=sheet,tags=self.tags)) 
