from mozaik.analysis.analysis_data_structures import PerNeuronValue
from mozaik.analysis.analysis import Analysis
from sets import Set
import quantities as qt

class NeuronAnnotationsToPerNeuronValues(Analysis):
      """
      Creates a PerNeuronValues analysis data structure per each neuron annotation 
      that is defined for all neurons in a given sheet.
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
                   print 'BBBBBBBBBBBBBBBBBB'
                   self.datastore.full_datastore.add_analysis_result(PerNeuronValue(values,'key', qt.dimensionless , sheet, tags=self.tags),sheet_name=sheet) 
