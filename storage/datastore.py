from NeuroTools.parameters import ParameterSet
from neo.core.segment import Segment
from neo.core.block import Block
from neo.core.spiketrain import SpikeTrain
from neo.io.hdf5io import IOManager
from MozaikLite.stimuli.stimulus_generator import load_from_string, parse_stimuls_id
from MozaikLite.framework.interfaces import MozaikLiteParametrizeObject
from NeuroTools.parameters import ParameterSet
import pickle


class DataStoreView(MozaikLiteParametrizeObject):
    """
    This class represents a subset of a DataStore and defines the query interface and 
    the structure in which the data are stored in the memory of any datastore. Main role 
    of this class is to allow for creating subsets of data stored in the DataStore, so that one can 
    restrict other parts of Mozaik to work only over these subsets. This is done via means of queries
    (see storage.queries) which produce DataStoreView objects and can be chained to work like filters.

    Note that the actual datastores inherit from this object and defines how the data
    are actualy stored on other media (i.e. HD via i.e. hdf5 file format).

    Data store should aggregate all data and analysis results collected across experiments
    and a given model. 

    Experiments results storage:

    TODO

    Analysis results storage:

    There are three categories of results that map to the MoziakLite model structural building blocks:

    Model specific - most general
    Sheet specific 
    Neuron specific

    This is a nested structure of which each level is addressed with appropriate identifier
    and all levels can hold results. In future this structure might be extended with 
    more levels depending on the native structures added to MozaikLite 
    (i.e. cortical area, cortical layer etc.)

    Further at each level the analysis results are addressed by the AnalysisDataStructure
    identifier. The call with this specification returns a set of AnalysisDataStructure's
    that correspond to the above addressing. This can be a list. If further more specific 
    'addressing' is required it has to be done by the corresponding visualization or analysis 
    code that asked for the AnalysisDataStructure's based on the knowledge of their content.
    Or specific querie filters can be written that understand the specific type
    of AnalysisDataStructure and can filter them based on their internal data.
    """
    
    def __init__(self,parameters):
        """
        Just check the parameters, and load the data.
        """
        MozaikLiteParametrizeObject.__init__(self,parameters)
        # we will hold the recordings as one neo Block
        self.block = Block()
        self.analysis_results = {}
        self.analysis_results['data'] = {}
        
    def get_recordings(self,stimuli_name,params=None):
        """
        It will return all recordings to stimuli with name stimuli_name        
        Additionally one can filter out parameters:
        params is a list that should have the same number of elements as the 
        number of free parameters of the given stimulus. Each parameter can be
        set either to '*' indicating pick any stimulus with respect to this parameter
        or to a number which indicates pick only stimuli that have this value of the 
        parameter.
        """
        recordings = []
        
        if stimuli_name == None:
           for seg in self.block._segments: 
               recordings.append(seg)  
        else:   
            for seg in self.block._segments:
                sid = parse_stimuls_id(seg.stimulus)
                if sid.name == stimuli_name:
                   if params: 
                        flag=True    
                        for (f,i) in zip(params,len(params)):
                            if f != '*' and float(f) != sid.parameters[i]:
                               flag=False;
                               break;
                        if flag:
                            recordings.append(seg) 
                   else:
                       recordings.append(seg) 
        
        return recordings


    def get_analysis_result(self,result_id,sheet_name=None,neuron_idx=None):
        if not sheet_name:
           node = self.analysis_results['data'] 
        elif not neuron_idx:
           node = self.analysis_results[sheet_name]['data'] 
        else:
           node = self.analysis_results[sheet_name][neuron_idx]['data'] 
           
        return node[result_id]    
    
    def fromDataStoreView(data_store_view):
        new_dsv = DataStoreView(data_store_view.parameters)
        
        

class DataStore(DataStoreView):
    """
    Abstract DataStore class the declares the parameters, and enforces the interface
    """
    
    required_parameters = ParameterSet({
        'root_directory':str,
    })
    
    def __init__(self, load,parameters):
        """
        Just check the parameters, and load the data.
        """
        DataStoreView.__init__(self,parameters)
        
        # used as a set to quickly identify whether a stimulus was already presented
        # stimuli are otherwise saved with segments within the block as annotations
        self.stimulus_dict = {}
        
        # load the datastore
        if load: 
            self.load()
        
    def identify_unpresented_stimuli(self,stimuli):
        """
        It will filter out from stimuli all those which have already been presented.
        """
        
        unpresented_stimuli = []
        for s in stimuli:
            if not self.stimulus_dict.has_key(str(s)):
               unpresented_stimuli.append(s)
        return unpresented_stimuli
        
    def load(self):
        pass
        
    def save(self):
        pass    
    
    def add_recording(self,segment,stimulus):
        pass
    
    def add_analysis_result(self,result,sheet_name=None,neuron_idx=None):
        pass        

class Hdf5DataStore(DataStore):
    """
    An DataStore that saves all it's data in a hdf5 file and an associated analysis results file, 
    which just becomes the pickled self.analysis_results dictionary.
        
    """
    
    def load(self):
        #load the data
        #iom = IOManager(filename=self.parameters.root_directory+'/datastore.hdf5');
        #self.block = iom.get('/block_0')
        # now just construct the stimulus dictionary
        #for s in self.block._segments:
        #    self.stimulus_dict[s.stimulus]=True
        
        f = open(self.parameters.root_directory+'/datastore.analysis.pickle','r')
        self.analysis_results = pickle.load(f)
        
            
    def save(self):
        #save the recording itself
        #iom = IOManager(filename=self.parameters.root_directory+'/datastore.hdf5');
        #iom.write_block(self.block)
        f = open(self.parameters.root_directory+'/datastore.analysis.pickle','w')
        pickle.dump(self.analysis_results,f)
        
    
    def add_recording(self,segment,stimulus):
        # we get recordings as seg
        segment.stimulus = str(stimulus)
        self.block._segments.append(segment)
        self.stimulus_dict[str(stimulus)]=True

    def add_analysis_result(self,result,sheet_name=None,neuron_idx=None):
        if not sheet_name:
            node = self.analysis_results['data']
        else:
            if not self.analysis_results.has_key(sheet_name):
               self.analysis_results[sheet_name]={}
               self.analysis_results[sheet_name]['data']={}
            
            if not neuron_idx:
               node = self.analysis_results[sheet_name]['data']
            else:
               if not self.analysis_results[sheet_name].has_key(neuron_idx): 
                    self.analysis_results[sheet_name][neuron_idx]={}
                    self.analysis_results[sheet_name][neuron_idx]['data']={}
               node = self.analysis_results[sheet_name][neuron_idx]['data']             
        
        if node.has_key(result.identifier):
            node[result.identifier].append(result)
        else:
            node[result.identifier] = [result]
        
           
