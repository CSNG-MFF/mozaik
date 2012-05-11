from NeuroTools.parameters import ParameterSet
from neo.core.block import Block
#from neo.io.hdf5io import NeoHdf5IO
from mozaik.framework.interfaces import MozaikParametrizeObject
from NeuroTools.parameters import ParameterSet
from neo_neurotools_wrapper import NeoNeurotoolsWrapper, PickledDataStoreNeoWrapper

import cPickle
import numpy
import logging


logger = logging.getLogger("mozaik")


class DataStoreView(MozaikParametrizeObject):
    """
    This class represents a subset of a DataStore and defines the query interface and 
    the structure in which the data are stored in the memory of any datastore. Main role 
    of this class is to allow for creating subsets of data stored in the DataStore, so that one can 
    restrict other parts of Mozaik to work only over these subsets. This is done via means of queries
    (see storage.queries) which produce DataStoreView objects and can be chained to work like filters.

    Note that the actual datastores inherit from this object and define how the data
    are actualy stored on other media (i.e. HD via i.e. hdf5 file format).

    Data store should aggregate all data and analysis results collected across experiments
    and a given model. 

    Experiments results storage:

    These are stored a a simple list in the self.block.segments variable. Each segment corresponds to recordings
    from a single sheet to a single stimulus. The information about the sheet stimulus are stored in the segments
    but are not exposed to the user. Instead there is a set of function that the user can use to access the results:
    
    sheets(self):
    get_stimuli(self):
    get_spike_lists(self):
    get_vm_lists(self):
    get_gsyn_e_lists(self):
    get_gsyn_i_lists(self):

    Analysis results storage:

    There are three categories of results that map to the MoziakLite model structural building blocks:

    Model specific - most general
    Sheet specific 
    Neuron specific

    This is a nested structure of which each level is addressed with appropriate identifier
    and all levels can hold results. In future this structure might be extended with 
    more levels depending on the native structures added to mozaik 
    (i.e. cortical area, cortical layer etc.)

    Further at each level the analysis results are addressed by the AnalysisDataStructure
    identifier. The call with this specification returns a set of AnalysisDataStructure's
    that correspond to the above addressing. This can be a list. If further more specific 
    'addressing' is required it has to be done by the corresponding visualization or analysis 
    code that asked for the AnalysisDataStructure's based on the knowledge of their content.
    Or specific querie filters can be written that understand the specific type
    of AnalysisDataStructure and can filter them based on their internal data.
    
    DataStoreView also keeps a reference to a full Datastore <full_datastore> from which it was originally created 
    (this might be have happened via a chain of DSVs). This is on order to allow for operations that
    work over DSV to insert their results into the original full datastore as this is (almost?) always
    the desired behaviours (note DSV does not actually have functions to add new recordings or analysis 
    results).
    
    """
    
    def __init__(self,parameters,full_datastore):
        """
        Just check the parameters, and load the data.
        """
        MozaikParametrizeObject.__init__(self,parameters)
        # we will hold the recordings as one neo Block
        self.block = Block()
        self.analysis_results = []
        self.retinal_stimulus = {}
        self.full_datastore = full_datastore # should be self if actually the instance is actually DataStore
        
    def get_segments(self):
        return self.block.segments
        
    def sheets(self):
        """
        Returns the list of all sheets that are present in at least one of the segments in the given DataStoreView
        """
        sheets={}
        for s in self.block.segments:
            sheets[s.annotations['sheet_name']] = 1
        return sheets.keys()

    def get_neuron_postions(self):
        return self.full_datastore.block.annotations['neuron_positions'] 

    def get_neuron_annotations(self):
        return self.full_datastore.block.annotations['neuron_annotations'] 
               
    
    def get_stimuli(self):
        """
        Returns a list of stimuli ids. The order of the stimuli ids corresponds to the order of
        segments returned by the get_segments() call.
        """
        return [s.annotations['stimulus'] for s in self.block.segments]
        
   
    def get_analysis_result(self,**kwargs):
        ars = []
        
        for ads in self.analysis_results:
            flag = True
            for k in kwargs.keys():
                if not ads.params().has_key(k):
                   flag=False
                   break
                if ads.inspect_value(k) != kwargs[k]:
                   flag=False
                   break
            if flag:
               ars.append(ads)
        return ars
    
    def get_retinal_stimulus(self,stimuli=None):
        if stimuli == None:
           return self.retinal_stimulus.values()
        else:
           return [self.retinal_stimulus[s] for s in stimuli]
   
    def retinal_stimulus_copy(self):
        new_dict = {}
        for k in self.retinal_stimulus.keys():
            new_dict[k] = self.retinal_stimulus[k]
        return new_dict

    def analysis_result_copy(self):
        return self.analysis_results[:]
                
    def recordings_copy(self):
        return self.block.segments[:]

    def fromDataStoreView(self):
        return DataStoreView(ParameterSet({}),self.full_datastore)

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
        DataStoreView.__init__(self,parameters,self)
        
        # used as a set to quickly identify whether a stimulus was already presented
        # stimuli are otherwise saved with segments within the block as annotations
        self.stimulus_dict = {}
        
        # load the datastore
        if load: 
            self.load()
            
        
    
    def set_neuron_positions(self,neuron_positions):
        self.block.annotations['neuron_positions'] = neuron_positions

    def set_neuron_annotations(self,neuron_annotations):
        self.block.annotations['neuron_annotations'] = neuron_annotations

       
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
        raise NotImplementedError
        pass
        
    def save(self):
        raise NotImplementedError
        pass    
    
    def add_recording(self,segment,stimulus):
        raise NotImplementedError
        pass
    
    def add_analysis_result(self,result,sheet_name=None,neuron_idx=None):
        raise NotImplementedError
        pass        

    def add_retinal_stimulus(self,data,stimulus):
        raise NotImplementedError
        pass        

class Hdf5DataStore(DataStore):
    """
    An DataStore that saves all it's data in a hdf5 file and an associated analysis results file, 
    which just becomes the pickled self.analysis_results dictionary.
    """
    def load(self):
        #load the data
        iom = NeoHdf5IO(filename=self.parameters.root_directory+'/datastore.hdf5');
        self.block = iom.get('/block_0')
        
        # re-wrap segments
        new = []
        for s in self.block.segments:
            new.append(NeoNeurotoolsWrapper(s))
        
        selg.block.segments  = new
        
        #now just construct the stimulus dictionary
        for s in self.block.segments:
             self.stimulus_dict[s.stimulus]=True
             
        f = open(self.parameters.root_directory+'/datastore.analysis.pickle','rb')
        self.analysis_results = cPickle.load(f)
             
            
    def save(self):
        # we need to first unwrap segments from MozaikWrapper
        old = self.block.segments[:]
        self.block.segments = []
        for s in old:
            self.block.segments.append(s.original_segment)    
        
        #save the recording itself
        iom = NeoHdf5IO(filename=self.parameters.root_directory+'/datastore.hdf5');
        iom.write_block(self.block)
        
        # put back wrapped segments
        self.block.segments = old
        
        f = open(self.parameters.root_directory+'/datastore.analysis.pickle','wb')
        cPickle.dump(self.analysis_results,f)
        f.close()

    
    def add_recording(self,segments,stimulus):
        # we get recordings as seg
        for s in segments:
            s.annotations['stimulus'] = str(stimulus)
            self.block.segments.append(NeoNeurotoolsWrapper(s))
        self.stimulus_dict[str(stimulus)]=True

    def add_retinal_stimulus(self,data,stimulus):
        self.retinal_stimulus[str(stimulus)] = data
        
    def add_analysis_result(self,result):
        for ads in self.analysis_results:
            for k in result.params().keys():
                if not ads.params().has_key(k):
                   self.analysis_results.append(result)
                   return
                if ads.inspect_value(k) != result.inspect_value(k):
                   self.analysis_results.append(result)
                   return
        
        if len(self.analysis_results) == 0:
           self.analysis_results.append(result)
           return
        
                
        logger.error("Analysis Data Structure with the same parametrization already added in the datastore. Currently uniquenes is required. The ADS was not added. User should modify analasys specification to avoid this!")
    
class PickledDataStore(Hdf5DataStore):
    """
    An DataStore that saves all it's data as a simple pickled files 
    """
    
    def load(self):
        f = open(self.parameters.root_directory+'/datastore.recordings.pickle','rb')
        self.block = cPickle.load(f)
        for s in self.block.segments:
            self.stimulus_dict[s.annotations['stimulus']]=True
            s.datastore_path = self.parameters.root_directory
        
        f = open(self.parameters.root_directory+'/datastore.analysis.pickle','rb')
        self.analysis_results = cPickle.load(f)
        
        #f = open(self.parameters.root_directory+'/datastore.retinal.stimulus.pickle','rb')
        #self.retinal_stimulus = pickle.load(f)
        
            
    def save(self):
        f = open(self.parameters.root_directory+'/datastore.recordings.pickle','wb')
        cPickle.dump(self.block,f)
        f.close()
       
        f = open(self.parameters.root_directory+'/datastore.analysis.pickle','wb')
        cPickle.dump(self.analysis_results,f)
        f.close()

        #f = open(self.parameters.root_directory+'/datastore.retinal.stimulus.pickle','wb')
        #pickle.dump(self.retinal_stimulus,f)
        #f.close()                
        

    def add_recording(self,segments,stimulus):
        # we get recordings as seg
        for s in segments:
            s.annotations['stimulus'] = str(stimulus)
            self.block.segments.append(PickledDataStoreNeoWrapper(s,'Segment' + str(len(self.block.segments)),self.parameters.root_directory))
            f = open(self.parameters.root_directory+ '/' + 'Segment' + str(len(self.block.segments)-1)+".pickle",'wb')
            cPickle.dump(s,f)
        
        self.stimulus_dict[str(stimulus)]=True
        
    
