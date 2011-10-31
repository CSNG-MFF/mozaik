from NeuroTools.parameters import ParameterSet
from neo.core.segment import Segment
from neo.core.block import Block
from neo.core.spiketrain import SpikeTrain
from neo.io.hdf5io import IOManager
from NeuroTools import signals
from MozaikLite.stimuli.stimulus_generator import load_from_string, parse_stimuls_id
from MozaikLite.tools.misc import spike_dic_to_list
from MozaikLite.framework.interfaces import MozaikLiteParametrizeObject
from NeuroTools.parameters import ParameterSet

import pickle
import numpy


class DataStoreView(MozaikLiteParametrizeObject):
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
    more levels depending on the native structures added to MozaikLite 
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
        MozaikLiteParametrizeObject.__init__(self,parameters)
        # we will hold the recordings as one neo Block
        self.block = Block()
        self.analysis_results = {}
        self.retinal_stimulus = {}
        self.analysis_results['data'] = {}
        self.full_datastore = full_datastore # should be self if actually the instance is actually DataStore
        
    def get_segments(self):
        return self.block.segments
        
    def sheets(self):
        """
        Returns the list of all sheets that are present in at least one of the segments in the give DataStoreView
        """
        sheets={}
        for s in self.block.segments:
            sheets[s.annotations['sheet_name']] = 1
        return sheets.keys()
    
    def get_stimuli(self):
        """
        Returns a list of stimuli ids. The order of the stimuli ids corresponds to the order of
        segments returned by the get_segments() call.
        """
        return [s.annotations['stimulus'] for s in self.block.segments]
        
    
        
    def get_spike_lists(self):
        """
        Returns a list of Neurotools SpikeList objects. The order of the spikelists corresponds to the order of
        segments returned by the get_segments() call.
        """
        sl = []    
        for s in self.block.segments:
            t_start = s.spiketrains[0].t_start
            t_stop = s.spiketrains[0].t_stop
           
            d = {}
            for st in s.spiketrains:
                d[st.annotations['index']] = numpy.array(st)
           
            spikes = signals.SpikeList(spike_dic_to_list(d),d.keys(),float(t_start),float(t_stop))
            sl.append(spikes)
        return sl    
        
    def get_vm_lists(self):
        """
        Returns a list of Neurotools SpikeList objects. The order of the spikelists corresponds to the order of
        segments returned by the get_segments() call.
        """
        ass = []    
        for s in self.block.segments:
            a = []
            for i in s.annotations['vm']:
                a.append(signals.AnalogSignal(s.analogsignals[i],dt=s.analogsignals[i].sampling_period))
            ass.append(a)    
        return ass    

    def get_gsyn_e_lists(self):
        """
        Returns a list of Neurotools SpikeList objects. The order of the spikelists corresponds to the order of
        segments returned by the get_segments() call.
        """
        ass = []    
        for s in self.block.segments:
            a = []
            for i in s.annotations['gsyn_e']:
                a.append(signals.AnalogSignal(s.analogsignals[i],dt=s.analogsignals[i].sampling_period))
            ass.append(a)
        return ass

    def get_gsyn_i_lists(self):
        """
        Returns a list of Neurotools SpikeList objects. The order of the spikelists corresponds to the order of
        segments returned by the get_segments() call.
        """
        ass = []    
        for s in self.block.segments:
            a = []
            for i in s.annotations['gsyn_i']:
                a.append(signals.AnalogSignal(s.analogsignals[i],dt=s.analogsignals[i].sampling_period))
            ass.append(a)
        return ass    
   
    def get_analysis_result(self,result_id,sheet_name=None,neuron_idx=None):
        if not sheet_name:
           node = self.analysis_results['data'] 
        elif not neuron_idx:
           node = self.analysis_results[sheet_name]['data'] 
        else:
           node = self.analysis_results[sheet_name][neuron_idx]['data'] 
        
        if node.has_key(result_id):
            return node[result_id]    
        else:
            return []
    

    def get_retinal_stimulus(self,stimulus):
        return self.retinal_stimulus[stimulus]
   
    def _analysis_result_copy(self,d):
        nd = {}
        for k in d.keys():
            if k != 'data':
               nd[k] =  self._analysis_result_copy(d[k])
            else:
               nd[k] = d[k].copy() 
        return nd    

    def analysis_result_copy(self):
        return self._analysis_result_copy(self.analysis_results)
                
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
        #iom = IOManager(filename=self.parameters.root_directory+'/datastore.hdf5');
        #self.block = iom.get('/block_0')
        # now just construct the stimulus dictionary
        #for s in self.block.segments:
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
        for s in segment:
            s.annotations['stimulus'] = str(stimulus)
        self.block.segments.extend(segment)
        self.stimulus_dict[str(stimulus)]=True

    def add_retinal_stimulus(self,data,stimulus):
        self.retinal_stimulus[str(stimulus)] = data
        
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
        
class PickledDataStore(Hdf5DataStore):
    """
    An DataStore that saves all it's data as a simple pickled files 
    """
    
    def load(self):
        f = open(self.parameters.root_directory+'/datastore.recordings.pickle','r')
        self.block = pickle.load(f)
        for s in self.block.segments:
            self.stimulus_dict[s.annotations['stimulus']]=True
        
        f = open(self.parameters.root_directory+'/datastore.analysis.pickle','r')
        self.analysis_results = pickle.load(f)
        
        f = open(self.parameters.root_directory+'/datastore.retinal.stimulus.pickle','r')
        self.retinal_stimulus = pickle.load(f)
        
            
    def save(self):
        f = open(self.parameters.root_directory+'/datastore.recordings.pickle','w')
        pickle.dump(self.block,f)
        f.close()
        
        f = open(self.parameters.root_directory+'/datastore.retinal.stimulus.pickle','w')
        pickle.dump(self.retinal_stimulus,f)
        f.close()                
        
        f = open(self.parameters.root_directory+'/datastore.analysis.pickle','w')
        pickle.dump(self.analysis_results,f)
        f.close()
        
