from NeuroTools.parameters import ParameterSet
from neo.core.segment import Segment
from neo.core.block import Block
from neo.core.spiketrain import SpikeTrain
from neo.io.hdf5io import IOManager
from MozaikLite.stimuli.stimulus_generator import load_from_string, parse_stimuls_id
from NeuroTools.parameters import ParameterSet
import pickle

class DataStore(object):
    """
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
    """
    
    required_parameters = ParameterSet({
        'root_directory':str,
    })
    
    def __init__(self, load,parameters):
        """
        Just check the parameters, and load the data.
        """
        self.check_parameters(parameters)
        self.parameters = parameters
        
        # we will hold the recordings as one neo Block
        self.block = Block()
        
        # used as a set to quickly identify whether a stimulus was already presented
        # stimuli are otherwise saved with segments within the block as annotations
        self.stimulus_dict = {}
        
        self.analysis_results = {}
        self.analysis_results['data'] = {}
        
        # load the datastore
        if load: 
            self.load()
        
        
    
    def check_parameters(self, parameters):
        def walk(tP, P, section=None):
            if set(tP.keys()) != set(P.keys()):
                raise KeyError("Invalid parameters for %s.%s Required: %s. Supplied: %s" % (self.__class__.__name__, section or '', tP.keys(), P.keys()))
            for k,v in tP.items():
                if isinstance(v, ParameterSet):
                    assert isinstance(P[k], ParameterSet), "Type mismatch: %s !=  ParameterSet, for %s " % (type(P[k]),P[k]) 
                    walk(v, P[k],section=k)
                else:
                    assert isinstance(P[k], v), "Type mismatch: %s !=  %s, for %s" % (v,type(P[k]),P[k])
        try:
    	    # we first need to collect the required parameters from all the classes along the parent path
            new_param_dict={}
	    for cls in self.__class__.__mro__:
	        # some parents might not define required_parameters 
	        # if they do not require one or they are the object class
	        if hasattr(cls, 'required_parameters'):
		       new_param_dict.update(cls.required_parameters.as_dict())
	    walk(ParameterSet(new_param_dict), parameters)
    	except AssertionError as err:
            raise Exception("%s\nInvalid parameters.\nNeed %s\nSupplied %s" % (err,ParameterSet(new_param_dict),parameters))  


    def identify_unpresented_stimuli(self,stimuli):
        """
        It will filter out from stimuli all those which have already been presented.
        """
        
        unpresented_stimuli = []
        for s in stimuli:
            if not self.stimulus_dict.has_key(str(s)):
               unpresented_stimuli.append(s)
        return unpresented_stimuli
    
    def get_recordings(self,stimuli_name,params=None):
        """
        It will return all recordings to stimuli with name stimuli_name        
        Additionally one can filter our parameters:
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
    
    def load(self):
        #load the data
        iom = IOManager(filename=self.parameters.root_directory+'/datastore.hdf5');
        self.block = iom.get('/block_0')
        # now just construct the stimulus dictionary
        for s in self.block._segments:
            self.stimulus_dict[s.stimulus]=True
        
        f = open(self.parameters.root_directory+'/datastore.analysis.pickle','r')
        self.analysis_results = pickle.load(f)
        
            
    def save(self):
        #save the recording itself
        iom = IOManager(filename=self.parameters.root_directory+'/datastore.hdf5');
        iom.write_block(self.block)
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
        
    def get_analysis_result(self,result_id,sheet_name=None,neuron_idx=None):
        if not sheet_name:
           node = self.analysis_results['data'] 
        elif not neuron_idx:
           node = self.analysis_results[sheet_name]['data'] 
        else:
           node = self.analysis_results[sheet_name][neuron_idx]['data'] 
           
        return node[result_id]    
           
