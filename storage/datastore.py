from NeuroTools.parameters import ParameterSet
from neo.core.segment import Segment
from neo.core.block import Block
from neo.core.spiketrain import SpikeTrain
from neo.io.hdf5io import IOManager
from MozaikLite.stimuli.stimulus_generator import load_from_string, parse_stimuls_id
from NeuroTools.parameters import ParameterSet

class DataStore(object):
    """
        Data store should aggregate all data collected across experiments
        a given model. 
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
            
    def save(self):
        #save the recording itself
        iom = IOManager(filename=self.parameters.root_directory+'/datastore.hdf5');
        iom.write_block(self.block)
    
    def add_recording(self,segment,stimulus):
        # we get recordings as seg
        segment.stimulus = str(stimulus)
        self.block._segments.append(segment)
        self.stimulus_dict[str(stimulus)]=True
