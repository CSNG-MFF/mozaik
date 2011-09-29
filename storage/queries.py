from MozaikLite.framework.interfaces import MozaikLiteParametrizeObject
from MozaikLite.storage.datastore import Hdf5DataStore
from MozaikLite.stimuli.stimulus_generator import load_from_string, parse_stimuls_id
from MozaikLite.tools.misc import create_segment_for_sheets
from NeuroTools.parameters import ParameterSet

class Query(MozaikLiteParametrizeObject):
    """
    Query accepts a DataStoreView and returns a DataStoreView with potentially reduced
    set of recorded data or analysis results
    
    We recommend to write queries in a way where it can be invoked via the 
    ParamterSet method uding a class, but also directly as a function with
    (potentially with default paramter values)
    
    See SelectStimuliTypeQuery for a simple example
    """
    required_parameters = ParameterSet({
    })
    
    def __init__(self,parameters):
        """
        Just check the parameters, and load the data.
        """
        MozaikLiteParametrizeObject.__init__(self,parameters)

    def query(self,dsv):
        pass


########################################################################
def select_stimuli_type_query(dsv,stimulus_id=None): 
    new_dsv = dsv.fromDataStoreView()
    new_dsv.analysis_results = dsv.analysis_result_copy()
    
    for s in dsv.block.segments:
        sid = parse_stimuls_id(s.stimulus)
        if sid.name == stimuli_name:
           new_dsv.block.segments.append(s) 

    return new_dsv

class SelectStimuliTypeQuery(Query):
    """
    This query selects recordings in response to a selected query type.
    """
    required_parameters = ParameterSet({
     'stimulus_id' : str
    })
      
    def query(self,dsv):  
        return select_stimuli_type_query(dsv,**self.paramters)    
########################################################################

########################################################################
def select_result_sheet_query(dsv,sheet_name):  
    new_dsv = dsv.fromDataStoreView()
    new_dsv.analysis_results = dsv.analysis_result_copy()
    
    for seg in dsv.block.segments:
        ns = create_segment_for_sheets([sheet_name])
        ns.stimulus = seg.stimulus
        
        for s in seg.sheets: 
            if s == sheet_name:
                    idxs = ns.annotations[s+'_'+'_spikes']
                    for k in seg.annotations[s+'_'+'_spikes']:
                        ns.spiketrains.append(seg._spiketrains[k])
                        idxs.append(len(ns._spiketrains)-1)
                        
                    idxs = ns.annotations[s+'_'+'_vm']
                    for k in seg.annotations[s+'_'+'_vm']:
                        ns.analogsignals.append(seg._analogsignals[k])
                        idxs.append(len(ns._analogsignals)-1)

                    idxs = ns.annotations[s+'_'+'_gsyn_e']
                    for k in seg.annotations[s+'_'+'_gsyn_e']:
                        ns.analogsignals.append(seg._analogsignals[k])
                        idxs.append(len(ns._analogsignals)-1)

                    idxs = ns.annotations[s+'_'+'_gsyn_i']
                    for k in seg.annotations[s+'_'+'_gsyn_i']:
                        ns.analogsignals.append(seg._analogsignals[k])
                        idxs.append(len(ns._analogsignals)-1)
                        
         
        new_dsv.block.segments.append(ns)
    return new_dsv    


class SelectResultSheetQuery(Query):
    
    required_parameters = ParameterSet({
     'sheet_name' : str
    })
    def query(self,dsv):  
        return select_result_sheet_query(dsv,**self.paramters)    
########################################################################      

########################################################################          
def tag_based_query(dsv,tags=[]):  
        new_dsv = dsv.fromDataStoreView()
        new_dsv.block.segments = dsv.recordings_copy()
        new_dsv.analysis_results = _tag_based_query(dsv.analysis_results,tags)
        return new_dsv
        
def _tag_based_query(d,tags):
    nd = {}
    for k in d.keys():
        if k != 'data':
           nd[k] =  _tag_based_query(d[k],tags)
        else:
           nd[k] = {}
           
           for key in d[k].keys():
               nd[k][key] = []
               for ar in d[k][key]:
                   flag=True 
                   for t in tags:
                       if not (t in ar.tags): 
                          flag = False 

                   if flag:
                      nd[k][key].append(ar)
    return nd

class TagBasedQuery(Query):
    
    """
    This query filters out all AnalysisDataStructure's corresponding to the given tags
    """
    required_parameters = ParameterSet({
     'tags' : list, # list of tags the the query will look for
    })
    def query(self,dsv):  
        return tag_based_query(dsv,**self.parameters)    

########################################################################              
        

    
