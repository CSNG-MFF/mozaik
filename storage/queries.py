from MozaikLite.framework.interfaces import MozaikLiteParametrizeObject
from MozaikLite.storage.datastore import Hdf5DataStore
from MozaikLite.stimuli.stimulus_generator import load_from_string, parse_stimuls_id
from MozaikLite.tools.misc import create_segment_for_sheets
from NeuroTools.parameters import ParameterSet

class Query(MozaikLiteParametrizeObject):
    """
    Query accepts a DataStoreView and returns a DataStoreView with potentially reduced
    set of recorded data or analysis results
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

class SelectStimuliTypeQuery(Query):
    
    required_parameters = ParameterSet({
     'stimulus_id' : str
    })
      
    def query(self,dsv):  
        new_dsv = dsv.fromDataStoreView()
        new_dsv.analysis_results = dsv.analysis_result_copy()
        
        for s in dsv.block.segments:
            sid = parse_stimuls_id(s.stimulus)
            if sid.name == stimuli_name:
               new_dsv.block.segments.append(s) 

        return new_dsv

class SelectResultSheetQuery(Query):
    
    required_parameters = ParameterSet({
     'sheet_name' : str
    })
      
    def query(self,dsv):  
        new_dsv = dsv.fromDataStoreView()
        new_dsv.analysis_results = dsv.analysis_result_copy()
        
        for seg in dsv.block.segments:
            ns = create_segment_for_sheets([self.parameters.sheet_name])
            ns.stimulus = seg.stimulus
            
            for s in seg.sheets: 
                if s == self.parameters.sheet_name:
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

class TagBasedQuery(Query):
    
    """
    This query filters out all AnalysisDataStructure's corresponding to the given tags
    """
    required_parameters = ParameterSet({
     'tags' : list, # list of tags the the query will look for
    })
      
    def query(self,dsv):  
        new_dsv = dsv.fromDataStoreView()
        new_dsv.block.segments = dsv.recordings_copy()
        new_dsv.analysis_results = self._query(dsv.analysis_results)
        return new_dsv
        
    def _query(self,d):
        nd = {}
        for k in d.keys():
            if k != 'data':
               nd[k] =  self._query(d[k])
            else:
               nd[k] = {}
               
               for key in d[k].keys():
                   nd[k][key] = []
                   for ar in d[k][key]:
                       flag=True 
                       for t in self.parameters.tags:
                           if not (t in ar.tags): 
                              flag = False 

                       if flag:
                          nd[k][key].append(ar)
        return nd
        
        

    
