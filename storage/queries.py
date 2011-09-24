from MozaikLite.framework.interfaces import MozaikLiteParametrizeObject
from MozaikLite.storage.datastore import Hdf5DataStore
from MozaikLite.stimuli.stimulus_generator import load_from_string, parse_stimuls_id
from MozaikLite.tools.misc import create_segment_for_sheets


class Query(MozaikLiteParametrizeObject):
"""
Query accepts a DataStoreView and returns a DataStoreView with potentially reduced
set of recorded data or analysis results
"""
    required_parameters = ParameterSet({
    })
    
    def __init__(self, load,parameters):
        """
        Just check the parameters, and load the data.
        """
        MozaikLiteParametrizeObject.__init__(self,parameters)

    def query(self,dsv):
        pass

class SelectStimuliTypeQuery(MozaikLiteParametrizeObject):
    
    required_parameters = ParameterSet({
     'stimulus_id' : str
    })
      
    def query(self,dsv):  
        new_dsv = fromDataStoreView(dsv)
        new_dsv.analysis_results = dsv.analysis_results
        
        for s in dsv._segments:
            sid = parse_stimuls_id(s.stimulus)
            if sid.name == stimuli_name:
               new_dsv._segments.append(s) 

        return new_dsv

class SelectResultSheetQuery(MozaikLiteParametrizeObject):
    
    required_parameters = ParameterSet({
     'sheet_name' : str
    })
      
    def query(self,dsv):  
        new_dsv = fromDataStoreView(dsv)
        new_dsv.analysis_results = dsv.analysis_results
        
        for seg in dsv.block._segments:
            ns = create_segment_for_sheets([self.parameters.sheet_name])
            ns.stimulus = seg.stimulus
            
            for s in seg.sheets: 
                if s == self.parameters.sheet_name:
                        idxs = ns.__getattr__(s+'_'+'_spikes')
                        for k in seg.__getattr__(s+'_'+'_spikes'):
                            ns._spiketrains.append(seg._spiketrains[k])
                            idxs.append(len(ns._spiketrains)-1)
                            
                        idxs = ns.__getattr__(s+'_'+'_vm')
                        for k in seg.__getattr__(s+'_'+'_vm'):
                            ns._analogsignals.append(seg._analogsignals[k])
                            idxs.append(len(ns._analogsignals)-1)

                        idxs = ns.__getattr__(s+'_'+'_gsyn_e')
                        for k in seg.__getattr__(s+'_'+'_gsyn_e'):
                            ns._analogsignals.append(seg._analogsignals[k])
                            idxs.append(len(ns._analogsignals)-1)

                        idxs = ns.__getattr__(s+'_'+'_gsyn_i')
                        for k in seg.__getattr__(s+'_'+'_gsyn_i'):
                            ns._analogsignals.append(seg._analogsignals[k])
                            idxs.append(len(ns._analogsignals)-1)
                            
             
            new_dsv.block._segments.append(ns)
        return new_dsv    
