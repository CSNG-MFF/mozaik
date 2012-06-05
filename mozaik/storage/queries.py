from mozaik.framework.interfaces import MozaikParametrizeObject
from mozaik.storage.datastore import Hdf5DataStore
from mozaik.stimuli.stimulus_generator import StimulusID
from NeuroTools.parameters import ParameterSet
from mozaik.stimuli.stimulus_generator import colapse
import numpy



########################################################################
class Query(MozaikParametrizeObject):
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
        MozaikParametrizeObject.__init__(self,parameters)

    def query(self,dsv):
        raise NotImplementedError
        pass


########################################################################
def select_stimuli_type_query(dsv,stimulus_name,params=None): 
    new_dsv = dsv.fromDataStoreView()
    new_dsv.analysis_results = dsv.analysis_result_copy()
    new_dsv.retinal_stimulus = dsv.retinal_stimulus_copy()

   
    for seg in dsv.block.segments:
        sid = StimulusID(seg.annotations['stimulus'])
        if sid.name == stimulus_name:
           if params: 
                flag=True    
                for n,f in params.items():
                    if float(f) != float(sid.params[n]):
                       flag=False;
                       break;
                if flag:
                    new_dsv.block.segments.append(seg) 
           else:
               new_dsv.block.segments.append(seg) 
    return new_dsv

class SelectStimuliTypeQuery(Query):
    """
    It will return all recordings to stimuli with name stimuli_name        
    Additionally one can filter out parameters:
    params is a list that should have the same number of elements as the 
    number of free parameters of the given stimulus. Each parameter can be
    set either to None indicating pick any stimulus with respect to this parameter
    or to a value (number or string) which indicates pick only stimuli that have this value of the 
    parameter.
    """
    required_parameters = ParameterSet({
     'stimulus_id' : str,
     'params' : list
    })
      
    def query(self,dsv):  
        return select_stimuli_type_query(dsv,**self.paramters)    
########################################################################

########################################################################
def select_result_sheet_query(dsv,sheet_name):  
    new_dsv = dsv.fromDataStoreView()
    new_dsv.analysis_results = dsv.analysis_result_copy()
    new_dsv.retinal_stimulus = dsv.retinal_stimulus_copy()
    
    for seg in dsv.block.segments:
        if seg.annotations['sheet_name'] == sheet_name:
            new_dsv.block.segments.append(seg)
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
        new_dsv.retinal_stimulus = dsv.retinal_stimulus_copy()
        new_dsv.analysis_results = _tag_based_query(dsv.analysis_results,tags)
        return new_dsv
        
def _tag_based_query(d,tags):
    nd = []
    for a in d:
        flag = True
        for t in tags:
            if not (t in a.tags): 
               flag = False 
    
        if flag:
           nd.append(a)
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


########################################################################          
def identifier_based_query(dsv,identifier):  
        new_dsv = dsv.fromDataStoreView()
        new_dsv.block.segments = dsv.recordings_copy()
        new_dsv.retinal_stimulus = dsv.retinal_stimulus_copy()
        new_dsv.analysis_results = _identifier_based_query(dsv.analysis_results,identifier)
        return new_dsv
        
def _identifier_based_query(d,identifier):
    nd = []
    for a in d:
        if a.identifier == identifier:
           nd.append(a)
    return nd

class IdentifierBasedQuery(Query):
    
    """
    This query filters out all AnalysisDataStructure's corresponding to the given tags
    """
    required_parameters = ParameterSet({
     'identifier' : str, # list of tags the the query will look for
    })
    def query(self,dsv):  
        return identifier_based_query(dsv,**self.parameters)    
########################################################################                      


########################################################################
def partition_by_stimulus_paramter_query(dsv,parameter_name):  
        st = dsv.get_stimuli()
        values,st = colapse(numpy.arange(0,len(st),1),st,parameter_list=[parameter_name])
        dsvs = []
        for vals in values:
            new_dsv = dsv.fromDataStoreView()
            new_dsv.analysis_results = dsv.analysis_result_copy()
            for v in vals:
                new_dsv.block.segments.append(dsv.block.segments[v])
            dsvs.append(new_dsv)
        return dsvs


class PartitionByStimulusParamterQuery(Query):
    """
    This query will take all recordings and return list of DataStoreViews
    each holding recordings measured to the same stimulus with exception of
    the paramter reference by parameter_name.
    
    Note that in most cases one wants to do this only against datastore holding only
    single Stimulus type! In that case the datastore is partitioned into subsets each holding 
    recordings to the same stimulus with the same paramter values, with the
    exception to the parameter_name
    """
    
    required_parameters = ParameterSet({
     'parameter_name' : list, # the index of the parameter against which to partition
    })
    
    def query(self,dsv):  
        return partition_by_stimulus_paramter_query(dsv,**self.parameters)    
########################################################################


########################################################################
def partition_recordings_by_sheet_query(dsv):  
        dsvs = []
        for sheet in dsv.sheets():
            new_dsv = dsv.fromDataStoreView()
            new_dsv.analysis_results = dsv.analysis_result_copy()
            new_dsv.retinal_stimulus = dsv.retinal_stimulus_copy()

            for seg in dsv.block.segments:
                if seg.annotations['sheet_name'] == sheet:
                   new_dsv.block.segments.append(seg)
    
            dsvs.append(new_dsv)
        return dsvs


class PartitionRecordingsBySheetQuery(Query):
    """
    This query will take all recordings and return list of DataStoreViews
    each corresponding to one of the sheets and holding recordings comming
    from the corresponding sheet.
    """
    
    def query(self,dsv):  
        return partition_recordings_by_sheet_query(dsv)    
########################################################################


########################################################################
def partition_analysis_results_by_parameter_name_query(dsv,ads_identifier='',parameter_name=''):
    dsv = identifier_based_query(dsv,ads_identifier)
    
    partiotioned_dsvs = {}
    
    for ads_object in dsv.analysis_results:
        partiotioned_dsvs.setdefault(ads_object.inspect_value(parameter_name),[]).append(ads_object)
    
    dsvs = []
    
    for k in partiotioned_dsvs.keys():
        new_dsv = dsv.fromDataStoreView()
        new_dsv.block.segments = dsv.recordings_copy()
        new_dsv.retinal_stimulus = dsv.retinal_stimulus_copy()
        new_dsv.analysis_results = partiotioned_dsvs[k]
        dsvs.append(new_dsv)    
    
    return dsvs

class PartitionAnalysisResultsByParameterNameQuery(Query):
    """
    This query takes in a name of a _parameter_.
    This query will take all analysis results and it will parition them into DSVs
    each holding only analysis results that have the same value of the _parameter_.
    Thus for each value of the _parameter_ existing in the DSV there will be new DSV
    created.
    
    Note that this query will fail if it encounters any analysis results that do not
    specify the _parameter_.
    """
    
    required_parameters = ParameterSet({
     'ads_identifier' : str, # the ADS identifier for which to the the partition
     'parameter_name' : str, # the index of the parameter against which to partition
    })
    
    def query(self,dsv):  
        return partition_analysis_results_by_parameter_name_query(dsv,**self.parameters)    
########################################################################

########################################################################          
def analysis_data_structure_parameter_filter_query(dsv,identifier,**kwargs):
        """
        The 
        """
        dsv = identifier_based_query(dsv,identifier)  
        new_dsv = dsv.fromDataStoreView()
        new_dsv.block.segments = dsv.recordings_copy()
        new_dsv.retinal_stimulus = dsv.retinal_stimulus_copy()
        new_dsv.analysis_results = _adspfq_recursive(dsv.analysis_results,**kwargs)
        return new_dsv
        
def _adspfq_recursive(d,**kwargs):
    new_ads = []
    for ads_object in d:
        flag=True
        for k in kwargs.keys():
            
            if not ads_object.params().has_key(k):
               raise ValueError("analysis_data_structure_parameter_filter_query: no Parameter %s in object" % k)
            if ads_object.inspect_value(k) != kwargs[k]:
               flag=False
               break
        if flag:
           new_ads.append(ads_object)
    return new_ads
########################################################################
