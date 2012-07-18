from mozaik.framework.interfaces import MozaikParametrizeObject
from mozaik.stimuli.stimulus_generator import StimulusID
from NeuroTools.parameters import ParameterSet
from mozaik.stimuli.stimulus_generator import colapse
import numpy
from queries import *

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
def analysis_data_structure_parameter_filter_query(dsv,**kwargs):
        """
        Returns DSV containing ADSs with matching parameter values
        defined in **kwargs.
        """
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
               flag=False
               break
            if ads_object.inspect_value(k) != kwargs[k]:
               flag=False
               break
        if flag:
           new_ads.append(ads_object)
    return new_ads
########################################################################

########################################################################          
def analysis_data_structure_stimulus_filter_query(dsv,stimulus_name,**kwargs):
        """
        Returns DSV containing ADSs with matching stimulus and matching parameter values
        defined in **kwargs.
        """
        new_dsv = dsv.fromDataStoreView()
        new_dsv.block.segments = dsv.recordings_copy()
        new_dsv.retinal_stimulus = dsv.retinal_stimulus_copy()
        
        for asd in dsv.analysis_results:
            if asd.stimulus_id != None:
                sid = StimulusID(asd.stimulus_id)
                if sid.name == stimulus_name:
                        flag=True    
                        for n,f in kwargs.items():
                            if float(f) != float(sid.params[n]):
                               flag=False;
                               break;
                        if flag:
                            new_dsv.analysis_results.append(asd) 
            
        return new_dsv
########################################################################             

########################################################################
def equal_ads_except(dsv,except_params):
    """
    This functions tests whether DSV contains only ADS of the same kind 
    and parametrization with the exception of parameters listed in
    except_params.
    """
    if ads_is_empty(dsv): return True
    
    first = dsv.analysis_results[0]
    for ads_object in dsv.analysis_results:
        if not first.equalParamsExcept(ads_object,except_params): 
           return False

    return True
########################################################################

########################################################################
def ads_with_equal_stimulus_type(dsv,not_None=False):
    """
    This functions tests whether DSV contains only ADS associated
    with the same stimulus type.
    
    not_None - if true it will not allow ADS that are not associated with stimulus
    """
    if ads_is_empty(dsv): return True 
    
    if dsv.analysis_results[0].stimulus_id != None:
        first = StimulusID(dsv.analysis_results[0].stimulus_id).name
    else:
        if not_None:
            return False
        first = None
    
    for ads_object in dsv.analysis_results:
        if ads_object.stimulus_id != None:
           comp = StimulusID(ads_object.stimulus_id).name
        else:
           comp = None
        
        if comp != first:
            return False
        
    return True
########################################################################

########################################################################
def ads_is_empty(dsv):
    if len(dsv.analysis_results) == 0:
       return True
    return False
########################################################################
