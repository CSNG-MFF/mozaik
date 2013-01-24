"""
docstring goes here
"""
from mozaik.framework.interfaces import MozaikParametrizeObject
from NeuroTools.parameters import ParameterSet
from mozaik.tools.mozaik_parametrized import colapse,  MozaikParametrized, filter_query, matching_parametrized_object_params
import numpy


class Query(MozaikParametrizeObject):
    """
    Query accepts a DataStoreView and returns a DataStoreView with potentially
    reduced set of recorded data or analysis results

    We recommend to write queries in a way where it can be invoked via the
    ParamterSet method uding a class, but also directly as a function with
    (potentially with default paramter values)

    See SelectStimuliTypeQuery for a simple example
    """
    required_parameters = ParameterSet({
    })

    def __init__(self, parameters):
        """
        Just check the parameters, and load the data.
        """
        MozaikParametrizeObject.__init__(self, parameters)

    def query(self, dsv):
        raise NotImplementedError


########################################################################
def param_filter_query(dsv,**kwargs):
    new_dsv = dsv.fromDataStoreView()
    
    st_kwargs = dict([(k[3:],kwargs[k]) for k in kwargs.keys() if k[0:3] == 'st_'])
    kwargs = dict([(k,kwargs[k]) for k in kwargs.keys() if k[0:3] != 'st_'])
    
    seg_st = [MozaikParametrized.idd(seg.annotations['stimulus']) for seg in dsv.block.segments]
    ads_st = [MozaikParametrized.idd(ads.stimulus_id) for ads in dsv.analysis_results if ads.stimulus_id != None]
    
    if 'sheet_name' in set(kwargs):
       if len(kwargs) == 1:
           # This means that there is only one 'non-stimulus' parameter sheet, and thus we need
           # to filter out all recordings that are associated with that sheet (otherwsie we do not pass any recordings)
           seg_filtered = set([s for s in dsv.block.segments if s.annotations['sheet_name'] == kwargs['sheet_name']])
       else:
           seg_filtered = set([]) 
    else:
           seg_filtered = set(dsv.block.segments)
           
    ads_filtered= set(filter_query(dsv.analysis_results,**kwargs))
    
    if st_kwargs != {}:
       seg_filtered_st= set(filter_query(seg_st,extra_data_list=dsv.block.segments,**st_kwargs)[1]) 
       ads_filtered_st= set(filter_query(ads_st,extra_data_list=[ads for ads in dsv.analysis_results if ads.stimulus_id != None],**st_kwargs)[1])
    else:
       ads_filtered_st = set(dsv.analysis_results)
       seg_filtered_st = set(dsv.block.segments)
    
    seg = seg_filtered_st & seg_filtered
    ads = ads_filtered_st & ads_filtered
    
    new_dsv.retinal_stimulus = dsv.retinal_stimulus_copy()
    new_dsv.block.segments = list(seg)
    new_dsv.analysis_results = list(ads)
    return new_dsv

class ParamFilterQuery(Query):
    """
    It will restrict the DSV to only recordings and ADS with parameters 
    whose values match the params. 
    
    To restrict parameters of the stimuli to which the ADS or recordings 
    have been done pre-pend 'st_' to the parameter name.
    
    For the recordings parameter sheet refers to the sheet for which
    the recording was done. 
    """
    required_parameters = ParameterSet({
        'params' : ParameterSet,
    })

    def query(self, dsv):
        return param_filter_query(dsv, **self.params)
########################################################################


########################################################################
def tag_based_query(dsv, tags=[]):
        new_dsv = dsv.fromDataStoreView()
        new_dsv.block.segments = dsv.recordings_copy()
        new_dsv.retinal_stimulus = dsv.retinal_stimulus_copy()
        new_dsv.analysis_results = _tag_based_query(dsv.analysis_results, tags)
        return new_dsv


def _tag_based_query(d, tags):
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
    This query filters out all AnalysisDataStructure's corresponding to the
    given tags
    """
    required_parameters = ParameterSet({
        'tags': list,  # list of tags the the query will look for
    })

    def query(self, dsv):
        return tag_based_query(dsv, **self.parameters)
########################################################################

########################################################################
def partition_by_stimulus_paramter_query(dsv, parameter_list):
        st = dsv.get_stimuli()
        values, st = colapse(dsv.block.segments,st,parameter_list=parameter_list,allow_non_identical_objects=True)
        dsvs = []
        for vals in values:
            new_dsv = dsv.fromDataStoreView()
            new_dsv.analysis_results = dsv.analysis_result_copy()
            new_dsv.block.segments.extend(vals)
            dsvs.append(new_dsv)
        return dsvs


class PartitionByStimulusParamterQuery(Query):
    """
    This query will take all recordings and return list of DataStoreViews
    each holding recordings measured to the same stimulus with exception of
    the parameters reference by parameter_list.

    Note that in most cases one wants to do this only against datastore holding
    only single Stimulus type! In that case the datastore is partitioned into
    subsets each holding recordings to the same stimulus with the same paramter
    values, with the exception to the parameters in parameter_list
    """

    required_parameters = ParameterSet({
        'parameter_list': list,  # the index of the parameter against which to partition
    })

    def query(self, dsv):
        return partition_by_stimulus_paramter_query(dsv, **self.parameters)
########################################################################


######################################################################################################################################
def partition_analysis_results_by_parameters_query(dsv, parameter_list):
        values, st = colapse(dsv.analysis_results,dsv.analysis_results,parameter_list=parameter_list,allow_non_identical_objects=True)
        dsvs = []

        for vals in values:
            new_dsv = dsv.fromDataStoreView()
            new_dsv.block.segments = dsv.recordings_copy()
            new_dsv.retinal_stimulus = dsv.retinal_stimulus_copy()
            new_dsv.analysis_results.extend(vals)
            dsvs.append(new_dsv)
        return dsvs

class PartitionAnalysisResultsByParameterNameQuery(Query):
    
    """
    This query will take all analysis results and return list of DataStoreViews
    each holding analysis results that have the same parameters with exception of
    the parameters reference by parameter_list.

    Note that in most cases one wants to do this only against datastore holding
    only single analysis results type! In that case the datastore is partitioned into
    subsets each holding recordings to the same stimulus with the same paramter
    values, with the exception to the parameters in parameter_list.
    """

    required_parameters = ParameterSet({
        'parameter_list': list,  # the index of the parameter against which to partition
    })

    def query(self, dsv):
        return partition_analysis_results_by_parameters_query(dsv,**self.parameters)
######################################################################################################################################


######################################################################################################################################
def partition_analysis_results_by_parameter_values_query(dsv, parameter_list):
        
        if dsv.analysis_results != []:
            p = set(dsv.analysis_results[0].params().keys()) - (set(parameter_list) | set(['name']))
            values, st = colapse(dsv.analysis_results,dsv.analysis_results,parameter_list=p)
            dsvs = []

            for vals in values:
                new_dsv = dsv.fromDataStoreView()
                new_dsv.block.segments = dsv.recordings_copy()
                new_dsv.retinal_stimulus = dsv.retinal_stimulus_copy()
                new_dsv.analysis_results.extend(vals)
                dsvs.append(new_dsv)
            return dsvs
        else:
           return [] 
           
class PartitionAnalysisResultsByParameterValuesQuery(Query):
    
    """
    This query will take all analysis results and return list of DataStoreViews
    each holding analysis results that have the same parameter values with exception of
    the parameters reference by parameter_list.

    This is allowed only on DSVs holding the same AnalysisDataStructures.
    """

    required_parameters = ParameterSet({
        'parameter_list': list,  # the index of the parameter against which to partition
    })

    def query(self, dsv):
        return partition_analysis_results_by_parameter_values_query(dsv,**self.parameters)
######################################################################################################################################









########################################################################
### Not queries, but some helper functions that make it easy to test 
### whether given datastoreview has certain common properties
########################################################################


########################################################################
def equal_stimulus_type(dsv):
    """
    This functions tests whether DSV contains only recordings associated
    with the same stimulus type.
    """
    return matching_parametrized_object_params([MozaikParametrized.idd(s) for s in dsv.get_stimuli()],params=['name'])
########################################################################

########################################################################
def equal_ads_except(dsv, except_params):
    """
    This functions tests whether DSV contains only ADS of the same kind
    and parametrization with the exception of parameters listed in
    except_params.
    """
    return matching_parametrized_object_params(dsv.analysis_results,except_params=except_params)
########################################################################

########################################################################
def ads_with_equal_stimulus_type(dsv, allow_None=False):
    """
    This functions tests whether DSV contains only ADS associated
    with the same stimulus type.

    not_None - if true it will not allow ADS that are not associated with
               stimulus
    """
    if allow_None:
        return matching_parametrized_object_params([MozaikParametrized.idd(ads.stimulus_id) for ads in dsv.analysis_results if ads.stimulus_id != None],params=['name'])
    else:
        if len([0 for ads in dsv.analysis_results if ads.stimulus_id == None]) > 0:
           return False
        return matching_parametrized_object_params([MozaikParametrized.idd(ads.stimulus_id) for ads in dsv.analysis_results],params=['name'])    
########################################################################
