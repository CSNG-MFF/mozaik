"""
This module contain query manipulation system that is used to filter information stored in data store (:class:`.DataStore`).

The basic principle is that each query takes a existing :class:`.DataSore` (or :class:`.DataStoreView`) as an input and 
returns and new :class:`.DataSoreView` that is a subset of the input DSV.
"""
from mozaik.core import ParametrizedObject
from parameters import ParameterSet
from mozaik.tools.mozaik_parametrized import colapse,  MozaikParametrized, filter_query, matching_parametrized_object_params
import numpy
import mozaik

logger = mozaik.getMozaikLogger()

class Query(ParametrizedObject):
    """
    Query accepts a :class:`.DataStoreView` and returns a :class:`.DataStoreView` (or set of DSVs) with potentially
    reduced set of recorded data or analysis results

    We recommend to write queries in a way where it can be invoked via the
    ParamterSet method using a class, but also directly as a function with
    (potentially with default paramter values)

    See :class:`.ParamFilterQuery` for a simple example.
    """
    required_parameters = ParameterSet({
    })

    def __init__(self, parameters):
        ParametrizedObject.__init__(self, parameters)

    def query(self, dsv):
        """
        Abstract function to be implemented by each query.
        
        This is the function that executes the query. It receives a DSV as input and has to return a DSV (or set of DSVs).
        """
        raise NotImplementedError


########################################################################
def param_filter_query(dsv,ads_unique=False,rec_unique=False,**kwargs):
    """
    It will return DSV with only recordings and ADSs with mozaik parameters 
    whose values match the parameter values combinations provided in `kwargs`. 
    
    To restrict mozaik parameters of the stimuli associated with the ADS or recordings 
    pre-pend 'st_' to the parameter name.
    
    For the recordings, parameter sheet refers to the sheet for which the recording was done. 
    
    
    Parameters
    ----------
    
    dsv : DataStoreView
        The input DSV.
    
    ads_unique : bool, optional
               If True the query will raise an exception if the query does not identify a unique ADS.

    rec_unique : bool, optional
               If True the query will raise an exception if the query does not identify a unique recording.
    
    \*\*kwargs : dict
               Remaining keyword arguments will be interepreted as the mozaik parameter names and their associated values that all ASDs
               or recordings have to match. The values of the parameters should be either directly the values to match or list of values in which
               case this list is interpreted as *one of* of the values that each returned recording or ASD has to match (thus effectively there
               is an *and* operation between the different parameters and *or* operation between the values specified for the given mozaik parameters). 
               
    Examples
    --------
    >>> datastore.param_filter_query(datastore,identifier=['PerNeuronValue','SingleValue'],sheet_name=sheet,value_name='orientation preference')
    
    This command should return DSV containing all recordings and ADSs whose identifier is *PerNeuronValue* or *SingleValue*, and are associated with sheet named *sheet_name* and as their value name have 'orientation preference'.
    Note that since recordings do not have these parameters, this query would return a DSV containing only ADSs.
    
    >>> datastore.param_filter_query(datastore,st_orientation=0.5)
    
    This command should return DSV containing all recordings and ADSs that are associated with stimuli whose mozaik parameter orientation has value 0.5.
    """
    
    new_dsv = dsv.fromDataStoreView()
    
    st_kwargs = dict([(k[3:],kwargs[k]) for k in kwargs.keys() if k[0:3] == 'st_'])
    kwargs = dict([(k,kwargs[k]) for k in kwargs.keys() if k[0:3] != 'st_'])
    
    seg_st = [MozaikParametrized.idd(seg.annotations['stimulus']) for seg in dsv.block.segments]
    ads_st = [MozaikParametrized.idd(ads.stimulus_id) for ads in dsv.analysis_results if ads.stimulus_id != None]
    if 'sheet_name' in set(kwargs):
       if len(kwargs) == 1:
           # This means that there is only one 'non-stimulus' parameter sheet, and thus we need
           # to filter out all recordings that are associated with that sheet (otherwsie we do not pass any recordings)
           kw = kwargs['sheet_name'] if isinstance(kwargs['sheet_name'],list) else [kwargs['sheet_name']]
           seg_filtered = set([s for s in dsv.block.segments if s.annotations['sheet_name'] in kw])
       else:
           seg_filtered = set([]) 
    else:
        if len(kwargs) == 0:
           seg_filtered = set(dsv.block.segments)
        else:
           seg_filtered = set([])  
        
    ads_filtered= set(filter_query(dsv.analysis_results,**kwargs))
    
    if st_kwargs != {}:
       seg_filtered_st= set(filter_query(seg_st,extra_data_list=dsv.block.segments,**st_kwargs)[1]) 
       ads_filtered_st= set(filter_query(ads_st,extra_data_list=[ads for ads in dsv.analysis_results if ads.stimulus_id != None],**st_kwargs)[1])
    else:
       ads_filtered_st = set(dsv.analysis_results)
       seg_filtered_st = set(dsv.block.segments)
    
    
    seg = seg_filtered_st & seg_filtered
    ads = ads_filtered_st & ads_filtered
    
    new_dsv.sensory_stimulus = dsv.sensory_stimulus_copy()
    new_dsv.block.segments = list(seg)
    new_dsv.analysis_results = list(ads)
    
    if ads_unique and len(ads) != 1:
       raise ValueError("Result was expected to have only single ADS, it contains %d" % len(ads)) 
        
    if rec_unique and len(seg) != 1:
       raise ValueError("Result was expected to have only single Segment, it contains %d" % len(seg)) 
    
    return new_dsv

class ParamFilterQuery(Query):
    """
    See :func:`.param_filter_query` for description.
    
    Other parameters
    ----------------
    
    params : ParameterSet
               The set of mozaik parameters and their associated values to which to restrict the DSV. (see \*\*kwargs in :func:.`param_filter_query`)
    ads_unique : bool, optional
               If True the query will raise an exception if the query does not identify a unique ADS.

    rec_unique : bool, optional
               If True the query will raise an exception if the query does not identify a unique recording.
    """
    
    required_parameters = ParameterSet({
        'params' : ParameterSet,
        'ads_unique' : bool, # It will raise exception if result does not contain a single AnalysisDataStructure
        'rec_unique' : bool, # It will raise exception if result does not contain a single segment (Recording structure)
    })

    def query(self, dsv):
        return param_filter_query(dsv,ads_unique=self.parameters.ads_unique,rec_unique=self.parameters.rec_unique, **self.parameters.params)
########################################################################


########################################################################
def tag_based_query(dsv, tags):
        """
        This query filters out all AnalysisDataStructure's corresponding to the given tags.
        
        Parameters
        ----------
        tags : list(str)
                 The list of tags that each ADS has to contain.
        """
    
        new_dsv = dsv.fromDataStoreView()
        new_dsv.block.segments = dsv.recordings_copy()
        new_dsv.sensory_stimulus = dsv.sensory_stimulus_copy()
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
    See  :func:`.tag_based_query`.
    
    Parameters
    ----------
    tags : list(str)
         The list of tags that each ADS has to contain.
    """
    
    required_parameters = ParameterSet({
        'tags': list,  # list of tags the the query will look for
    })

    def query(self, dsv):
        return tag_based_query(dsv, **self.parameters)
########################################################################

########################################################################
def partition_by_stimulus_paramter_query(dsv, parameter_list):
    """
    This query will take all recordings and return list of DataStoreViews
    each holding recordings measured to the same stimulus with exception of
    the parameters reference by parameter_list.

    Note that in most cases one wants to do this only against datastore holding
    only single Stimulus type! In that case the datastore is partitioned into
    subsets each holding recordings to the same stimulus with the same paramter
    values, with the exception to the parameters in parameter_list.
    
    Parameters
    ----------
    
    dsv : DataStoreView
        The input DSV.
    
    parameter_list : list(string)
               The list of parameters of the associated stimuli that will vary in the returned DSVs, all other stimulus parameters will have the same value within each of the 
               returned DSVs.

    """
    assert 'name' not in parameter_list, "One cannot partition against <name> parameter"
    st = dsv.get_stimuli() + dsv.get_stimuli(null=True)
    values, st = colapse(dsv.get_segments()+dsv.get_segments(null=True),st,parameter_list=parameter_list,allow_non_identical_objects=True)
    dsvs = []
    for vals in values:
        new_dsv = dsv.fromDataStoreView()
        new_dsv.analysis_results = dsv.analysis_result_copy()
        new_dsv.block.segments.extend(vals)
        dsvs.append(new_dsv)
    return dsvs


class PartitionByStimulusParamterQuery(Query):
    """
    See  :func:`.partition_by_stimulus_paramter_query`.
    
    Other parameters
    ----------------
    
    parameter_list : list(string)
               The list of parameters that will vary in the returned DSVs, all other parameters will have the same value within each of the 
               returned DSVs.
    """

    required_parameters = ParameterSet({
        'parameter_list': list,  # the index of the parameter against which to partition
    })

    def query(self, dsv):
        return partition_by_stimulus_paramter_query(dsv, **self.parameters)
########################################################################


######################################################################################################################################
def partition_analysis_results_by_parameters_query(dsv,parameter_list=None,excpt=False):
        """
        This query will take all analysis results and return list of DataStoreViews
        each holding analysis results that have the same values of
        the parameters in parameter_list.

        Note that in most cases one wants to do this only against datastore holding
        only single analysis results type! In that case the datastore is partitioned into
        subsets each holding recordings to the same stimulus with the same paramter
        values, with the exception to the parameters in parameter_list.
        
        Parameters
        ----------
        
        dsv : DataStoreView
            The input DSV.
        
        parameter_list : list(string)
               The list of parameters that will vary in the returned DSVs, all other parameters will have the same value within each of the 
               returned DSVs.

        except : bool
               If excpt is True the query is allowed only on DSVs holding the same AnalysisDataStructures.
        """
        if dsv.analysis_results == []: return []
        assert parameter_list != None , "parameter_list has to be given"
        if excpt:
            assert equal_ads_type(dsv), "If excpt==True you have to provide a dsv containing the same ADS type"
            parameter_list = set(dsv.analysis_results[0].getParams().keys()) - (set(parameter_list) | set(['name']))
            
        values, st = colapse(dsv.analysis_results,dsv.analysis_results,parameter_list=parameter_list,allow_non_identical_objects=True)
        dsvs = []

        for vals in values:
            new_dsv = dsv.fromDataStoreView()
            new_dsv.block.segments = dsv.recordings_copy()
            new_dsv.sensory_stimulus = dsv.sensory_stimulus_copy()
            new_dsv.analysis_results.extend(vals)
            dsvs.append(new_dsv)
        return dsvs

class PartitionAnalysisResultsByParameterNameQuery(Query):
    """
    See  :func:`.partition_analysis_results_by_parameters_query`.
    
    Other parameters
    ----------------
    
    parameter_list : list(string)
               The list of parameters that will vary in the returned DSVs, all other parameters will have the same value within each of the 
               returned DSVs.
    excpt : bool
               If excpt is True the query is allowed only on DSVs holding the same AnalysisDataStructures.
               
    """


    required_parameters = ParameterSet({
        'parameter_list': list,  # the index of the parameter against which to partition
        'excpt' : bool, # will treat the parameter list as except list - i.e. it will partition again all parameter except those in parameter_list
    })

    def query(self, dsv):
        return partition_analysis_results_by_parameters_query(dsv,**self.parameters)
######################################################################################################################################

######################################################################################################################################
def partition_analysis_results_by_stimulus_parameters_query(dsv,parameter_list=None,excpt=False):
        """
        This query will take all analysis results and return list of DataStoreViews
        each holding analysis results that have the same values of
        of stimulus parameters in parameter_list.

        Note that in most cases one wants to do this only against datastore holding
        only analysis results  measured to the same stimulus type! In that case the datastore is partitioned into
        subsets each holding recordings to the same stimulus with the same paramter
        values, with the exception to the parameters in parameter_list.
    
        Parameters
        ----------
        
        dsv : DataStoreView
            The input DSV.
        
        parameter_list : list(string)
               The list of stimulus parameters that will vary between the ASDs in the returned DSVs, all other parameters will have the same value within each of the 
               returned DSVs.

        except : bool
               If excpt is True the query is allowed only on DSVs holding the same AnalysisDataStructures type.
        """
        if dsv.analysis_results == []: return []
            
        for ads in dsv.analysis_results:
            assert ads.stimulus_id != None , "partition_analysis_results_by_stimulus_parameters_query accepts only DSV with ADS that all have defined stimulus id"
            
        st = [MozaikParametrized.idd(ads.stimulus_id) for ads in dsv.analysis_results]
        assert parameter_list != None , "parameter_list has to be given"
        assert type(parameter_list) == list , "parameter_list has to be list"
        
        if excpt:
            assert matching_parametrized_object_params(st,params=['name']), "If excpt==True you have to provide a dsv containing the same ADS type"
            parameter_list = set(st[0].getParams().keys()) - (set(parameter_list) | set(['name']))
        
        
        
        values, st = colapse(dsv.analysis_results,st,parameter_list=parameter_list,allow_non_identical_objects=True)
        dsvs = []

        for vals in values:
            new_dsv = dsv.fromDataStoreView()
            new_dsv.block.segments = dsv.recordings_copy()
            new_dsv.sensory_stimulus = dsv.sensory_stimulus_copy()
            new_dsv.analysis_results.extend(vals)
            dsvs.append(new_dsv)
        return dsvs

class PartitionAnalysisResultsByStimulusParameterQuery(Query):
    """
    See  :func:`.partition_analysis_results_by_stimulus_parameters_query`.
    
    Other parameters
    ----------------
    
    
    parameter_list : list(string)
               The list of parameters that will vary in the returned DSVs, all other parameters will have the same value within each of the 
               returned DSVs.
    excpt : bool
               If excpt is True the query is allowed only on DSVs holding the same AnalysisDataStructures.
    """

    required_parameters = ParameterSet({
        'parameter_list': list,  # the index of the parameter against which to partition
        'excpt' : bool, # will treat the parameter list as except list - i.e. it will partition again all parameter except those in parameter_list
    })

    def query(self, dsv):
        return partition_analysis_results_by_stimulus_parameters_query(dsv,**self.parameters)
######################################################################################################################################



########################################################################
### Not queries, but some helper functions that make it easy to test 
### whether given datastoreview has certain common properties
########################################################################


########################################################################
def equal_stimulus_type(dsv):
    """
    This functions returns True if DSV contains only recordings associated
    with the same stimulus type. Otherwise False.
    """
    return matching_parametrized_object_params([MozaikParametrized.idd(s) for s in dsv.get_stimuli()],params=['name'])
########################################################################

########################################################################
def equal_stimulus(dsv,except_params):
    """
    This functions returns True if DSV contains only recordings associated
    with stimuli of identical parameter values, with the exception of parameters in *except_params*
    """
    return matching_parametrized_object_params([MozaikParametrized.idd(s) for s in dsv.get_stimuli()],except_params=['name']+except_params)
########################################################################


########################################################################
def equal_ads(dsv,params=None,except_params=None):
    """
    This functions returns true if DSV contains only ADS of the same kind
    and with the same values for parameters supplied in *params* or 
    with the exception of parameters listed in *except_params*. 
    Otherwise False.
    """
    return matching_parametrized_object_params(dsv.analysis_results,params=params,except_params=except_params)
########################################################################

########################################################################
def ads_with_equal_stimuli(dsv,params=None,except_params=None):
    """
    This functions returns true if DSV contains only ADS associated with stimuli 
    of the same kind and with the same values for parameters supplied in *params* or 
    with the exception of parameters listed in *except_params*. 
    Otherwise False.
    """
    return matching_parametrized_object_params([MozaikParametrized.idd(ads.stimulus_id) for ads in dsv.analysis_results],params=params,except_params=except_params)
########################################################################


########################################################################
def equal_ads_type(dsv):
    """
    Returns True if the dsv contains ADS of the same type. Otherwise False.
    """
    return matching_parametrized_object_params(dsv.analysis_results,params=['name'])
########################################################################

########################################################################
def ads_with_equal_stimulus_type(dsv, allow_None=False):
    """
    This functions tests whether DSV contains only ADS associated
    with the same stimulus type.
    
    Parameters
    ----------
    not_None : bool
             If true it will not allow ADS that are not associated with stimulus
    """
    if allow_None:
        return matching_parametrized_object_params([MozaikParametrized.idd(ads.stimulus_id) for ads in dsv.analysis_results if ads.stimulus_id != None],params=['name'])
    else:
        if len([0 for ads in dsv.analysis_results if ads.stimulus_id == None]) > 0:
           return False
        return matching_parametrized_object_params([MozaikParametrized.idd(ads.stimulus_id) for ads in dsv.analysis_results],params=['name'])    
########################################################################
