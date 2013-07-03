import pickle
from mozaik.framework.experiment_controller import result_directory_name
from mozaik.storage.datastore import PickledDataStore
from mozaik.storage.queries import *
import pylab
from scipy.interpolate import griddata
import matplotlib.cm as cm

def load_fixed_parameter_set_parameter_search(simulation_name,master_results_dir):
    """
    Loads all datastores of parameter search over a fixed set of parameters. 
    
    Parameters
    ----------
    simulation_name : str
                    The name of the simulation.
    master_results_dir : str
                       The directory where the parameter search results are stored.
    
    Returns
    -------
    A tuple (parameters,datastores), where `parameters` is a list of parameters over which the parameter search was performed.
    The dsvs is a list of tuples (values,datastore) where `values` is a list of values (in the order as im `parameters`) of the
    parameters, and dsv is a DataStore with results recorded to the combination of parameter values.
    """
    f = open(master_results_dir+'/parameter_combinations','rb')
    combinations = pickle.load(f)
    f.close()
    
    # first check whether all parameter combinations contain the same parameter names
    assert len(set([tuple(set(comb.keys())) for comb in combinations])) == 1 , "The parameter search didn't occur over a fixed set of parameters"
    
    parameters = combinations[0].keys()
    
    datastore = []
    
    for i,combination in enumerate(combinations):
        print i
        rdn = result_directory_name('ParameterSearch',simulation_name,combination)
        try:
            data_store = PickledDataStore(load=True,parameters=ParameterSet({'root_directory': master_results_dir + '/' + rdn}),replace=False)
            datastore.append(([combination[k] for k in parameters],data_store))
        except IOError:
            print "Error loading datastore: " + rdn
        
    return (parameters,datastore)
        
    
    

def single_value_visualization(simulation_name,master_results_dir,query,value_names=None,filename=None,resolution=None,treat_nan_as_zero=False,ranges={}):
    """
    Visualizes all single values (or those whose names match ones in `value_names` argument)
    present in the datastores of parameter search over a fixed set of parameters. 
    
    Parameters
    ----------
    simulation_name : str
                    The name of the simulation.
    master_results_dir : str
                    The directory where the parameter search results are stored.
    query : ParamFilterQuery
          ParamFilterQuery filter query instance that will be applied to each datastore before records are retrieved.
    
    value_names : list(str)
                  List of value names to visualize.  
    file_name : str
              The file name into which to save the resulting figure. If None figure is just displayed.  
    resolution : int
               If not None data will be plotted on a interpolated grid of size (resolution,...,resolution)
    ranges : dict
           A dictionary with value names as keys, and tuples of (min,max) ranges as values indicating what range of values should be displayed.
               
    """
    (parameters,datastores) = load_fixed_parameter_set_parameter_search(simulation_name,master_results_dir)
    # if value_names isNone lets set it to set of value_names in the first datastore
    if value_names == None:
        value_names = set([ads.value_name for ads in param_filter_query(datastores[0][1],identifier='SingleValue').get_analysis_result()])
    
    # Lets first make sure that the value_names uniqly identify a SingleValue ADS in each DataStore and 
    # that they exist in each DataStore.
    for (param_values,datastore) in datastores:
        dsv = query.query(datastore)
        for v in value_names:
            assert len(param_filter_query(dsv,identifier='SingleValue',value_name=v).get_analysis_result()) == 1, "Error, %d ADS with value_name %s found for parameter combination:" % (len(param_filter_query(datastore,identifier='SingleValue').get_analysis_result()), str([str(a) + ':' + str(b) + ', ' for (a,b) in zip(parameters,param_values)]))
    
    pylab.figure(figsize=(12*len(value_names), 6), dpi=2000, facecolor='w', edgecolor='k')
    for i,value_name in enumerate(value_names): 
        pylab.subplot(1,len(value_names),i+1)
        
        if len(parameters) == 1:
               x = []
               y = []
               for (param_values,datastore) in datastores: 
                   dsv = query.query(datastore)
                   x.append(param_values[0]) 
                   z.append(float(param_filter_query(dsv,identifier='SingleValue',value_name=value_name).get_analysis_result()[0].value))
               pylab.plot(x,y,marker='o-')
               
        if len(parameters) == 2:
               x = []
               y = []
               z = []
               for (param_values,datastore) in datastores: 
                   dsv = query.query(datastore)
                   x.append(param_values[1]) 
                   y.append(param_values[0]) 
                   z.append(float(param_filter_query(dsv,identifier='SingleValue',value_name=value_name).get_analysis_result()[0].value))
               if treat_nan_as_zero: 
                  z = numpy.nan_to_num(z)
               print value_name
               print numpy.max(z) 
               
               if value_name in ranges:
                  vmin,vmax = ranges[value_name] 
               else:
                  vmin = min(z) 
                  vmax = max(z) 
               if resolution != None:
                   xi = numpy.linspace(numpy.min(x),numpy.max(x),resolution)
                   yi = numpy.linspace(numpy.min(y),numpy.max(y),resolution)
                   #gr = griddata((x,y),z,(xi[None, :], yi[:, None]),method='cubic')
                   #pylab.imshow(gr,interpolation='none',vmin=vmin,vmax=vmax,aspect='auto',cmap=cm.gray,origin='lower',extent=[numpy.min(x),numpy.max(x),numpy.min(y),numpy.max(y)])
                   #pylab.hold('on')
                   pylab.scatter(x,y,marker='o',s=50,c=z,cmap=cm.jet,vmin=vmin,vmax=vmax)
                   pylab.colorbar()
               else:     
                   pylab.scatter(x,y,marker='o',s=300,c=z,cmap=cm.jet,vmin=vmin,vmax=vmax)
                   pylab.colorbar()
               
               pylab.xlabel(parameters[1]) 
               pylab.ylabel(parameters[0]) 
        else:
            raise ValueError("Currently cannot handle more than 2D data")
        pylab.title(value_name)    

    if filename != None:
       pylab.savefig(master_results_dir+'/'+filename)
    
    
    
    
    
    
    
    


