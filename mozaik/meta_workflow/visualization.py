from mozaik.storage.queries import *
import pylab
import math
from scipy.interpolate import griddata
import matplotlib.cm as cm
from analysis import load_fixed_parameter_set_parameter_search
        
def single_value_visualization(simulation_name,master_results_dir,query,value_names=None,filename=None,resolution=None,treat_nan_as_zero=False,ranges={},cols=4):
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
           
    cols : int
         The number of columns in which to show plots, default is 4.
               
    """
    (parameters,datastores,n) = load_fixed_parameter_set_parameter_search(simulation_name,master_results_dir,filter=ParamFilterQuery(ParameterSet({'ads_unique' : False, 'rec_unique' : False, 'params' : ParameterSet({'identifier' : 'SingleValue'})})))

    # Lets first filter out stuff we were asked by user
    datastores = [(a,query.query(b)) for a,b in datastores]
    
    sorted_parameter_indexes = zip(*sorted(enumerate(parameters), key=lambda x: x[1]))[0]
    
    # if value_names is None lets set it to set of value_names in the first datastore
    if value_names == None:
        value_names = [ads.value_name for ads in param_filter_query(datastores[10][1],identifier='SingleValue').get_analysis_result()]
        print value_names
        value_names = set(sorted(value_names))

    # Lets first make sure that the value_names uniqly identify a SingleValue ADS in each DataStore and 
    # that they exist in each DataStore.
    for (param_values,datastore) in datastores:
        for v in value_names:
			assert len(param_filter_query(datastore,identifier='SingleValue',value_name=v).get_analysis_result()) == 1, "Error, %d ADS with value_name %s found for parameter combination: %s" % (len(param_filter_query(datastore,identifier='SingleValue').get_analysis_result()),v, str([str(a) + ':' + str(b) for (a,b) in zip(parameters,param_values)]))
    
    rows = math.ceil(1.0*len(value_names)/cols)
    
    pylab.figure(figsize=(12*cols, 6*rows), dpi=300, facecolor='w', edgecolor='k')
                
    print rows
    print cols
    for i,value_name in enumerate(value_names): 
        pylab.subplot(rows,cols,i+1)
        if len(parameters) == 1:
               x = []
               y = []
               for (param_values,datastore) in datastores: 
                   x.append(param_values[0]) 
                   y.append(float(param_filter_query(datastore,identifier='SingleValue',value_name=value_name).get_analysis_result()[0].value))
               pylab.plot(x,y)
               pylab.plot(x,y,marker='o')
               pylab.xlabel(parameters[sorted_parameter_indexes[0]]) 
               pylab.ylabel(value_name) 
               
        elif len(parameters) == 2:
               x = []
               y = []
               z = []
               for (param_values,datastore) in datastores: 
                   x.append(param_values[sorted_parameter_indexes[0]]) 
                   y.append(param_values[sorted_parameter_indexes[1]]) 
                   z.append(float(param_filter_query(datastore,identifier='SingleValue',value_name=value_name).get_analysis_result()[0].value))
               if treat_nan_as_zero: 
                  z = numpy.nan_to_num(z)
               
               if value_name in ranges:
                  vmin,vmax = ranges[value_name] 
               else:
                  vmin = min(z) 
                  vmax = max(z) 

               if resolution != None:
                   xi = numpy.linspace(numpy.min(x),numpy.max(x),resolution)
                   yi = numpy.linspace(numpy.min(y),numpy.max(y),resolution)
                   gr = griddata((x,y),z,(xi[None, :], yi[:, None]),method='cubic')
                   print gr
                   pylab.imshow(gr,interpolation='none',vmin=vmin,vmax=vmax,aspect='auto',cmap=cm.gray,origin='lower',extent=[numpy.min(x),numpy.max(x),numpy.min(y),numpy.max(y)])
               else:     
                   pylab.scatter(x,y,marker='o',s=300,c=z,cmap=cm.jet,vmin=vmin,vmax=vmax)
                   pylab.xlim(min(x)-0.1*(max(x)-min(x)),max(x)+0.1*(max(x)-min(x)))
                   pylab.ylim(min(y)-0.1*(max(y)-min(y)),max(y)+0.1*(max(y)-min(y)))
                   pylab.colorbar()

                   
               
               pylab.xlabel(parameters[sorted_parameter_indexes[0]]) 
               pylab.ylabel(parameters[sorted_parameter_indexes[1]]) 
        else:
            raise ValueError("Currently cannot handle more than 2D data")
        pylab.title(value_name)    

    if filename != None:
       pylab.savefig(master_results_dir+'/'+filename, bbox_inches='tight')
    
    

def fixed_point_visualization(simulation_name,rate_name,master_results_dir,query,filename=None):
    """
    Visualizes all single values (or those whose names match ones in `value_names` argument)
    present in the datastores of parameter search over a fixed set of parameters. 
    
    Parameters
    ----------
    rate_name : str
              The parameter full name that corresponds to the input rate.  
                
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
    """
    (parameters,datastores,n) = load_fixed_parameter_set_parameter_search(simulation_name,master_results_dir)
    
    assert len(parameters) == 3
    
    sorted_parameter_indexes = zip(*sorted(enumerate(parameters), key=lambda x: x[1]))[0]
    # if value_names isNone lets set it to set of value_names in the first datastore
    value_names = set([ads.value_name for ads in param_filter_query(datastores[0][1],identifier='SingleValue').get_analysis_result()])
   
    # Lets first make sure that the value_names uniqly identify a SingleValue ADS in each DataStore and 
    # that they exist in each DataStore.
    for (param_values,datastore) in datastores:
        dsv = query.query(datastore)
        for v in value_names:
            assert len(param_filter_query(dsv,identifier='SingleValue',value_name=v).get_analysis_result()) == 1, "Error, %d ADS with value_name %s found for parameter combination:" % (len(param_filter_query(datastore,identifier='SingleValue').get_analysis_result()), str([str(a) + ':' + str(b) + ', ' for (a,b) in zip(parameters,param_values)]))

    parameter_indexes = [0,1,2]
    #parameter_indexes = parameter_indexes[]
    values = {}

    #for (param_values,datastore) in datastores: 
        
        
    
    
    
def multi_curve_visualzition(simulation_name,master_results_dir,x_axis_parameter_name,query,filename=None,value_name=None,treat_nan_as_zero=False):
    """
    Parameters
    ----------
    
    x_axis_parameter_name : str
                          The parameter that will be varied along x axis.
                          
    simulation_name : str
                    The name of the simulation.
                    
    master_results_dir : str
                    The directory where the parameter search results are stored.
    query : ParamFilterQuery
          ParamFilterQuery filter query instance that will be applied to each datastore before records are retrieved.
    
    value_name : list(str)
                 The value name to visualize
    file_name : str
              The file name into which to save the resulting figure. If None figure is just displayed.  
               
    """
    (parameters,datastores,n) = load_fixed_parameter_set_parameter_search(simulation_name,master_results_dir)
    
    sorted_parameter_indexes = zip(*sorted(enumerate(parameters), key=lambda x: x[1]))[0]
    print sorted_parameter_indexes

    # if value_names isNone lets set it to set of value_names in the first datastore
    if value_name == None:
        value_name = set([ads.value_name for ads in param_filter_query(datastores[0][1],identifier='SingleValue').get_analysis_result()])
        assert len(value_name) == 1
        value_name = list(value_name)[0]
    
    # Lets first make sure that the value_names uniqly identify a SingleValue ADS in each DataStore and 
    # that they exist in each DataStore.
    for (param_values,datastore) in datastores:
        dsv = query.query(datastore)
        assert len(param_filter_query(dsv,identifier='SingleValue',value_name=value_name).get_analysis_result()) == 1, "Error, %d ADS with value_name %s found for parameter combination:" % (len(param_filter_query(datastore,identifier='SingleValue').get_analysis_result()), str([str(a) + ':' + str(b) + ', ' for (a,b) in zip(parameters,param_values)]))
    
    
    x_axis_parameter_index = parameters.index(x_axis_parameter_name)
    
    pylab.figure(figsize=(24, 12), dpi=2000, facecolor='w', edgecolor='k')
    
    #assert len(parameters) == 3, "We required there to be three changing parameters for this visualization, you provided %d" % (len(parameters))
    
    a = numpy.array([param_values for (param_values,datastore) in datastores])
    a = a[:,[j for j in [0,1,2] if j != x_axis_parameter_index]]
    mmax = numpy.max(a,axis=0)
    mmin = numpy.min(a,axis=0)
    
    d = {}
    for (param_values,datastore) in datastores: 
        key = param_values[:]
        key.pop(x_axis_parameter_index)
        t = d.get(tuple(key),([],[]))
        dsv = query.query(datastore)
        t[0].append(param_values[x_axis_parameter_index])
        t[1].append(float(param_filter_query(dsv,identifier='SingleValue',value_name=value_name).get_analysis_result()[0].value))
        d[tuple(key)] = t
    
    x = []
    y = []
    z = []
    pylab.subplot(2,2,1)
    for k in d.keys():
        color = ((k[0] - mmin[0]) /(mmax[0] - mmin[0]),
                 (k[1] - mmin[1]) /(mmax[1] - mmin[1]),
                 0)
        x.append(k[0])
        y.append(k[1])
        z.append(color)
        pylab.plot(d[k][0],d[k][1])
        pylab.plot(d[k][0],d[k][1],marker='o',color=color)
    
    pylab.xlabel(parameters[sorted_parameter_indexes[0]]) 
    pylab.ylabel(value_name)     
    pylab.ylim(0,20)

    pylab.subplot(2,2,2)        
    pylab.scatter(x,y,marker='o',s=100,c=z)
    parameters.remove(x_axis_parameter_name)
    pylab.xlabel(parameters[0]) 
    pylab.ylabel(parameters[1])     
    
    
    # let's add ratio colored plot
    x = []
    y = []
    z = []

    pylab.subplot(2,2,3)
    for k in d.keys():
        a = (k[0] - mmin[0]) /(mmax[0] - mmin[0])
        b = (k[1] - mmin[1]) /(mmax[1] - mmin[1])
        color = [a,
                 b,
                 0]
        if numpy.sqrt(numpy.sum(numpy.power(color,2))) != 0:
            color = color / numpy.sqrt(numpy.sum(numpy.power(color,2)))
            
        color[2] = numpy.sqrt(a*a+b*b)/numpy.sqrt(2)
            
        x.append(k[0])
        y.append(k[1])
        z.append(color)
        pylab.plot(d[k][0],d[k][1])
        pylab.plot(d[k][0],d[k][1],color=color)
    
    pylab.xlabel(parameters[sorted_parameter_indexes[0]]) 
    pylab.ylabel(value_name)     


    pylab.subplot(2,2,4)        
    pylab.scatter(x,y,marker='o',s=100,c=z)
    pylab.xlabel(parameters[0]) 
    pylab.ylabel(parameters[1])         
    

    if filename != None:
       pylab.savefig(master_results_dir+'/'+filename)
    
    
    
    
    
    
    
    
