import sys
import subprocess
from mozaik.tools.distribution_parametrization import MozaikExtendedParameterSet
from datetime import datetime
import os

class ParameterSearchBackend(object):
    """
    This is the parameter search backend interface. The :func:.`execute_job`
    implements the execution of the job, using the information given to the 
    constructor, and the dictionary of modified parameters given in its arguments.
    
    Parameters
    ----------
    run_script : str
               The name of the script corresponding to the simulation
    
    simulator_name : str
                   The name of the simulator to use
    
    parameters_url : str
                   The path of the root simulation parameter file.
    """
     
     def __init__(self,run_script,simulator_name,parameters_url):
        self.run_script = run_script
        self.simulator_name = simulator_name
        self.parameters_url = parameters_url
     
     def execute_job(parameters):
         """
         This function recevies the list of parameters to modify and their values, and has to 
         execute the corresponding mozaik simulation.
         
         Parameters
         ----------
         parameters : dict
                    The dictionary holding the names of parameters to be modified as keys, and the values to set them to as the corresponding values. 
         """
         raise NotImplemented



class LocalSequentialBackend(object):
    """
    This is the simplest backend that simply executes the simulation on the present 
    machine sequentially (i.e. it waits for the simulation to end before starting new one).
    """
     
     
     def execute_job(parameters):
         """
         This function recevies the list of parameters to modify and their values, and has to 
         execute the corresponding mozaik simulation.
         
         Parameters
         ----------
         parameters : dict
                    The dictionary holding the names of parameters to be modified as keys, and the values to set them to as the corresponding values. 
         """
         modified_parameters = []
         for k in parameters.keys():
             modified_parameters.append(k)
             modified_parameters.append(str(parameters[k]))
         
         subprocess.call(' '.join(["python", run_script, simulator_name, parameters_url]+modified_parameters),shell=True)


class ParameterSearch(object):
    """
    This class defines the interface of parameter search.
    Each ParameterSearch has to implement the function `generate_parameter_combinations`.
    
    The parameter search is executed with the function run_parameter_search.
    
    Furthermore each ParameterSearch recieves a backend object, that determines how the simulation
    with a given parameter combination is executed. This allows for user to define executaion
    mechanisms using various cluster scheaduling architectures. See :class:.`ParameterSearchBackend`
    for more details.
     
    Parameters
    ----------
    params : ParameterSearchBackend
           The job execution backend to use. 
    """
    
    def __init__(self,backend):
        self.backend = backend
        
    def run_parameter_search():
        """
        This method will run the parameter search replacing each combination of values defined by dictionary params
        in the default parametrization and runing the simulation with each such modified parameters,
        storing the results of each simulation run in a subdirectory named based on the given modified parameter names and their
        values.
        
        It will read the command line for the name of the script that runs individual simulations, the simulator name and the root parameter file path
        Command line syntax:
        
        python parameter_search_script simulation_run_script simulator_name root_parameter_file_name
        """
        
        # Read parameters
        #exec("import pyNN.nest as sim" )
        
        if len(sys.argv) == 4:
            run_script = sys.argv[1]
            simulator_name = sys.argv[2]
            parameters_url = sys.argv[3]
        else:
            raise ValueError("Usage: python parameter_search_script simulation_run_script simulator_name root_parameter_file_name")
        
        parameters = MozaikExtendedParameterSet(parameters_url)
        
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        master_directory = timestamp + "ParamSearch{" + ','.join([str(k) + ':' + str(params[k]) for k in params.keys()]) + '}/'
        os.mkdir(master_directory)
        
        for combination in parameter_combinations(params.values()):
            modified_parameters = []
            for v,p in zip(combination,params.keys()):
                modified_parameters.append(p)
                modified_parameters.append(str(v))
            modified_parameters.append('results_dir')
            modified_parameters.append('\"\'' + parameters.results_dir + master_directory + '\'\"')
            


class GridParameterSearch(ParameterSearch):

    
def parameter_combinations(arrays):
    return _parameter_combinations_rec([],arrays)
    
def _parameter_combinations_rec(combination,arrays):
 if arrays == []:
    return [combination]
 else:
    return sum([_parameter_combinations_rec(combination[:] + [value],arrays[1:]) for value in arrays[0]],[])
    
