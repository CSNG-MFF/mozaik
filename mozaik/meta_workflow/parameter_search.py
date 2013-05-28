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
    """
    def execute_job(self,run_script,simulator_name,parameters_url,parameters):
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
     
    def execute_job(self,run_script,simulator_name,parameters_url,parameters):
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
         
         subprocess.call(' '.join(["python", run_script, simulator_name, parameters_url]+modified_parameters+['ParameterSearch']),shell=True)


class SlurmSequentialBackend(object):
    """
    This is a back end that runs each simulation run as a slurm job. 
    
    Parameters
    ----------
    num_processes : int
                  If None, job is run without mpi as a single threaded job.
                  Otherwise it specify how many processes to allocate for each simulation run.
    """
    def __init__(self,num_processes=None):
        self.num_processes = num_processes
        
        
    def execute_job(self,run_script,simulator_name,parameters_url,parameters):
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
         
         if self.num_processes:
            subprocess.call(' '.join(["salloc mpirun -np", str(self.num_processes), run_script, simulator_name, parameters_url]+modified_parameters+['ParameterSearch']),shell=True)
         else:
            subprocess.call(' '.join(["salloc", run_script, simulator_name, parameters_url]+modified_parameters+['ParameterSearch']),shell=True) 



class ParameterSearch(object):
    """
    This class defines the interface of parameter search.
    Each ParameterSearch has to implement the function `generate_parameter_combinations`
    and `master_directory_name`.
    
    The parameter search is executed with the function run_parameter_search.
    
    Furthermore each ParameterSearch receives a backend object, that determines how the simulation
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
    
    def generate_parameter_combinations(self):
        """
        Returns a list of dictionaries, each holding the modified parameters as keys, and a combination of their values as the values.
        """
        raise NotImplemented
    
    def master_directory_name(self):
        """
        Returns the name of the master directory which will contain results from the invididual simulation runs.
        """
        raise NotImplemented
        
    def run_parameter_search(self):
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
        if len(sys.argv) == 4:
            run_script = sys.argv[1]
            simulator_name = sys.argv[2]
            parameters_url = sys.argv[3]
        else:
            raise ValueError("Usage: python parameter_search_script simulation_run_script simulator_name root_parameter_file_name")
        
        parameters = MozaikExtendedParameterSet(parameters_url)
        
        mdn = self.master_directory_name()
        os.mkdir(mdn)
        
        for combination in self.generate_parameter_combinations():
            combination['results_dir']='\"\'' + parameters.results_dir + mdn + '\'\"'
            self.backend.execute_job(run_script,simulator_name,parameters_url,combination)


class CombinationParameterSearch(ParameterSearch):
    """
    A ParameterSearch that recevies a list of parameters and list of values for each parameter to test.
    It will then test each of the combination of values.
    
    Parameters
    ----------
    parameter_values : dict
                      Dictionary containing parameter names as keys, and lists as values, each corresponding to the list of values to test for the given parameter.
    """
    def __init__(self,backend,parameter_values):
        CombinationParameterSearch.__init__(backend)
        self.parameter_values = parameter_values
    
    def generate_parameter_combinations(self):
        combs = []
        for combination in parameter_combinations(self.parameter_values.values()):
            combs.append({a : b for (a,b) in zip (self.parameter_values.keys(),combination)})
        return combs    
        
    def master_directory_name(self):
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        return timestamp + "CombinationParamSearch{" + ','.join([str(k) + ':' + str(self.parameter_values[k]) for k in self.parameter_values.keys()]) + '}/'
            
def parameter_combinations(arrays):
    return _parameter_combinations_rec([],arrays)
    
def _parameter_combinations_rec(combination,arrays):
 if arrays == []:
    return [combination]
 else:
    return sum([_parameter_combinations_rec(combination[:] + [value],arrays[1:]) for value in arrays[0]],[])
    
