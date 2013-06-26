import sys
import subprocess
import pickle
from datetime import datetime
import os
import time
import re

class ParameterSearchBackend(object):
    """
    This is the parameter search backend interface. The :func:.`execute_job`
    implements the execution of the job, using the information given to the 
    constructor, and the dictionary of modified parameters given in its arguments.
    """
    def execute_job(self,run_script,simulator_name,parameters_url,parameters,simulation_run_name):
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
     
    def execute_job(self,run_script,simulator_name,parameters_url,parameters,simulation_run_name):
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
    num_threads : int
                  Number of threads per mpi process.

    num_mpi : int
                  Number of mpi processes to spawn per job.
                  
    slurm_options : list(string), optional 
                  List of strings that will be passed to slurm sbatch command as options.  
    Note:
    -----
    The most common usage of slurm_options is the let slurm know how many mpi processed to spawn per job, and how to allocates resources to them.
    """
    def __init__(self,num_threads,num_mpi,slurm_options=None):
        self.num_threads = num_threads
        self.num_mpi = num_mpi
        if slurm_options==None:
           self.slurm_options=[]
        else:
           self.slurm_options=slurm_options 
        
        
        
        
    def execute_job(self,run_script,simulator_name,parameters_url,parameters,simulation_run_name):
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
        
     
         from subprocess import Popen, PIPE, STDOUT
        
         p = Popen(['sbatch'] + self.slurm_options +  ['-o',parameters['results_dir'][2:-2]+"/slurm-%j.out"],stdin=PIPE,stdout=PIPE,stderr=PIPE)

         data = '\n'.join([
                            '#!/bin/bash',
                            '#SBATCH -J MozaikParamSearch',
                            '#SBATCH -n ' + str(self.num_mpi),
                            '#SBATCH -c ' + str(self.num_threads),
                            'source  /opt/software/mpi/openmpi-1.6.3-gcc/env',
                            'source /home/antolikjan/env/mozaik-pynn0.8/bin/activate',
                            'cd ' + os.getcwd(),
                            ' '.join(["mpirun"," --mca mtl ^psm python",run_script, simulator_name, str(self.num_threads) ,parameters_url]+modified_parameters+[simulation_run_name]+['>']  + [parameters['results_dir'][1:-1] +'/OUTFILE'+str(time.time())]),
                        ]) 
         print p.communicate(input=data)[0]                  
         print data
         p.stdin.close()



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
           
           
    Examples
    --------
    The commandline usage should be:
    
    >>> parameter_search_script simulation_run_script simulator_name path_to_root_parameter_file
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
        
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        mdn = timestamp + "[" + parameters_url.replace('/','.') + "]" +  self.master_directory_name()
        os.mkdir(mdn)
        
        counter=0
        combinations = self.generate_parameter_combinations()
        
        f = open(mdn + '/parameter_combinations','wb')
        pickle.dump(combinations,f)
        f.close()
        
        for combination in combinations:
            combination['results_dir']='\"\'' + os.getcwd() + '/' + mdn + '/\'\"'
            self.backend.execute_job(run_script,simulator_name,parameters_url,combination,'ParameterSearch')
            counter = counter + 1
            
        print ("Submitted %d jobs." % counter)


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
        ParameterSearch.__init__(self,backend)
        self.parameter_values = parameter_values
    
    def generate_parameter_combinations(self):
        combs = []
        for combination in parameter_combinations(self.parameter_values.values()):
            combs.append({a : b for (a,b) in zip (self.parameter_values.keys(),combination)})
        return combs    
        
    def master_directory_name(self):
        return "CombinationParamSearch{" + ','.join([str(k) + ':' + (str(self.parameter_values[k]) if len(self.parameter_values[k]) < 5 else str(len(self.parameter_values[k]))) for k in self.parameter_values.keys()]) + '}/'
            
def parameter_combinations(arrays):
    return _parameter_combinations_rec([],arrays)
    
def _parameter_combinations_rec(combination,arrays):
 if arrays == []:
    return [combination]
 else:
    return sum([_parameter_combinations_rec(combination[:] + [value],arrays[1:]) for value in arrays[0]],[])
    
