import 

def run_parameter_search(**params):
    """
    This method will run parameter search replacing each combination of values defined by **params
    in the default parametrization and runing the simulation with each such modified parameters,
    storing the results of each simulation run in a subdirectory named based on the given modified parameter names and their
    values.
    """
    
    # Read parameters
    #exec("import pyNN.nest as sim" )
    
    if len(sys.argv) == 3:
        simulator_name = sys.argv[1]
        parameters_url = sys.argv[2]
    else:
        raise ValueError("Usage: runscript simulator_name")
    for combination in parameter_combinations(params.values()):
        modified_parameters = []
        for p,v in zip(combination,param.keys())
            modified_parameters.append(p)
            modified_parameters.append(v)
        subprocess.call(["python", simulator_name, parameters_url]+modified _parameters)
    
def parameter_combinations(arrays):
    return _parameter_combinations_rec([],arrays)
    
def _parameter_combinations_rec(combination,arrays):
 if arrays == []:
    return [combination]
 else:
    return [_parameter_combinations_rec(combination.copy() + value,arrays[1:]) for value in arrays[1]]
    
