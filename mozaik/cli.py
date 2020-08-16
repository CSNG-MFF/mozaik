import sys


def parse_parameter_search_args():
    if len(sys.argv) == 4:
        run_script = sys.argv[1]
        simulator_name = sys.argv[2]
        parameters_url = sys.argv[3]
        return run_script, simulator_name, parameters_url
    else:
        raise ValueError(
            "Usage: python parameter_search_script simulation_run_script simulator_name root_parameter_file_name"
        )


def parse_workflow_args():
    if len(sys.argv) > 4 and len(sys.argv) % 2 == 1:
        simulation_run_name = sys.argv[-1]
        simulator_name = sys.argv[1]
        num_threads = sys.argv[2]
        parameters_url = sys.argv[3]
        modified_parameters = {
            sys.argv[i * 2 + 4]: eval(sys.argv[i * 2 + 5])
            for i in range(0, (len(sys.argv) - 5) // 2)
        }
        return (
            simulation_run_name,
            simulator_name,
            num_threads,
            parameters_url,
            modified_parameters,
        )
    else:
        raise ValueError(
            "Usage: runscript simulator_name num_threads parameter_file_path modified_parameter_path_1 modified_parameter_value_1 ... modified_parameter_path_n modified_parameter_value_n simulation_run_name"
        )
