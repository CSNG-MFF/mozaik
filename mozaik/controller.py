r"""
This is the nexus of workflow execution controll of *mozaik*.
"""

from mozaik.cli import parse_workflow_args
from mozaik.storage.datastore import Hdf5DataStore, PickledDataStore
from mozaik.tools.distribution_parametrization import (
    MozaikExtendedParameterSet,
    load_parameters,
)
from mozaik.tools.misc import result_directory_name
from mozaik.stimuli import EndOfSimulationBlank
from collections import OrderedDict
import sys
import os
import mozaik
import time
from datetime import datetime
import logging
from mozaik.tools.json_export import (
    save_json,
    get_experimental_protocols,
    get_recorders,
    get_stimuli,
)
from parameters import ParameterSet
import copy

logger = mozaik.getMozaikLogger()


class Global:
    r"""global variable container currently only containing the root_directory variable that points to the root directory of the model specification"""

    root_directory = "./"


class FancyFormatter(logging.Formatter):
    r"""
    A log formatter that colours and indents the log message depending on the level.
    """

    DEFAULT_INDENTS = {
        "CRITICAL": "",
        "ERROR": "",
        "WARNING": "",
        "HEADER": "",
        "INFO": "  ",
        "DEBUG": "    ",
    }

    def __init__(self, fmt=None, datefmt=None, mpi_rank=None):
        logging.Formatter.__init__(self, fmt, datefmt)
        self._indents = FancyFormatter.DEFAULT_INDENTS
        if mpi_rank is None:
            self.prefix = ""
        else:
            self.prefix = "%-3d" % mpi_rank

    def format(self, record):
        s = logging.Formatter.format(self, record)
        if record.levelname == "HEADER":
            s = "=== %s ===" % s
        return self.prefix + self._indents[record.levelname] + s


def init_logging(
    filename, file_level=logging.INFO, console_level=logging.WARNING, mpi_rank=None
):
    if mpi_rank is None:
        mpi_fmt = ""
    else:
        mpi_fmt = "%3d " % mpi_rank
    logging.basicConfig(
        level=file_level,
        format="%%(asctime)s %s%%(name)-10s %%(levelname)-6s %%(message)s [%%(pathname)s:%%(lineno)d]"
        % mpi_fmt,
        filename=filename,
        filemode="w",
    )
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(FancyFormatter("%(message)s", mpi_rank=mpi_rank))
    logging.getLogger("").addHandler(console)
    return console


def setup_logging():
    r"""
    This functions sets up logging.
    """
    if mozaik.mpi_comm:
        init_logging(
            Global.root_directory + "log",
            file_level=logging.INFO,
            console_level=logging.INFO,
            mpi_rank=mozaik.mpi_comm.rank,
        )
    else:
        init_logging(
            Global.root_directory + "log",
            file_level=logging.INFO,
            console_level=logging.INFO,
        )


def split_modified_parameters(modified_parameters):
    r"""
    Split modified parameters into model parameters and experiment parameters.

    Parameters
    ----------

    modified_parameters : dict
        Dictionary of parameter overrides parsed from the command line.

    Returns
    -------

    model_modified_parameters : dict
        Dictionary containing only overrides that should be applied to the
        model parameter tree.

    experiment_modified_parameters : dict
        Dictionary containing only overrides that should be applied to named
        experiment definitions.

    Notes
    -----

    Experiment parameters are expected in the form
    ``experiments.<experiment_name>.<parameter_path>`` and are handled
    separately from the model parameter tree.
    """
    experiment_modified_parameters = {}
    model_modified_parameters = {}

    for key, value in modified_parameters.items():
        if key.startswith("experiments."):
            experiment_modified_parameters[key] = value
        else:
            model_modified_parameters[key] = value

    return model_modified_parameters, experiment_modified_parameters


def group_experiment_overrides(experiment_modified_parameters):
    r"""
    Group experiment overrides by experiment name.

    Parameters
    ----------

    experiment_modified_parameters : dict
        Dictionary containing experiment overrides in the form
        ``experiments.<experiment_name>.<parameter_path>``.

    Returns
    -------

    grouped_overrides : OrderedDict
        Ordered dictionary mapping experiment names to dictionaries of
        parameter paths and override values.

    Raises
    ------

    ValueError
        If an override does not contain both an experiment name and a
        parameter path.
    """
    grouped_overrides = OrderedDict()
    for key, value in experiment_modified_parameters.items():
        parts = key.split(".")
        if len(parts) < 3:
            raise ValueError(
                "Experiment override '%s' must have the form "
                "'experiments.<experiment_name>.<parameter_path>'" % key
            )

        experiment_name = parts[1]
        parameter_path = ".".join(parts[2:])
        if not parameter_path:
            raise ValueError("Experiment override '%s' is missing parameter path" % key)

        if experiment_name not in grouped_overrides:
            grouped_overrides[experiment_name] = OrderedDict()
        grouped_overrides[experiment_name][parameter_path] = value

    return grouped_overrides


def instantiate_named_experiments(
    model, experiment_specs, experiment_modified_parameters
):
    r"""
    Instantiate named experiment specifications after applying overrides.

    Parameters
    ----------

    model : Model
        The model on which to execute the experiments.

    experiment_specs : OrderedDict
        Ordered dictionary mapping experiment names to tuples
        ``(experiment_class, parameter_set)`` where ``experiment_class`` is the
        class to instantiate and ``parameter_set`` contains the default
        parameters for that experiment.

    experiment_modified_parameters : dict
        Dictionary containing experiment overrides in the form
        ``experiments.<experiment_name>.<parameter_path>``.

    Returns
    -------

    experiment_list : list
        List of instantiated experiment objects in execution order.

    experiment_parameter_list : list
        List describing the final parameterization of each experiment in a form
        suitable for storing in the datastore.

    Raises
    ------

    ValueError
        If an override references a missing experiment name or cannot be
        applied to the corresponding parameter set.

    TypeError
        If the experiment specification does not follow the expected
        ``(experiment_class, parameter_set)`` tuple format.
    """
    grouped_overrides = group_experiment_overrides(experiment_modified_parameters)
    unknown_experiments = [
        experiment_name
        for experiment_name in grouped_overrides.keys()
        if experiment_name not in experiment_specs
    ]
    if unknown_experiments:
        raise ValueError(
            "Unknown experiment override target(s): %s"
            % ", ".join(sorted(unknown_experiments))
        )

    experiment_list = []
    experiment_parameter_list = []

    for experiment_name, spec in experiment_specs.items():
        if not isinstance(spec, tuple) or len(spec) != 2:
            raise TypeError(
                "OrderedDict-based experiments must contain "
                "(experiment_class, parameter_set) tuples. Invalid spec for '%s'."
                % experiment_name
            )

        experiment_class, default_parameters = spec
        if not isinstance(default_parameters, ParameterSet):
            raise TypeError(
                "Experiment spec '%s' must define a ParameterSet as second tuple item."
                % experiment_name
            )

        parameters = copy.deepcopy(default_parameters)
        overrides = grouped_overrides.get(experiment_name, OrderedDict())
        if overrides:
            try:
                parameters.replace_values(**overrides)
            except Exception as exc:
                raise ValueError(
                    "Failed to apply overrides for experiment '%s': %s"
                    % (experiment_name, exc)
                ) from exc

        experiment = experiment_class(model, parameters)
        experiment_list.append(experiment)
        experiment_parameter_list.append(
            (str(experiment.__class__), str(experiment.parameters))
        )

    return experiment_list, experiment_parameter_list


def prepare_workflow(simulation_name, model_class):
    r"""
    Executes the following preparatory steps for simulation workflow:

    - Load simulation parameters
    - Split model and experiment parameter overrides
    - Initialize random seeds
    - Create directory for results
    - Store loaded parameters
    - Setup logging
    - Store some initial info about the simulation

    Returns
    -------

    sim : module
        NEST module, to use for simulation

    num_threads : int
        Number of threads to use for the simulation

    parameters : dict
        Loaded parameters to initialize the simulation and model with

    experiment_modified_parameters : dict
        Dictionary of experiment overrides separated from the model parameter
        tree. These are applied only if the experiment factory returns named
        experiment specifications.

    """
    (
        simulation_run_name,
        simulator_name,
        num_threads,
        parameters_url,
        modified_parameters,
    ) = parse_workflow_args()

    (
        model_modified_parameters,
        experiment_modified_parameters,
    ) = split_modified_parameters(modified_parameters)

    # First we load the parameters just to retrieve seeds. We will throw them away, because at this stage the PyNNDistribution values were not yet initialized correctly.
    parameters = load_parameters(parameters_url, model_modified_parameters)
    mozaik.setup_seeds(
        model_seed=parameters["model_seed"],
        simulation_seed=parameters["simulation_seed"],
        experiment_seed=parameters["experiment_seed"],
        prevent_reinitialization=True,
    )

    # Now initialize mpi
    print("START MPI")
    mozaik.setup_mpi()

    # Now really load parameters
    print("Loading parameters")
    parameters = load_parameters(parameters_url, model_modified_parameters)
    print("Finished loading parameters")

    import pyNN.nest as sim

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    ddir = result_directory_name(
        simulation_run_name, simulation_name, modified_parameters
    )

    if mozaik.mpi_comm and mozaik.mpi_comm.rank != mozaik.MPI_ROOT:
        Global.root_directory = (
            parameters.results_dir + ddir + "/" + str(mozaik.mpi_comm.rank) + "/"
        )
        mozaik.mpi_comm.barrier()
    else:
        Global.root_directory = parameters.results_dir + ddir + "/"

    os.makedirs(Global.root_directory)
    if mozaik.mpi_comm and mozaik.mpi_comm.rank == mozaik.MPI_ROOT:
        mozaik.mpi_comm.barrier()

    if mozaik.mpi_comm.rank == mozaik.MPI_ROOT:
        # Store simulation run info, if we are the 0 rank process,
        # with several components to be stored/filled in later during the simulation run
        sim_info = {
            "submission_date": None,
            "run_date": datetime.now().strftime("%d/%m/%Y-%H:%M:%S"),
            "simulation_run_name": simulation_run_name,
            "model_name": simulation_name,
            "model_description": model_class.__doc__,
            "results": {"$ref": "results.json"},
            "stimuli": {"$ref": "stimuli.json"},
            "recorders": {"$ref": "recorders.json"},
            "experimental_protocols": {"$ref": "experimental_protocols.json"},
            "parameters": {"$ref": "parameters.json"},
        }
        save_json(sim_info, Global.root_directory + "sim_info.json")
        save_json(parameters.to_dict(), Global.root_directory + "parameters.json")
        save_json(
            modified_parameters, Global.root_directory + "modified_parameters.json"
        )
        recorders = get_recorders(parameters.to_dict())
        save_json(recorders, Global.root_directory + "recorders.json")

    setup_logging()

    return sim, num_threads, parameters, experiment_modified_parameters


def run_workflow(simulation_name, model_class, create_experiments):
    r"""
    This is the main function that executes a workflow.
    It expects it gets the simulation, class of the model, and a function that will create_experiments.
    The create_experiments function gets an instance of a model as its only parameter and it is expected to return
    either a list of Experiment instances that should be executed over the model, or an OrderedDict mapping experiment
    names to tuples ``(experiment_class, parameter_set)``.
    The run workflow will automatically parse the command line to determine the simulator to be used and the path to the root parameter file.
    It will also accept . (point) delimited path to parameteres in the configuration tree, and corresponding values. It will replace each such provided
    parameter's value with the provided one on the command line.

    Parameters
    ----------

    simulation_name : str
        The name of the simulation.

    model_class : class
        The class from which the model instance will be created from.

    create_experiments : func
        The function that returns either the list of experiments that will be
        executed on the model or an OrderedDict of named experiment
        specifications.

    Examples
    --------

    The intended syntax of the commandline is as follows (note that the simulation run name is the last argument):
    >>> python userscript simulator_name num_threads parameter_file_path modified_parameter_path_1 modified_parameter_value_1 ... modified_parameter_path_n modified_parameter_value_n simulation_run_name

    Experiment overrides use the dedicated syntax
    ``experiments.<experiment_name>.<parameter_path>`` and are only accepted
    when ``create_experiments`` returns an OrderedDict of named experiment
    specifications.

    """

    # Prepare workflow - read parameters, setup logging, etc.
    sim, num_threads, parameters, experiment_modified_parameters = prepare_workflow(
        simulation_name, model_class
    )
    # Prepare model to run experiments on
    model = model_class(sim, num_threads, parameters)
    experiment_definition = create_experiments(model)

    if isinstance(experiment_definition, list):
        if experiment_modified_parameters:
            raise ValueError(
                "Experiment overrides were provided, but create_experiments(model) "
                "returned a list. Use an OrderedDict of named experiment specs for "
                "experiment overrides."
            )
        experiment_list = experiment_definition
        experiment_parameter_list = None
    elif isinstance(experiment_definition, OrderedDict):
        experiment_list, experiment_parameter_list = instantiate_named_experiments(
            model,
            experiment_definition,
            experiment_modified_parameters,
        )
    else:
        raise TypeError(
            "create_experiments(model) must return either a list of experiments "
            "or an OrderedDict of named (experiment_class, parameter_set) specs."
        )

    # Run experiments with previously read parameters on the prepared model
    data_store = run_experiments(
        model,
        experiment_list,
        parameters,
        experiment_parameter_list=experiment_parameter_list,
    )

    if mozaik.mpi_comm.rank == mozaik.MPI_ROOT:
        data_store.save()
    import resource

    print(
        "Final memory usage: %iMB"
        % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024))
    )
    return (data_store, model)


def run_experiments(
    model, experiment_list, parameters, load_from=None, experiment_parameter_list=None
):
    r"""
    This is function called by :func:.run_workflow that executes the experiments in the `experiment_list` over the model.
    Alternatively, if load_from is specified it will load an existing simulation from the path specified in load_from.

    Parameters
    ----------

    model : Model
        The model to execute experiments on.

    experiment_list : list
        The list of experiments to execute.

    parameters : ParameterSet
        The parameters given to the simulation run.

    load_from : str
        If not None it will load the simulation from the specified directory.

    experiment_parameter_list : list, optional
        Optional explicit experiment parametrization metadata to store in the
        datastore. If not supplied it will be derived from the instantiated
        experiments in ``experiment_list``.

    Returns
    -------

    data_store : DataStore
        The data store containing the recordings.

    """

    # first lets run all the measurements required by the experiments
    logger.info("Starting Experiemnts")
    if load_from == None:
        data_store = PickledDataStore(
            load=False,
            parameters=MozaikExtendedParameterSet(
                {
                    "root_directory": Global.root_directory,
                    "store_stimuli": parameters.store_stimuli,
                }
            ),
        )
    else:
        data_store = PickledDataStore(
            load=True,
            parameters=MozaikExtendedParameterSet(
                {"root_directory": load_from, "store_stimuli": parameters.store_stimuli}
            ),
        )

    data_store.set_neuron_ids(model.neuron_ids())
    data_store.set_neuron_positions(model.neuron_positions())
    data_store.set_neuron_annotations(model.neuron_annotations())
    data_store.set_model_parameters(parameters.pretty(expand_urls=True))
    data_store.set_sheet_parameters(
        MozaikExtendedParameterSet(model.sheet_parameters()).pretty(expand_urls=True)
    )
    if experiment_parameter_list is None:
        experiment_parameter_list = [
            (str(exp.__class__), str(exp.parameters)) for exp in experiment_list
        ]
    data_store.set_experiment_parametrization_list(experiment_parameter_list)

    t0 = time.time()
    simulation_run_time = 0
    model_exploded = False
    for i, experiment in enumerate(experiment_list):
        logger.info("Starting experiment: " + experiment.__class__.__name__)
        stimuli = experiment.return_stimuli()
        unpresented_stimuli_indexes = data_store.identify_unpresented_stimuli(stimuli)
        logger.info("Running model")
        experiment_run_time, model_exploded = experiment.run(
            data_store, unpresented_stimuli_indexes
        )
        simulation_run_time += experiment_run_time
        if model_exploded:
            logger.info("ERROR: Model exploded, stopping simulation!")
            break
        logger.info("Experiment %d/%d finished" % (i + 1, len(experiment_list)))

    last_blank_run_time = 0
    # Do a reset after the last stimulus. If reset is done as blank stimulus, this makes sure we have some blank recorded also after last stimulus.
    ds = OrderedDict()

    if parameters.null_stimulus_period != 0:
        s = EndOfSimulationBlank(
            trial=0,
            duration=parameters.null_stimulus_period,
            frame_duration=parameters.null_stimulus_period,
        )
        (
            segments,
            null_segments,
            input_stimulus,
            last_blank_run_time,
            _,
        ) = model.present_stimulus_and_record(s, ds)
        data_store.add_recording(segments, s)
        data_store.add_stimulus(input_stimulus, s)
        data_store.add_direct_stimulation(ds, s)
        if null_segments != []:
            data_store.add_null_recording(null_segments, s)
    else:
        last_blank_run_time = 0

    total_run_time = time.time() - t0
    mozaik_run_time = total_run_time - simulation_run_time - last_blank_run_time

    # Adding the state (represented by a randomly generated number) of the rng of every MPI process to the datastore
    if mozaik.mpi_comm:
        rngs_state = mozaik.mpi_comm.gather(
            float(mozaik.simulation_rng.rand(1)), root=0
        )
        log = {"rngs_state": rngs_state, "explosion_detected": model_exploded}
    else:
        log = {"explosion_detected": model_exploded}
    data_store.set_simulation_log(log)

    if not model_exploded and mozaik.mpi_comm.rank == mozaik.MPI_ROOT:
        logger.info("Total simulation run time: %.0fs" % total_run_time)
        logger.info(
            "Simulator run time: %.0fs (%d%%)"
            % (simulation_run_time, int(simulation_run_time / total_run_time * 100))
        )
        logger.info(
            "Mozaik run time: %.0fs (%d%%)"
            % (mozaik_run_time, int(mozaik_run_time / total_run_time * 100))
        )

    experimental_protocols = get_experimental_protocols(data_store)
    stimuli = get_stimuli(data_store, parameters.store_stimuli, parameters.input_space)
    save_json(
        experimental_protocols, Global.root_directory + "experimental_protocols.json"
    )
    save_json(stimuli, Global.root_directory + "stimuli.json")

    return data_store
