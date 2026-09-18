import sys
import subprocess
import pickle
from datetime import datetime
import os
import time
import re
import shlex
from mozaik.cli import parse_parameter_search_args
from mozaik.tools.misc import result_directory_name
import json
from mozaik.tools.json_export import save_json


def _shell_export_lines(env):
    r"""
    Render *env* as shell ``export`` statements for a job script.

    Sorted, so that the same environment always produces the same script text, and quoted,
    so that a value containing whitespace or shell metacharacters cannot break the script.
    """
    return [
        "export %s=%s" % (k, shlex.quote(str(v)))
        for k, v in sorted((env or {}).items())
    ]


def _announce_dry_run(argv, script=None):
    r"""
    Print what would have been submitted, and return a placeholder job id.

    The id is a placeholder rather than None so that a dry run exercises the same code path as
    a real one -- dependencies between waves still render instead of being skipped as
    unavailable.
    """
    _announce_dry_run.counter += 1
    print("[dry-run] would submit: %s" % " ".join(str(t) for t in argv))
    if script is not None:
        print("[dry-run] job script:")
        for line in script.splitlines():
            print("    " + line)
    return "DRY%d" % _announce_dry_run.counter


_announce_dry_run.counter = 0


def _parse_slurm_job_id(sbatch_output):
    r"""
    Read the job id out of the output of ``sbatch --parsable``, which is the bare id,
    or ``<id>;<cluster>`` when a cluster name is reported.

    Returns None when what came back is not an id -- a failed submission, for instance.
    Callers must treat a None id as "no dependency available" rather than building a
    dependency on it.
    """
    job_id = (sbatch_output or "").strip().split(";")[0]
    return job_id if job_id.isdigit() else None


def _merged_env(env):
    r"""
    Merge *env* on top of the current process environment, for :mod:`subprocess`.

    Returns None when there is nothing to add, so that the child simply inherits the parent
    environment (the historical behaviour).
    """
    if not env:
        return None
    merged = dict(os.environ)
    merged.update({k: str(v) for k, v in env.items()})
    return merged


class ParameterSearchBackend(object):
    r"""
    This is the parameter search backend interface. The :func:.`execute_job`
    implements the execution of the job, using the information given to the
    constructor, and the dictionary of modified parameters given in its arguments.

    A backend may additionally implement :func:.`execute_script`, which submits an arbitrary
    command rather than a mozaik simulation. It is required only for workflows that schedule
    a second wave of jobs (for instance an export pass running after the simulations).
    """

    def execute_job(
        self,
        run_script,
        simulator_name,
        parameters_url,
        parameters,
        simulation_run_name,
        env=None,
    ):
        """
        This function recevies the list of parameters to modify and their values, and has to
        execute the corresponding mozaik simulation.

        Parameters
        ----------

        parameters : dict
            The dictionary holding the names of parameters to be modified as keys, and the values to set them to as the corresponding values.

        env : dict, optional
            Environment variables to set for the job, on top of the inherited environment.
            Used for per-job configuration that cannot be expressed as a mozaik parameter.
            None or empty means the job simply inherits the environment.

        Returns
        -------
        The scheduler's identifier for the submitted job, or None if the backend has no such
        concept (for instance one that runs the simulation in the foreground). The identifier
        is what :func:.`execute_script` accepts in its ``depends_on`` argument.
        """
        raise NotImplemented

    def execute_script(self, command, env=None, depends_on=None, output=None):
        """
        Execute an arbitrary command (rather than a mozaik simulation).

        Parameters
        ----------

        command : list
            The command and its arguments, as a list of tokens.

        env : dict, optional
            Environment variables to set, on top of the inherited environment.

        depends_on : list, optional
            Job identifiers (as returned by :func:.`execute_job`) this command must not start
            before. Backends that execute in the foreground ignore it, having already waited.

        output : str, optional
            Where the scheduler should write the job's console output; a scheduler-specific
            pattern (for slurm, ``%j`` expands to the job id). Backends that execute in the
            foreground ignore it, the output having gone to their own stdout.

        Returns
        -------
        The scheduler's identifier for the submitted job, or None.
        """
        raise NotImplemented


class LocalSequentialBackend(object):
    r"""
    This is the simplest backend that simply executes the simulation on the present
    machine sequentially (i.e. it waits for the simulation to end before starting new one).

    Parameters
    ----------

    dry_run : bool, optional
        Print the commands that would be run instead of running them.
    """

    def __init__(self, dry_run=False):
        self.dry_run = dry_run

    def execute_job(
        self,
        run_script,
        simulator_name,
        parameters_url,
        parameters,
        simulation_run_name,
        env=None,
    ):
        r"""
        This function recevies the list of parameters to modify and their values, and has to
        execute the corresponding mozaik simulation.

        Parameters
        ----------

        parameters : dict
            The dictionary holding the names of parameters to be modified as keys, and the values to set them to as the corresponding values.

        env : dict, optional
            Environment variables to set for the simulation, on top of the inherited environment.

        Returns
        -------
        None -- this backend runs the simulation in the foreground, so there is no job
        identifier to depend on.
        """
        modified_parameters = []
        for k in parameters.keys():
            modified_parameters.append(k)
            modified_parameters.append(repr(parameters[k]))

        command = (
            ["python", run_script, simulator_name, "1", parameters_url]
            + modified_parameters
            + [simulation_run_name]
        )
        if self.dry_run:
            return _announce_dry_run(command)
        subprocess.call(
            " ".join(shlex.quote(str(token)) for token in command),
            shell=True,
            env=_merged_env(env),
        )
        return None

    def execute_script(self, command, env=None, depends_on=None, output=None):
        r"""
        Run *command* in the foreground.

        ``depends_on`` is ignored: this backend has already waited for every job it started.
        ``output`` is ignored: the command writes to this process's stdout.
        """
        if self.dry_run:
            return _announce_dry_run(command)
        subprocess.call(
            " ".join(shlex.quote(str(token)) for token in command),
            shell=True,
            env=_merged_env(env),
        )
        return None


class SlurmSequentialBackend(object):
    r"""
    This is a back end that runs each simulation run as a slurm job.

    Parameters
    ----------

    num_threads : int
        Number of threads per mpi process.

    num_mpi : int
        Number of mpi processes to spawn per job.

    path_to_mozaik_env : string
        Path to virtual environment in which mozaik is installed.

    slurm_options : list(string), optional
        List of strings that will be passed to slurm sbatch command as options.

    Notes
    -----
    The most common usage of slurm_options is to let slurm know how many mpi processed to spawn per job, and how to allocates resources to them.

    """

    def __init__(
        self,
        num_threads,
        num_mpi,
        path_to_mozaik_env,
        slurm_options=None,
        dry_run=False,
    ):
        self.num_threads = num_threads
        self.num_mpi = num_mpi
        self.path_to_mozaik_env = path_to_mozaik_env
        self.dry_run = dry_run
        if slurm_options == None:
            self.slurm_options = []
        else:
            self.slurm_options = slurm_options

    def execute_job(
        self,
        run_script,
        simulator_name,
        parameters_url,
        parameters,
        simulation_run_name,
        env=None,
    ):
        r"""
        This function recevies the list of parameters to modify and their values, and has to
        execute the corresponding mozaik simulation.

        Parameters
        ----------

        parameters : dict
            The dictionary holding the names of parameters to be modified as keys, and the values to set them to as the corresponding values.

        env : dict, optional
            Environment variables to set for the job, exported at the top of the job script.
            ``srun`` propagates the job environment to its tasks, so they reach the simulation.

        Returns
        -------
        The slurm job id as a string, or None if it could not be read back from ``sbatch``.
        """
        modified_parameters = []
        for k in parameters.keys():
            modified_parameters.append(k)
            modified_parameters.append(repr(parameters[k]))

        from subprocess import Popen, PIPE, STDOUT

        # use sbatch to queue job with params as in  slurm options (except job-geometry).
        # --parsable makes sbatch answer with the bare job id, so it can be read back.
        argv = (
            ["sbatch", "--parsable"]
            + self.slurm_options
            + ["-o", parameters["results_dir"] + "/slurm-%j.out"]
        )

        # pass jobfile: sets slurm job geometry, sources env and starts simulation job from cwd
        command = (
            [
                "srun",
                "--mpi=pmix_v5",
                "python",
                run_script,
                simulator_name,
                str(self.num_threads),
                parameters_url,
            ]
            + modified_parameters
            + [simulation_run_name]
        )
        data = "\n".join(
            [
                "#!/bin/bash",
                "#SBATCH -n " + str(self.num_mpi),
                "#SBATCH -c " + str(self.num_threads),
            ]
            + _shell_export_lines(env)
            + [
                "source " + str(self.path_to_mozaik_env),
                "cd " + os.getcwd(),
                " ".join(shlex.quote(str(token)) for token in command)
                + " > "
                + shlex.quote(
                    parameters["results_dir"] + "/OUTFILE" + str(time.time())
                ),
            ]
        )
        if self.dry_run:
            return _announce_dry_run(argv, data)

        p = Popen(argv, stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True)
        out = p.communicate(input=data)[0]
        print(out)
        print(data)
        p.stdin.close()
        return _parse_slurm_job_id(out)

    def execute_script(self, command, env=None, depends_on=None, output=None):
        r"""
        Queue *command* as a single-task slurm job.

        Parameters
        ----------

        command : list
            The command and its arguments, as a list of tokens.

        env : dict, optional
            Environment variables to set, exported at the top of the job script.

        depends_on : list, optional
            Job ids this command must not start before; entries that are None (a backend
            without job ids, or an id that could not be read back) are dropped, and if
            nothing is left the job is queued without a dependency.

        output : str, optional
            Passed to sbatch as ``-o``; ``%j`` expands to the job id. Without it the job's
            console output lands in slurm's default, ``slurm-%j.out`` in the submission
            directory, which is rarely where the rest of the run's output is.

        Returns
        -------
        The slurm job id as a string, or None if it could not be read back from ``sbatch``.
        """
        options = list(self.slurm_options)
        dependencies = [str(j) for j in (depends_on or []) if j is not None]
        if dependencies:
            options.append("--dependency=afterok:" + ":".join(dependencies))
        if output is not None:
            options += ["-o", output]

        from subprocess import Popen, PIPE

        # --parsable makes sbatch answer with the bare job id, so it can be read back.
        argv = ["sbatch", "--parsable"] + options

        data = "\n".join(
            [
                "#!/bin/bash",
                "#SBATCH -n 1",
                "#SBATCH -c " + str(self.num_threads),
            ]
            + _shell_export_lines(env)
            + [
                "source " + str(self.path_to_mozaik_env),
                "cd " + os.getcwd(),
                " ".join(shlex.quote(str(token)) for token in command),
            ]
        )
        if self.dry_run:
            return _announce_dry_run(argv, data)

        p = Popen(argv, stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True)
        out = p.communicate(input=data)[0]
        print(out)
        print(data)
        p.stdin.close()
        return _parse_slurm_job_id(out)


class ParameterSearch(object):
    r"""
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

    def __init__(self, backend):
        self.backend = backend

    def generate_parameter_combinations(self):
        r"""
        Returns a list of dictionaries, each holding the modified parameters as keys, and a combination of their values as the values.
        """
        raise NotImplemented

    def master_directory_name(self):
        r"""
        Returns the name of the master directory which will contain results from the invididual simulation runs.
        """
        raise NotImplemented

    def master_directory(self, parameters_url):
        r"""
        Returns the directory this search's results are written to.

        A fresh timestamped directory per invocation by default, so that runs never collide.
        Override it to return a fixed path when successive invocations are meant to *build on
        each other* -- submitting part of a search now and the rest later -- since a later
        invocation can only see the earlier one's results if they share a directory.

        Parameters
        ----------

        parameters_url : str
            The root parameter file of the search, named in the default directory name.
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return (
            timestamp
            + "["
            + parameters_url.replace("/", ".")
            + "]"
            + self.master_directory_name()
        )

    def prepare(self, master_directory):
        r"""
        Hook executed once in the launcher process, after the master directory has been
        created and before any parameter combination is generated or job submitted.

        Does nothing by default. Override it to do work the whole search depends on and that
        must happen exactly once -- writing an input the jobs will read, for instance -- so
        that it does not have to be run by hand beforehand, and cannot be raced by the jobs.

        The Experanto workflow overrides it to build its chunk lists (see
        :mod:`mozaik.tools.experanto_chunks`), which decide how many jobs the search fans out
        into and are read again when the simulations are exported.

        Parameters
        ----------

        master_directory : str
            The freshly created master directory, relative to the current working directory.
        """
        pass

    def job_environment(self, combination):
        r"""
        Returns the environment variables to set for the job running *combination*.

        Empty by default. Override it for per-job configuration that cannot be expressed as a
        mozaik parameter -- mozaik rejects parameters a model does not declare, so a value the
        experiments need but the model does not know about has to travel out of band.

        The Experanto workflow overrides it to tell each job which chunk of the stimulus set
        to present: the model knows nothing about chunks, so they cannot be model parameters.

        Parameters
        ----------

        combination : dict
            The modified parameters of the job, as returned by
            :func:.`generate_parameter_combinations` (plus ``results_dir``).
        """
        return {}

    def simulation_run_name(self, combination):
        r"""
        Returns the simulation run name to pass to the backend for *combination*.

        ``"ParameterSearch"`` by default, which is what the result directory of every
        combination is named after. Override it to make the run name carry something about
        the combination, which is the only way to tell two result directories apart when the
        combinations do not differ in their modified parameters alone.

        Note that :func:`mozaik.meta_workflow.analysis.load_parameter_search` and
        :func:`parameter_search_run_script_distributed_slurm` rebuild result directory names
        assuming the default, so a search that overrides this will not be found by them.

        Parameters
        ----------

        combination : dict
            The modified parameters of the job, plus ``results_dir``.
        """
        return "ParameterSearch"

    def run_parameter_search(self):
        r"""
        This method will run the parameter search replacing each combination of values defined by dictionary params
        in the default parametrization and runing the simulation with each such modified parameters,
        storing the results of each simulation run in a subdirectory named based on the given modified parameter names and their
        values.

        It will read the command line for the name of the script that runs individual simulations, the simulator name and the root parameter file path
        Command line syntax:

        python parameter_search_script simulation_run_script simulator_name root_parameter_file_name

        Returns
        -------
        list
            The backend's identifier for each submitted job, in submission order. Entries are
            None for backends that have no such concept, or whose identifier could not be
            read back; a caller building job dependencies must tolerate that.
        """

        # Read parameters
        run_script, simulator_name, parameters_url = parse_parameter_search_args()

        mdn = self.master_directory(parameters_url)
        # exist_ok so that a search whose master_directory() is fixed can be run more than
        # once, each invocation adding to the same directory.
        os.makedirs(mdn, exist_ok=True)
        self.prepare(mdn)

        counter = 0
        job_ids = []
        combinations = self.generate_parameter_combinations()
        save_json(combinations, mdn + "/parameter_combinations.json")

        for combination in combinations:
            combination["results_dir"] = os.getcwd() + "/" + mdn + "/"
            job_ids.append(
                self.backend.execute_job(
                    run_script,
                    simulator_name,
                    parameters_url,
                    combination,
                    self.simulation_run_name(combination),
                    env=self.job_environment(combination),
                )
            )
            counter = counter + 1

        print("Submitted %d jobs." % counter)
        return job_ids


class CombinationParameterSearch(ParameterSearch):
    r"""
    A ParameterSearch that recevies a list of parameters and list of values for each parameter to test.
    It will then test each of the combination of values.

    Parameters
    ----------

    parameter_values : dict
        Dictionary containing parameter names as keys, and lists as values, each corresponding to the list of values to test for the given parameter.

    """

    def __init__(self, backend, parameter_values):
        ParameterSearch.__init__(self, backend)
        self.parameter_values = parameter_values

    def generate_parameter_combinations(self):
        combs = []
        for combination in parameter_combinations(list(self.parameter_values.values())):
            combs.append(
                {a: b for (a, b) in zip(self.parameter_values.keys(), combination)}
            )
        return combs

    def master_directory_name(self):
        s = (
            "CombinationParamSearch{"
            + ",".join(
                [
                    str(k)
                    + ":"
                    + (
                        str(self.parameter_values[k])
                        if len(self.parameter_values[k]) < 5
                        else str(len(self.parameter_values[k]))
                    )
                    for k in self.parameter_values.keys()
                ]
            )
            + "}/"
        )

        if len(s) > 200:
            s = (
                "CombinationParamSearch{"
                + str(len(self.parameter_values.keys()))
                + "}/"
            )
        return s


def parameter_combinations(arrays):
    return _parameter_combinations_rec([], arrays)


def _parameter_combinations_rec(combination, arrays):
    if arrays == []:
        return [combination]
    else:
        return sum(
            [
                _parameter_combinations_rec(combination[:] + [value], arrays[1:])
                for value in arrays[0]
            ],
            [],
        )


def parameter_search_run_script_distributed_slurm(
    simulation_name, master_results_dir, run_script, core_number, path_to_mozaik_env
):
    r"""
    Scheadules the execution of *run_script*, one per each parameter combination of an existing parameter search run.
    Each execution receives as the first commandline argument the directory in which the results for the given
    parameter combination were stored.

    Parameters
    ----------

    simulation_name : str
        The name of the simulation.

    master_results_dir : str
        The directory where the parameter search results are stored.

    run_script : str
        The name of the script to be run.
        The directory name of the given parameter combination datastore will be passed to it
        as the first command line argument.

    core_number : int
        How many cores to reserve per process.

    path_to_mozaik_env : str
        Path to the virtual environment activation script to source before running
        the analysis script.

    """
    if not path_to_mozaik_env:
        raise ValueError("path_to_mozaik_env must be a non-empty path")
    path_to_mozaik_env = os.path.expanduser(os.path.expandvars(path_to_mozaik_env))
    if not os.path.isfile(path_to_mozaik_env):
        raise FileNotFoundError(
            "path_to_mozaik_env does not exist or is not a file: " + path_to_mozaik_env
        )

    with open(
        master_results_dir + "/parameter_combinations.json", "r", encoding="utf-8"
    ) as f:
        combinations = json.load(f)

    # first check whether all parameter combinations contain the same parameter names
    assert (
        len(set([tuple(set(comb.keys())) for comb in combinations])) == 1
    ), "The parameter search didn't occur over a fixed set of parameters"

    from subprocess import Popen, PIPE, STDOUT

    for i, combination in enumerate(combinations):
        rdn = (
            master_results_dir
            + "/"
            + result_directory_name("ParameterSearch", simulation_name, combination)
        )
        p = Popen(
            ["sbatch"] + ["-o", master_results_dir + "/slurm_analysis-%j.out"],
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
            text=True,
        )

        # THIS IS A BIT OF A HACK, have to add customization for other people ...
        data = "\n".join(
            [
                "#!/bin/bash",
                "#SBATCH -J MozaikParamSearchAnalysis",
                "#SBATCH -c " + str(core_number),
                "#SBATCH --hint=nomultithread",
                "source " + path_to_mozaik_env,
                "cd " + os.getcwd(),
                " ".join(
                    ["python", run_script, "'" + rdn + "'"]
                    + [">"]
                    + ["'" + rdn + "/OUTFILE_analysis" + str(time.time()) + "'"]
                ),
            ]
        )
        print(p.communicate(input=data)[0])
        print(data)
        p.stdin.close()
