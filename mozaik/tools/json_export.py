import os
import re
import json
import numpy as np
from numpyencoder import NumpyEncoder
from sphinx.util import docstrings
import imageio

PARAMETERS_REGEX = re.compile(".*Parameters.*")
OTHER_PARAMETER_REGEX = re.compile(".*Other\ [pP]arameters\ *\n-{15}-+")
PARAMETER_REGEX = re.compile(
    "\s*(?P<name>[^:\s]+)\s*\:\s* (?P<tpe>[^\n]*)\n\s*(?P<doc>[^\n]*)"
)

def save_json(d, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)

def get_params_from_docstring(cls):
    params = {}
    for cls1 in cls.__mro__:
        params.update(parse_docstring(cls1.__doc__)["params"])
    return params

def parse_docstring(docstring):
    """
    Parse the docstring into its components.

    Returns
    -------
    
    Returns a dictionary containing the parsed docstring components with these keys:
        - 'short_description': The first line of the docstring (str)
        - 'long_description': The remaining description text (str)
        - 'params': A list of parameter dictionaries, each containing:
            * 'name': The parameter name (str)
            * 'doc': The parameter description (str)
        - 'returns': Description of the return value (str)


    """

    short_description = long_description = returns = ""
    params = []

    if docstring:
        docstring = "\n".join(docstrings.prepare_docstring(docstring))

        lines = docstring.split("\n", 1)
        short_description = lines[0]

        if len(lines) > 1:
            reminder = lines[1].strip()
            match_parameters = PARAMETERS_REGEX.search(reminder)
            if match_parameters:
                long_desc_end = match_parameters.start()
                long_description = reminder[:long_desc_end].rstrip()
                reminder = reminder[long_desc_end:].strip()

            match = OTHER_PARAMETER_REGEX.search(reminder)

            if match:
                end = match.start()
                if not match_parameters:
                    long_description = reminder[:end].rstrip()
                reminder = reminder[end:].strip()

            if reminder:
                params = {}

                for name, tpe, doc in PARAMETER_REGEX.findall(reminder):
                    params[name] = (tpe, doc)

            if (not match_parameters) and (not match):
                long_description = reminder

    return {
        "short_description": short_description,
        "long_description": long_description,
        "params": params,
    }

def get_recorders(parameters):
    recorders_docs = []
    for sh in parameters["sheets"].keys():
        for rec in parameters["sheets"][sh]["params"]["recorders"].keys():
            recorder = parameters["sheets"][sh]["params"]["recorders"][rec]
            name = recorder["component"].split(".")[-1]
            module_path = ".".join(recorder["component"].split(".")[:-1])
            doc_par = get_params_from_docstring(
                getattr(__import__(module_path, globals(), locals(), name), name)
            )
            p = {
                k: (recorder["params"][k], doc_par[k][0], doc_par[k][1])
                for k in recorder["params"].keys()
            }

            recorders_docs.append(
                {
                    "code": module_path + "." + name,
                    "short_description": parse_docstring(
                        getattr(
                            __import__(module_path, globals(), locals(), name), name
                        ).__doc__
                    )["short_description"],
                    "long_description": parse_docstring(
                        getattr(
                            __import__(module_path, globals(), locals(), name), name
                        ).__doc__
                    )["long_description"],
                    "parameters": p,
                    "variables": recorder["variables"],
                    "source": sh,
                }
            )
    return recorders_docs

def get_experimental_protocols(data_store):
    experimental_protocols_docs = []
    for ep in data_store.get_experiment_parametrization_list():
        name = ep[0][8:-2].split(".")[-1]
        module_path = ".".join(ep[0][8:-2].split(".")[:-1])
        doc_par = get_params_from_docstring(
            getattr(__import__(module_path, globals(), locals(), name), name)
        )
        params = eval(ep[1])

        p = {
            k: (params[k], doc_par[k][0], doc_par[k][1])
            if k in doc_par
            else params[k]
            for k in params.keys()
        }

        experimental_protocols_docs.append(
            {
                "class": module_path + "." + name,
                "short_description": parse_docstring(
                    getattr(
                        __import__(module_path, globals(), locals(), name), name
                    ).__doc__
                )["short_description"],
                "long_description": parse_docstring(
                    getattr(
                        __import__(module_path, globals(), locals(), name), name
                    ).__doc__
                )["long_description"],
                "parameters": p,
            }
        )
    return experimental_protocols_docs

from mozaik.tools.mozaik_parametrized import MozaikParametrized

def reduce_dicts(dicts):
    constant = {k : True for k in dicts.keys()}
    for d in dicts():
        continue

def get_stimuli(data_store, store_stimuli, input_space):
    stim_docs = []
    if not store_stimuli:
        return stim_docs
    unique_stimuli = [s for s in set(data_store.get_stimuli())]
    stim_dir = "stimuli/"
    os.makedirs(data_store.parameters.root_directory + stim_dir, exist_ok=True)
    for s in unique_stimuli:
        sidd = MozaikParametrized.idd(s)
        params = sidd.get_param_values()
        params = {k: (v, sidd.params()[k].doc) for k, v in params}

        # Only save one trial of each different stimulus
        if params["trial"][0] != 0:
            continue

        raws = data_store.get_sensory_stimulus([s])

        if raws == [] or raws[0] == None:
            img = np.zeros((50,50)).astype(np.uint8)
            raws = [img,img]
        else:
            raws = raws[0]

        mov_duration = input_space["update_interval"] / 1000.0 if input_space != None else 0.1
        gif_name = params["name"][0] + str(hash(s)) + ".gif"
        imageio.mimwrite(data_store.parameters.root_directory + stim_dir + gif_name, raws, duration=mov_duration)

        stim_docs.append(
            {
                "code": sidd.name,
                "short_description": parse_docstring(
                    getattr(
                        __import__(
                            sidd.module_path, globals(), locals(), sidd.name
                        ),
                        sidd.name,
                    ).__doc__
                )["short_description"],
                "long_description": parse_docstring(
                    getattr(
                        __import__(
                            sidd.module_path, globals(), locals(), sidd.name
                        ),
                        sidd.name,
                    ).__doc__
                )["long_description"],
                "parameters": params,
                "movie": stim_dir + gif_name,
            }
        )
    return stim_docs
