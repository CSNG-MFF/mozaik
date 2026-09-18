"""
Tests for :func:`mozaik.meta_workflow.experanto_export.resolve_datastore`, which finds the
datastore a given ``(trial, chunk)`` was simulated into by globbing a stable name prefix
inside the results directory.

What is easy to get wrong there -- and did go wrong -- is that the results directory is a
literal path being spliced into a glob pattern, and a ``ParameterSearch`` results directory
contains glob metacharacters.
"""

import os

from mozaik.meta_workflow.experanto_export import resolve_datastore


def test_resolves_inside_a_parameter_search_master_directory(tmp_path):
    """
    A ParameterSearch master directory embeds the parameter file name in brackets, which glob
    reads as a character class. Unescaped, the pattern matches nothing and the export fails
    with "No datastore match" -- after every simulation of the run has already finished --
    even though the datastore is sitting right there.
    """
    master_dir = os.path.join(
        str(tmp_path),
        "20260917-123652[param_experanto.defaults]RandomizedExperanto{trials:2,chunks:5}",
    )
    datastore = os.path.join(
        master_dir, "SelfSustainedPushPull_trial0_chunk0_____simulation_seed:1_trial:0"
    )
    os.makedirs(datastore)

    assert resolve_datastore(master_dir, 0, 0) == datastore
