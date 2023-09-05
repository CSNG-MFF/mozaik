# -*- coding: utf-8 -*-

import matplotlib

matplotlib.use("Agg")
from model import SelfSustainedPushPull
from experiments import create_experiments
import mozaik
from mozaik.controller import run_workflow
from mpi4py import MPI

import nest
nest.Install("stepcurrentmodule")

mpi_comm = MPI.COMM_WORLD

data_store, model = run_workflow("LSV1M", SelfSustainedPushPull, create_experiments)

if mpi_comm.rank == 0:
    data_store.save()
