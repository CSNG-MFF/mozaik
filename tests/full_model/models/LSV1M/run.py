# -*- coding: utf-8 -*-

import matplotlib

matplotlib.use("Agg")
from model import SelfSustainedPushPull
from experiments import create_experiments
import mozaik
from mozaik.controller import run_workflow

data_store, model = run_workflow("LSV1M", SelfSustainedPushPull, create_experiments)
data_store.save()
