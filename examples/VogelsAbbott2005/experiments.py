#!/usr/local/bin/ipython -i 
from mozaik.experiments import *
from mozaik.sheets.population_selector import RCRandomPercentage
from parameters import ParameterSet
    
def create_experiments(model):
    l4exc_kick = RCRandomPercentage(model.sheets["Exc_Layer"],ParameterSet({'percentage': 20.0}))
    l4inh_kick = RCRandomPercentage(model.sheets["Inh_Layer"],ParameterSet({'percentage': 20.0}))

    return  [
                           #Lets kick the network up into activation
                           PoissonNetworkKick(model,duration=8*7,sheet_list=["Exc_Layer","Inh_Layer"],recording_configuration_list=[l4exc_kick,l4inh_kick],lambda_list=[100,100]),
                           #Spontaneous Activity 
                           NoStimulation(model,duration=3*8*7),
            ]
