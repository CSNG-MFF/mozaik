Mozaik is a workflow system for spiking neuronal network simulations written in Python that integrates model, experiment and stimulation specification, simulation execution, data storage, data analysis and visualization into a single automated workflow. This way, Mozaik increases the productivity of running virtual experiments on complex heterogenous spiking neuronal networks.

You can read more about Mozaik here:  https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2013.00034/full
You can look at the code here:https://github.com/CSNG-MFF/mozaik/tree/master

We have built a model of early visual system (LSV1M) in Mozaik you can read about here: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012342

The goal of this project is to have ability to instead of modeling visual input into the LSV1M, to be able to send events from 
DVS camera. The idea is to replace current LGN neurons with neurons that will have spikes injected into the corresponding to events from DVI camera.
Positive events will be injected to ON LGN cells, negative events to OFF LGN cells.

The milestones of the project are as follows:
1. Create new visual cortical sheet with regular grid of neurons (current sheets place neurons randomly accross the sheet).
2. Write new experiments that will inject the events from the DVS stored in a file to such a sheet of neurons respecting the visual field posiion of the DVS pixels. To be clear in this case the neurons in the sheet should output these spikes.
3. Replace the current visual input model in LSV1M with this new DVS input sheet.

Later we will work on analysing and visualising the data from this model.

The way the development will work is that you will execute the plan, milestone by milestone. Each milestone implementation will consists of following steps:
    1. Start the next unfinished milestone
    2. Create a more detailed milestone plan that also includes set of validation conditions that the implementation has to meet.
    3. Let me approve it.
    3. Write that more milestone plan into PLAN.md and update the milestone status 
    4. Write pytests that verify the milestone vlidation conditions
    5. Implement the milestone. Work until all the tests are passing.
    6. Let me approve the changes.
    7. Update the milestone status.
    8 Goto 1.
