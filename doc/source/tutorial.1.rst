=================
Mozaik Tutorial 1
=================

This tutorial assumes we have already installed mozaik. Being a workflow organizer, mozaik relies on several software packages and programs, e.g. a simulator like `NEST <http://www.nest-initiative.org/index.php/Software:About_NEST>`_, the `PyNN <http://neuralensemble.org/PyNN/>`_ simulation language, and their dependencies... Please refer to the installation instructions for further details.

This tutorial is based around example project that can be found in mozaik/examples/VogelsAbbott2005, which creates a simple randomly connected balanced network
that should be able to self-sustain its actvity with no external stimulus.

Project folder organization
---------------------------
There would be no need to have the code for our project in separate files but it is a good practice for larger projects. Let's go through each file of the VogelsAbbott project, each dealing with a part of the workflow:

* **model.py** contains the high level specification of the network (sheet creation, reference to parameters, connection schema, ...).

* **param** is a folder containing the parameter tree for the model, itself sepparated into several files

* **experiments.py** contains the experimental protocol we wish to execute over the model.

* **run.py** the top level script, executes the experiment and, eventually, saves the results to file, runs analysis, plotting, ...

* **analysis_and_visualization.py** specifies how to filter recorded data, what algorithm apply to these results and how to create figures.


File Content
------------
Now let's take a look at the content of the files. For simplicity, to force ourselves in looking more at the actual files and to keep focused on what we need to look at in them, only the relevant lines will be shown here. In any case the number of lines is very little compared to what we can accomplish with them.


model.py
~~~~~~~~
There are two main components of mozaik: **sheets** and **connectors**. These are the ones we are going to set in this file. 
Mozaik comes with a whole set of sheets and connectors, we can specify their properties by assigning their parameteters. In addition, we can define our own sheets and connectors, and we will do this in the advanced tutorial. Each model should be derived from the mozaik base class Model::

    class VogelsAbbott( Model )

Inside the model constructor we will specify a list of parameters that will correspond to each of the sheets our network will have.

Upon creation of the object *ParameterSet*, mozaik will load the parameters according to the names we specified in our model::

    required_parameters = ParameterSet({
        'exc_layer' : ParameterSet,
        'inh_layer' : ParameterSet,
    })

In fact *Model* is a hierarchy of objects which represent sheets, connectors and other subobjects. 
To create instances of them, we use a corresponding hierarchy of parameters which are read during init. 
Therefore each of the above parameters is itself ParameterSet, as it will correspond to the subtree of parameters
required to parameterize the given sheet and all its related components.
We will set some of them in the top level parameter file named *default* and others in other files inside param folder (more on this in the next section).


Then we will use the sheets component parameter to load the class corresponding to it::

    def __init__(self, sim, num_threads, parameters):
        Model.__init__(self, sim, num_threads, parameters)
        # Load components
        ExcLayer = load_component(self.parameters.exc_layer.component)
        InhLayer = load_component(self.parameters.inh_layer.component)

The function *load_component()* creates the classes that represent our sheets. We then create the instances
of the sheets and pass them the parameters that their require.

        exc = ExcLayer(self, self.parameters.exc_layer.params)
        inh = InhLayer(self, self.parameters.inh_layer.params)

Then we will use the sheet instances to specify how to connect each sheet in itself and against other sheets, using already available connectors or our custom defined ones (more on this in the advanced tutorial)::

        # initialize projections
        UniformProbabilisticArborization(
            self,
            'ExcExcConnection',
            exc,
            exc,
            self.parameters.exc_layer.ExcExcConnection
        ).connect()
        # … and so on


param folder
~~~~~~~~~~~~
Another important component of mozaik is the management of parameters. All parameters are loaded by mozaik automatically from the root parameter file that is given to it on command line (see below), and recursively any other paramter files that are referenced from it (or from the other parameter files being loaded). In our case the root parameter file is called *default*. The root parameter file contains a dictionary of basic parameters used by mozaik to setup
your project as well as references to other parameter files that will be included and expanded.

The file *default* contains some general parameters which are almost self explaining (have a look at the documentation of the Model class)::

  {
    'exc_layer': url("param/exc_layer"),
    'inh_layer': url("param/inh_layer"),
    'results_dir': '',
    'name' : 'Vogels&Abbott',
    'reset' : False,
    'null_stimulus_period' : 0.0,
    'input_space' : None,
    'input_space_type' : 'None',
  }

The other files inside *param* contain parameters specific for each sheet. Let's see some of them, as before only the relevant ones will be reported here.
In the *component* parameter we specify the class to generate this sheet::

    'component': 'mozaik.sheets.vision.VisualCorticalUniformSheet',

Then, in params, we can detail the parameters for the type of sheet we chose::

    'params':{
                'name':'Exc_Layer',
                ...,

The parameters can be nested, as for the cell model (with its own params!) used in this sheet::

                'cell': {
                        'model': 'IF_cond_exp',
                        'params': {

Note that each file contains as well parameters for all the Connector classes specifing projections originating from this sheet::

    'ExcExcConnection': {

In the parameter files we can refer to other parameter by using references, i.e.::

    'ExcInhConnection': ref('exc_layer.ExcExcConnection'),

Generally when looking at the parameter files user should be able to track down which parameter belongs to which class and then check it's
description in the documentation of that class.

We can check, and modify, what is recorded by looking at parameter 'recorders' for each sheet. For example, inside the parameter file for the excitatory sheet (as above), we find::

    'recorders' : url("param/exc_rec"),

which tells us to look at *exc_rec* file to know the details of recording specifications::

    {
        "1": {
            'component' : 'mozaik.sheets.population_selector.RCRandomN',
            'variables' : ("spikes"),
            'params' : { 'num_of_cells' : 100 }
        },
        "2": {
            'component' : 'mozaik.sheets.population_selector.RCRandomN',
            'variables' : ("spikes", "v", "gsyn_exc", "gsyn_inh"),
            'params' : { 'num_of_cells' : 21 }
        },
    }

This is neat! We are telling mozaik to record two *things* from the exc_layer. 
The first one is just "spikes" (spike trains) from 100 randomly selected cells. 
The second recording is a bit more complex, it instructs to store voltage and conductances (excitatory and inhibitory) from 21 randomly selected cells. 
All these recordings will be automatically stored in the datastore for us and will come handy afterwards when we will use the *datastore* to run analysis and create figures
based on the recorded data...


experiment.py
~~~~~~~~~~~~~
How to run the experiment is something unrelated to model creation. This is why we specify our experimental protocol in a separate file 
(then to reuse the same network with a different protocol we just need to use another experiment file).

We write a method *create_experiment* to establish our protocol. In this case, we only want to give an initial kick, external spike train, 
to the network followed by a period or recording when the network is running on its own. To do this we will have two experiments one that 
supplies the network with the initial Kick (see the `PoissonNetworkKick` below), and one 'dummy' experiment which effectively does nothing but keeps recording the
network (see the `NoStimulation` below)::

    return  [
                #Lets kick the network up into activation
                PoissonNetworkKick(
                                    model,duration=8*7,
                                    sheet_list=["V1_Exc_L4","V1_Inh_L4"],
                                    stimulation_configuration={
                                                              'component' : 'mozaik.sheets.population_selector.RCRandomPercentage',
                                                              'params' : {'percentage' : 20.0}
                                                            },
                                    lambda_list=[100.0,100.0],
                                    weight_list=[0.1,0.1]
                                  ),
                #Spontaneous Activity
                NoStimulation( model, duration=3*8*7 ),
    ]

As you can see the `PoissonNetworkKick` gets several parameters. One, common to all experiments, is the stimulation_configuration which specifies which
neurons will be stimulated during this experiment. This is defined with a sub-class of PopulationSelector for which the stimulation_configuration parameter holds parameters. 
The component parameter defines the population selector class and the param parameter the dictionary of parameters that will be passed to the population selector at initialization as ParameterSet.


run.py
~~~~~~
The run.py is our top level execution file.

We start our simulation with one line. We chose to put it in a separate file thus we can add other running-related operations, like logging and plotting.
The single interesting line here is::


   data_store,model = run_workflow( 'VogelsAbbott2005', VogelsAbbott, create_experiments )

As we can see, we pass to run_workflow the name of our project, the model class (that we specify in the model.py and configure via the configuration files) 
and a function which returns the list of experiments to run on it.
The run_workflow passes the control to mozaik, which will take care of construction and simulation the model as well as execution of all the experiments. 
It returns an instance of data_store contining the recorded data (dilligently labled with all 
relevant meta-data), which we can then use for analysis and visualization, see line below::

   perform_analysis_and_visualization(data_store)


analysis_and_visualization.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Last but not least, we can have a file to write our own analysis and plotting procedure. Mozaik comes with a set of analysis tools that we can further expand (which we will show how in later tutorials). In this example we will use just a couple of them in order to familirize with the process.

The method that gets called in the previous run.py file receives the simulation output *datastore*::

  def perform_analysis_and_visualization( data_store )

As seen before, *datastore* is a object holding a collection of recordings and analysis results.
The recordings are stored as a list of `NEO <http://pythonhosted.org/neo/>`_ segments containing analog signals and spikes of cells from our sheets that we have instructed mozaik to record. 
There are methods in the query mozaik subpackaged  that allow us to create sub-views of the *datastore* effectivly perform various filterings on our records base on range 
of metadata such as the identyty and parameters of stimuli to which the recordings have been obtained. 
Here we take only a simple subset and leave more sophisticated operations for a more advanced tutorial. 
To understand the analysis process we do a very simple, though still meaningful, example. 
We simply take all recordings done in layer 'Exc_Layer' (funtion `param_filter_query`), retrieve the segments from this data
store view, pick the first segment and get the ids of the neurons for which the excitatory conductances were stored, and saved this list 
in the variable `analog_ids`::

  analog_ids = sorted(param_filter_query(data_store,sheet_name="Exc_Layer").get_segments()[0].get_stored_esyn_ids())

We filter our data_store set by taking only the part of recorded traces that were obtained during spontaneous activity after the kick.
Then, we will process our traces using a Peri-Stimulus Time Histogram with bin length set at 5ms::

  PSTH(
     param_filter_query( data_store, st_direct_stimulation_name="None" ),
     ParameterSet({'bin_length' : 5.0})
  ).analyse()

Next, we compute the average firing rate, on the same data_store subset, which effectivly will be spike count per neuron
as in this experiment we had only one trial::

  TrialAveragedFiringRate(
     param_filter_query( data_store, st_direct_stimulation_name="None" ),
     ParameterSet({})
  ).analyse()

Then we check the correlation among analog signals on a per neuron basis::

 NeuronToNeuronAnalogSignalCorrelations(
     param_filter_query( data_store, analysis_algorithm='PSTH' ),
     ParameterSet({'convert_nan_to_zero' : True})
 ).analyse()

Note that, during the procedure, we can end up with some NaN values due to correlations not being defined between zero arrays. We can specify the conversion to be applied in this case. Finally we compute the population mean over any `PerNeuronValue` analysis data structures so far added to the datastore, which will effectively 
give us the average firing rate and PSTH correlations accross neurons::

  PopulationMean( data_store, ParameterSet({}) ).analyse()

Plotting
~~~~~~~~

There are several nice things about plotting in mozaik. Plots are easily defined in all their aspects, using the same parameters approach common across mozaik.
We simply take our data_store view from a query and pass it to the plot creator, which is based around matplotlib::

 dsv = param_filter_query(data_store,st_direct_stimulation_name=['None'])

 OverviewPlot(
     dsv,
     ParameterSet({
          'sheet_name' : 'Exc_Layer',
          'neuron' : analog_ids[0],
          'sheet_activity' : {}
     }),
     fig_param={'dpi' : 100,'figsize': (19,12)},
     plot_file_name='ExcAnalog1.png'
 ).plot({
     'Vm_plot.y_lim' : (-80,-50),
     'Conductance_plot.y_lim' : (0,500.0)
 })

Results
~~~~~~~
Running this project is as easy as enter this command line in the mozaik/contrib directory::

  $ mpirun -np 4 python run.py nest 1 param/defaults 'test'

In this example mozaik uses MPI to run the simulation as 4 processes, each using single thread and NEST as simulator. These are specified as command line parameters, together with the name for this specific simulation run (in this case 'test').

The command will produce a quite long series of logging lines in our terminal, which we can briefly review (and which can be shut down commenting out logging in the run.py file). At start, our backend simulator, NEST, is called by PyNN on behalf of mozaik::

              -- N E S T --

  Copyright (C) 2004 The NEST Initiative
  Version 2.2.2 Jul  5 2013 15:53:57

  This program is provided AS IS and comes with
  NO WARRANTY. See the file LICENSE for details.

  Problems or suggestions?
    Website     : http://www.nest-initiative.org
    Mailing list: nest_user@nest-initiative.org

  Type 'nest.help()' to find out more about NEST.

Then we have mozaik actually loading and working the classes to create our sheets,  connect them and execute the simulation::

  0    Loaded component VisualCorticalUniformSheet from module mozaik.sheets.vision
  0    Loaded component VisualCorticalUniformSheet from module mozaik.sheets.vision
  0    Creating VisualCorticalUniformSheet with 3200 neurons.
  0  NEST does not allow setting an initial value for g_ex
  0  NEST does not allow setting an initial value for g_in
  0    Loaded component RCRandomN from module mozaik.sheets.population_selector
  0    Loaded component RCRandomN from module mozaik.sheets.population_selector
  0    Creating VisualCorticalUniformSheet with 800 neurons.
  0  NEST does not allow setting an initial value for g_ex
  0  NEST does not allow setting an initial value for g_in
  0    Loaded component RCRandomN from module mozaik.sheets.population_selector
  0    Loaded component RCRandomN from module mozaik.sheets.population_selector
  0    Creating UniformProbabilisticArborization between VisualCorticalUniformSheet and VisualCorticalUniformSheet
  0    Connector UniformProbabilisticArborization took 1s to compute
  0    Creating UniformProbabilisticArborization between VisualCorticalUniformSheet and VisualCorticalUniformSheet
  0    Connector UniformProbabilisticArborization took 0s to compute
  0    Creating UniformProbabilisticArborization between VisualCorticalUniformSheet and VisualCorticalUniformSheet
  0    Connector UniformProbabilisticArborization took 1s to compute
  0    Creating UniformProbabilisticArborization between VisualCorticalUniformSheet and VisualCorticalUniformSheet
  0    Connector UniformProbabilisticArborization took 0s to compute
  0    Starting Experiemnts
  0    Starting experiment: PoissonNetworkKick
  0    Running model
  0    Simulating the network for 56 ms
  0    Finished simulating the network for 56 ms
  0    Stimulus 1/1 finished. Memory usage: 194MB
  0    Experiment 1/2 finished
  0    Starting experiment: NoStimulation
  0    Running model
  0    Simulating the network for 168 ms
  0    Finished simulating the network for 168 ms
  0    Stimulus 1/1 finished. Memory usage: 199MB
  0    Experiment 2/2 finished
  0    Total simulation run time: 14s
  0    Simulator run time: 6s (46%)
  0    Mozaik run time: 7s (53%)
  Final memory usage: 199MB
  There are some notes to these lines.

| We see that NEST emits some alerts, due to initializations which cannot be accomplished. Don't worry they don't affect our simulation (they are just specification of PyNN not met in NEST).
| Then we can see mozaik classes loaded to accomplish what we specified in our files: our model is derived from VisualCorticalUniformSheet and connected using UniformProbabilisticArborization. After network creation, our experiment is performed, which is composed of two phases (PoissonNetworkKick and NoStimulation). Data is then recorded and some runtime statistic about the simulation emitted.

Since we also chose to have some analysis and plotting, we can see logs also for these activities::

  Starting visualization
  0    Starting PSTH analysis
  0  PSTH analysis took: 0.262467861176seconds
  0    Starting TrialAveragedFiringRate analysis
  0  TrialAveragedFiringRate analysis took: 0.21697306633seconds
  ...
  0  OverviewPlot plotting took: 0.769396066666seconds
  0  OverviewPlot plotting took: 0.57945394516seconds
  0  OverviewPlot plotting took: 0.617439985275seconds
  0  RasterPlot plotting took: 0.31623506546seconds
  0  RasterPlot plotting took: 0.31383895874seconds

All data and figures from the experiment are saved by mozaik in an additional folder, having the name we specified in the run_workflow call, 
with appended the simulation run name specified in the command line (<model_name>_<simulation_instance_name>____). We specified result folder location in the file param/defaults::

'results_dir': ''

Left blank, mozaik will assume that we want our results in the same folder as our project. Indeed, there we find our results folder "VogelsAbbott2005_test_____*" containing several data_store pickled files and images containing the figure we have specified to plot.

Happy mozaiking!
