=================
Mozaik Tutorial 2
=================

In this tutorial we will design a new experiment, creating a new stimulus based on the functions already available in mozaik. 

Existing experiments
--------------------
In the first tutorial we briefly looked at the content of the file `experiment.py` from the VoglesAbbott2005 example project, where we specify experimental protocol.

In short an experiment consists of:

* The definition of a **stimulus** (possibly more than one) to be presented to our model.

* A **protocol** organizing the stimuli into repetitions (which typically change one or more parameters of the stimuli).

* An actual definition of the **experiment**, detailing parameter ranges and number of repetitions.

If we have a look at some of the experiments available in mozaik, we can see that the experiment regarding vision are stored in the file `/mozaik/experiments/vision.py`.

This file contains the definition of the base class for all the visual experiments, **VisualExperiment**. This class simply sets the two internal parameters
common to all visual experiments - density and background_luminance - to match values specified in the input space and input_layer.

The  classe below derived from VisualExperiment will guide us in understanding how a protocol is shaped. Each one has a meaningful name. Here we picked  `MeasureOrientationTuningFullfield` (see the mozaik/experiments/visio.py for the full class) that will assess the responses of our model to stimuli differing by their orientation and can also vary the contrast at which this is done.

The structure of the init function tells us the way a protocol works:

    def __init__(self, model, num_orientations, spatial_frequency,
                 temporal_frequency, grating_duration, contrasts, num_trials):
        VisualExperiment.__init__(self, model)
        for c in contrasts:
            for i in range(0, num_orientations):
                for k in range(0, num_trials):
                    self.stimuli.append(topo.FullfieldDriftingSinusoidalGrating(
                        frame_duration=7,
                        size_x=model.visual_field.size_x,
                        size_y=model.visual_field.size_y,
                        location_x=0.0,
                        location_y=0.0,
                        background_luminance=self.background_luminance,
                        contrast = c,
                        duration=grating_duration,
                        density=self.density,
                        trial=k,
                        orientation=numpy.pi/num_orientations*i,
                        spatial_frequency=spatial_frequency,
                        temporal_frequency=temporal_frequency))

Above we see::

* a cycle over the list of contrasts we want to check our model against,

* an inside cycle to turn over the number of orientation we would like to test

* an inner cycle to run several trials,

* a stimulus generator that fits our purpose, in this case `FullfieldDriftingSinusoidalGrating`.

The last point needs some comments. We append the stimulus to a list of `stimuli` which is owned by the `Experiment` class. For the experiment we have chosen, we use `FullfieldDriftingSinusoidalGrating` from the file `/mozaik/stimuli/vision/topographica_based.py`, that is::

    class FullfieldDriftingSinusoidalGrating(TopographicaBasedVisualStimulus):

        orientation = SNumber(rad, period=pi, bounds=[0,pi], doc="Grating orientation")
        spatial_frequency = SNumber(cpd, doc="Spatial frequency of grating")
        temporal_frequency = SNumber(Hz, doc="Temporal frequency of grating")
        contrast = SNumber(dimensionless,bounds=[0,100.0],doc="Contrast of the stimulus")

        def frames(self):
            self.current_phase=0
            i = 0
            while True:
                i += 1
                yield (imagen.SineGrating(orientation=self.orientation,
                          frequency=self.spatial_frequency,
                          phase=self.current_phase,
                          bounds=BoundingBox(radius=self.size_x/2),
                          offset = self.background_luminance*(100.0 - self.contrast)/100.0,
                          scale=2*self.background_luminance*self.contrast/100.0,
                          xdensity=self.density,
                          ydensity=self.density)(),
                      [self.current_phase])
                self.current_phase += 2*pi * (self.frame_duration/1000.0) * self.temporal_frequency

Simply put, this class is derived from a basic `TopographicaBasedVisualStimulus`, already containing in turn some functionalities and parameters from its base class `VisualStimulus`, to which this class adds its own parameters (orientation, spatial_frequency, temporal_frequency and contrast).

The stimulus itself relies on the stimulus generator (imagen) module (refer to their documentation for futher info, `imagen <http://topographica.org/Reference_Manual/imagen-module.html>`_). We are free to define the stimuli in any way we want, but so far in
mozaik (for visual stimuli) we have been using the imagen stimulus generator package. Therefore the stimuli are tipically only  wrappers around one of the imagen class.

As we can see, this last class produces a series of frames guided by the parameters. This means that each instantitation of the
stimulus class will produce a series of frames, in the case of the above class containing a drifting grating stimulus.
On top of that the `MeasureOrientationTuningFullfield` will create a series of such stimuli with changing paramters as dictated by the given experiemnt, in our case varing 
the orientation paramter to asses the orientation preference of neurons (and also contrast).

Let us now create our own new experiment...

Writing a new experiment
------------------------
We will now define a new type of stimulus and an experimental protocl that uses it. The experiment we want to design will test the responses of our model to full-field luminance step increments.

Stimulus
~~~~~~~~
First of all, we need a stimulus generator. There is already one that fits our purpose in the imagen library, and in turn in the topographica_based ones::

    class Null(TopographicaBasedVisualStimulus):
        """
        Blank stimulus.
        """
        def frames(self):
            while True:
                yield (imagen.Null(
                          scale=self.background_luminance,
                          bounds=BoundingBox(radius=self.size_x/2),
                          xdensity=self.density,
                          ydensity=self.density)(),
                      [self.frame_duration])

This generator produces frames with a constant luminance (the scale parameter) values accross the whole screen. That's the only value we will need to supply to this class (some other are assigned by default).

Protocol
~~~~~~~~
Then we need to define the protocol that will use this stimulus. We want to be able to specify a certain number of luminance steps, the duration of presentation and the number of trials. We create a new class, derived from VisualExperiment, into `mozaik/experiments/vision.py`.

Our parameters are::

* **model** : the Model object on which to execute the experiment.

* **luminances** : a list(float) of luminance (expressed as cd/m^2) at which to measure the response.
    
* **step_duration** : a float expressing the duration of single presentation of a luminance step.
    
* **num_trials** : an integer for the number of trials each stimulus is shown.

The init function contains the outmost cycle on the list of luminances and an inner one for the trials::

    class MeasureLuminanceSensitivity(VisualExperiment):
    
        def __init__(self, model, luminances, step_duration, num_trials):
            VisualExperiment.__init__(self, model)    
            # stimuli creation        
            for l in luminances:
                for k in range(0, num_trials):
                    self.stimuli.append( topo.Null(
                        frame_duration=7,
                        size_x=model.visual_field.size_x,
                        size_y=model.visual_field.size_y,
                        location_x=0.0,
                        location_y=0.0,
                        density=self.density,
                        background_luminance=l,
                        duration=step_duration,
                        trial=k))

Notice that we specified a duration of 7ms for each frame. It is hardcoded here and we will use it as a base when specifying the duration at the experiment level.

We didn't change much compared to other protocols, this will be often the case.


Experiment
~~~~~~~~~~
In the `experiments.py` file of our model we now can use the new protocol for an experiment. As expected we pass our model, a list of luminances, the duration of each step (as a multiple of the base frame duration) and the number of trials::

    MeasureLuminanceSensitivity(
        model, 
        luminances=[1.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 200.0, 300.0, 400.0],
        step_duration=147*7,
        num_trials=4
    ),

And that's it. Of course there can be more complex stimuli (see for example `DriftingSinusoidalGratingDisk` or `DriftingSinusoidalGratingCenterSurroundStimulus`) and more articulated protocols, although none of the ones needed up to now really is. This is one of the pros of working in mozaik, it allows for very general definitions, but usually keeps things very concise and simple (at least once we learn it :-))


Analysis and Visualization
--------------------------
Once we will have our experiment done, we will need to analyse the results that came out of the experiment. 
Let's have a brief look at an example of such analysis. As shown in tutorial 1, we will first get the ids of the units we recorded from::

    analog_Xon_ids = sorted( param_filter_query(data_store,sheet_name="X_ON").get_segments()[0].get_stored_vm_ids() )

We will filter our data_store set by taking only the part of recorded traces that were obtained during the 'Null' stimulus ::

    dsv = param_filter_query( data_store, st_name='Null', sheet_name='X_ON' )  

Next, we will compute the average firing rate::

    TrialAveragedFiringRate( dsv, ParameterSet({}) ).analyse()

We then select our results by specifying the stimulus who generated the data and the analysis algorithm we used::

    dsv = param_filter_query( data_store, st_name='Null', analysis_algorithm=['TrialAveragedFiringRate'] )

And finally plot them as a tuning curve for luminance sensitivity::

    PlotTuningCurve(
        dsv,
        ParameterSet({
            'parameter_name' : 'background_luminance', 
            'neurons': list(analog_Xon_ids), 
            'sheet_name' : 'X_ON'
        }), 
        fig_param={'dpi' : 100,'figsize': (16,6)}, 
        plot_file_name="LuminanceSensitivity_LGN_On.png"
    ).plot({
        '*.fontsize':7
    })


A final note. Over the various modification we will make in order to test our model against different stimuli, we shall remember that if we change something, we must check the consistency among the chain of classes involved: experiment, analysis and plotting!

Happy mozaiking!
