from mozaik.experiments.vision import VisualExperiment
from parameters import ParameterSet
import mozaik.stimuli.vision.topographica_based as topo
import numpy
import numpy as np
import random
import mozaik


class MapSimpleGabor(VisualExperiment):
    """
    Map RF with a Gabor patch stimuli.

    This experiment presents a series of flashed Gabor patches at the centers
    of regular hexagonal tides with given range of orientations.


    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------
    relative_luminance : float
        Luminance of the Gabor patch relative to background luminance.
        0. is dark, 1.0 is double the background luminance.

    central_rel_lum : float
        Luminance of the Gabor patch at the center of the RF relative to
        background luminance.
        0. is dark, 1.0 is double the background luminance.

    orientation : float
        The initial orientation of the Gabor patch.

    phase : float
        The phase of the Gabor patch.

    spatial_frequency : float
        The spatial freqency of the Gabor patch.

    rotations : int
        Number of different orientations at each given place.
        1 only one Gabor patch with initial orientation will be presented at
        given place, N>1 N different orientations will be presented,
        orientations are uniformly distributed between [0, 2*pi) + orientation.

    size : float
        Size of the tides. From this value the size of Gabor patch is derived
        so that it fits into a circle with diameter equal to this size.

        Gabor patch size is set so that sigma of Gaussian envelope is size/3

    x : float
        The x corrdinates of the central tide.

    y : float
        The y corrdinates of the central tide.

    flash_duration : float
        The duration of the presentation of a single Gabor patch.

    duration : float
        The duration of single presentation of the stimulus.

    num_trials : int
        Number of trials each each stimulus is shown.

    circles : int
        Number of "circles" where the Gabor patch is presented.
        1: only at the central point the Gabor patch is presented,
        2: stimuli are presented at the central hexagonal tide and 6 hexes
        forming a "circle" around the central

    grid : bool
        If True hexagonal tiding with relative luminance 0 is drawn over the
        stimmuli.
        Mostly for testing purposes to check the stimuli are generated
        correctly.

    Note on hexagonal tiding:
    -------------------------
        Generating coordinates of centers of regular (!) hexagonal tidings.
        It is done this way, because the centers of tides are not on circles (!)
        First it generates integer indexed centers like this:
              . . .                (-2,2) (0, 2) (2,2)
             . . . .           (-3,1) (-1,1) (1,1) (3,1)
            . . . . .   ==> (-4,0) (-2,0) (0,0) (2,0) (4,0)     (circles=3)
             . . . .           (-3,-1)(-1,-1)(1,-1)(3,-1)
              . . .                (-2,-2)(0,-2)(2,-2)

        coordinates then multiplied by non-integer factor to get the right position
            x coordinate multiplied by factor 1/2*size
            y coordinate multiplied by factor sqrt(3)/2*size

    Note on central relative luminance:
    -----------------------------------
        In the experiment they had lower luminance for Gabor patches presented
        at the central tide
    """

    required_parameters = ParameterSet(
        {
            "relative_luminance": float,
            "central_rel_lum": float,
            "orientation": float,
            "phase": float,
            "spatial_frequency": float,
            "size": float,
            "flash_duration": float,
            "x": float,
            "y": float,
            "rotations": int,
            "duration": float,
            "num_trials": int,
            "circles": int,
            "grid": bool,
        }
    )

    def generate_stimuli(self):
        if self.parameters.grid:
            # Grid is currently working only for special cases
            # Check if it is working
            assert self.parameters.x == 0, "X shift not yet implemented"
            assert self.parameters.y == 0, "Y shift not yet implemented"
            assert (
                self.model.visual_field.size_x == self.model.visual_field.size_y
            ), "Different sizes not yet implemented"
        for trial in range(0, self.parameters.num_trials):
            for rot in range(0, self.parameters.rotations):
                for row in range(
                    self.parameters.circles - 1, -self.parameters.circles, -1
                ):
                    colmax = 2 * self.parameters.circles - 2 - abs(row)
                    for column in range(-colmax, colmax + 1, 2):
                        # central coordinates of presented Gabor patch
                        # relative to the central tide
                        x = column * 0.5 * self.parameters.size
                        y = row * 0.5 * self.parameters.size
                        # different luminance for central tide
                        if column == 0 and row == 0:
                            rel_lum = self.parameters.central_rel_lum
                        else:
                            rel_lum = self.parameters.relative_luminance
                        self.stimuli.append(
                            topo.SimpleGaborPatch(
                                frame_duration=self.frame_duration,
                                duration=self.parameters.duration,
                                flash_duration=self.parameters.flash_duration,
                                size_x=self.model.visual_field.size_x,
                                size_y=self.model.visual_field.size_y,
                                background_luminance=self.background_luminance,
                                relative_luminance=rel_lum,
                                orientation=(
                                    self.parameters.orientation
                                    + numpy.pi * rot / self.parameters.rotations
                                ),
                                density=self.density,
                                phase=self.parameters.phase,
                                spatial_frequency=self.parameters.spatial_frequency,
                                size=self.parameters.size,
                                x=self.parameters.x + x,
                                y=self.parameters.y + y,
                                location_x=0.0,
                                location_y=0.0,
                                trial=trial,
                            )
                        )

    def do_analysis(self, data_store):
        pass


class MapTwoStrokeGabor(VisualExperiment):
    """
    Map RF with a two stroke Gabor patch stimuli to study response on apparent
    movement. First a Gabor patch is presented for specified time after that
    another Gabor patch is presented at neighbohring tide with same orientation
    and other properties.

    There are two configuration for the movement:
        ISO i.e. Gabor patch moves parallel to its orientation
        CROSS i.e. Gabor patch moves perpendicular to its orientation

        In any case it has to move into another tide, therefore orientation
        determines the configuration


    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------
    relative_luminance : float
        Luminance of the Gabor patch relative to background luminance.
        0. is dark, 1.0 is double the background luminance.

    central_rel_lum : float
        Luminance of the Gabor patch at the center of the RF relative to
        background luminance.
        0. is dark, 1.0 is double the background luminance.

    orientation : float
        The initial orientation of the Gabor patch.
        This changes orientation of the whole experiment, i.e. it also rotates
        the grid (because of the iso and cross configurations of movements).

    phase : float
        The phase of the Gabor patch.

    spatial_frequency : float
        The spatial freqency of the Gabor patch.

    rotations : int
        Number of different orientations at each given place.
        1 only one Gabor patch with initial orientation will be presented at
        given place, N>1 N different orientations will be presented,
        orientations are uniformly distributed between [0, 2*pi) + orientation.

    size : float
        Size of the tides. From this value the size of Gabor patch is derived
        so that it fits into a circle with diameter equal to this size.

        Gabor patch size is set so that sigma of Gaussian envelope is size/3

    x : float
        The x corrdinates of the central tide.

    y : float
        The y corrdinates of the central tide.

    stroke_time : float
        The duration of the first stroke of Gabor patch

    flash_duration : float
        The total duration of the presentation of Gabor patches. Therefore,
        the second stroke is presented for time equal:
            flash_duration - stroke_time

    duration : float
        The duration of single presentation of the stimulus.

    num_trials : int
        Number of trials each each stimulus is shown.

    circles : int
        Number of "circles" where the Gabor patch is presented.
        1: only at the central point the Gabor patch is presented,
        2: stimuli are presented at the central hexagonal tide and 6 hexes
        forming a "circle" around the central
        Trajectories starting or ending in the given number of circles are
        used, i.e. First Gabor patch can be out of the circles and vice versa.

    grid : bool
        If True hexagonal tiding with relative luminance 0 is drawn over the
        stimmuli.
        Mostly for testing purposes to check the stimuli are generated
        correctly.

    Note on hexagonal tiding:
    -------------------------
        Generating coordinates of centers of regular (!) hexagonal tidings.
        It is done this way, because the centers of tides are not on circles (!)
        First it generates integer indexed centers like this:
              . . .                (-2,2) (0, 2) (2,2)
             . . . .           (-3,1) (-1,1) (1,1) (3,1)
            . . . . .   ==> (-4,0) (-2,0) (0,0) (2,0) (4,0)     (circles=3)
             . . . .           (-3,-1)(-1,-1)(1,-1)(3,-1)
              . . .                (-2,-2)(0,-2)(2,-2)

        coordinates then multiplied by non-integer factor to get the right position
            x coordinate multiplied by factor 1/2*size
            y coordinate multiplied by factor sqrt(3)/2*size

    Note on central relative luminance:
    -----------------------------------
        In the experiment they had lower luminance for Gabor patches presented
        at the central tide


    Note on number of circles:
    --------------------------
        For 2 stroke the experiment includes also the trajectories that
        start inside the defined number of circles but get out as well as
        trajectories starting in the outside layer of tides comming inside.

        For example if we have number of circles = 2 -> that means we have
        central tide and the first circle of tides around, but for two stroke
        it is possible we start with Gabor patch at the distance 2 tides away
        from the central tide (i.e. tides that are in circles = 3) if we move
        inside and vice versa.

        This is solved by checking the distance of the final position of the
        Gabor patch, if the distance is bigger than a radius of a circle
        then opposite direction is taken into account.

        Since we have hexagonal tides this check is valid only for
        n <= 2/(2-sqrt(3)) ~ 7.5
        which is for given purposes satisfied, but should be mentioned.

    Note on rotations:
    ------------------
        This number is taken as a free parameter, but to replicate hexagonal
        tiding this number has to be 6 or 1 or 2. The code exploits symmetry and
        properties of the hexagonal tiding rather a lot!
        The ISO/CROSS configuration is determined from this number, so any other
        number generates moving paterns but in directions not matching hexes.
    """

    required_parameters = ParameterSet(
        {
            "relative_luminance": float,
            "central_rel_lum": float,
            "orientation": float,
            "phase": float,
            "spatial_frequency": float,
            "size": float,
            "flash_duration": float,
            "x": float,
            "y": float,
            "rotations": int,
            "duration": float,
            "num_trials": int,
            "circles": int,
            "stroke_time": float,
            "grid": bool,
        }
    )

    def generate_stimuli(self):
        # Assert explained in docstring
        assert self.parameters.circles < 7, "Too many circles, this won't work"
        if self.parameters.grid:
            # Grid is currently working only for special cases
            # Check if it is working
            assert self.parameters.orientation == 0.0, "Rotated grid is not implemented"
            assert self.parameters.x == 0, "X shift not yet implemented"
            assert self.parameters.y == 0, "Y shift not yet implemented"
            assert (
                self.model.visual_field.size_x == self.model.visual_field.size_y
            ), "Different sizes not yet implemented"

        for trial in range(0, self.parameters.num_trials):
            for rot in range(0, self.parameters.rotations):
                for row in range(
                    self.parameters.circles - 1, -self.parameters.circles, -1
                ):
                    colmax = 2 * self.parameters.circles - 2 - abs(row)
                    for column in range(-colmax, colmax + 1, 2):
                        for direction in (-1, 1):
                            # central coordinates of presented Gabor patch
                            # relative to the central tide
                            x = column * 0.5 * self.parameters.size
                            y = row * 0.5 * numpy.sqrt(3) * self.parameters.size
                            # rotation of the Gabor
                            angle = (
                                self.parameters.orientation
                                + numpy.pi * rot / self.parameters.rotations
                            )
                            if rot % 2 == 0:  # even rotations -> iso config
                                # Gabor orientation 0 -> horizontal
                                x_dir = numpy.cos(angle) * self.parameters.size
                                y_dir = numpy.sin(angle) * self.parameters.size
                            else:  # odd rotations -> cross config
                                # cross config means moving into perpendicular
                                # direction (aka + pi/2)
                                x_dir = -numpy.sin(angle) * self.parameters.size
                                y_dir = numpy.cos(angle) * self.parameters.size

                            # starting in the central tide
                            if x == 0 and y == 0:
                                first_rel_lum = self.parameters.central_rel_lum
                                second_rel_lum = self.parameters.relative_luminance
                            # ending in the central tide
                            elif (
                                abs(x + x_dir * direction) < self.parameters.size / 2.0
                            ) and (
                                abs(y + y_dir * direction) < self.parameters.size / 2.0
                            ):
                                first_rel_lum = self.parameters.relative_luminance
                                second_rel_lum = self.parameters.central_rel_lum
                            # far from the central tide
                            else:
                                first_rel_lum = self.parameters.relative_luminance
                                second_rel_lum = self.parameters.relative_luminance

                            # If the Gabor patch ends in outer circle
                            # we want also Gabor moving from outer circle to
                            # inner circles
                            # This condition is approximated by concentric
                            # circles more in docstring
                            outer_circle = numpy.sqrt(
                                (x + x_dir * direction) ** 2
                                + (y + y_dir * direction) ** 2
                            ) > ((self.parameters.circles - 1) * self.parameters.size)

                            # range here is 1 or 2
                            # In case of outer_circle == True generates two
                            # experiments, from and into the outer circle
                            # In case of outer_circle == False generates only
                            # one experiment
                            for inverse in range(1 + outer_circle):
                                self.stimuli.append(
                                    topo.TwoStrokeGaborPatch(
                                        frame_duration=self.frame_duration,
                                        duration=self.parameters.duration,
                                        flash_duration=self.parameters.flash_duration,
                                        size_x=self.model.visual_field.size_x,
                                        size_y=self.model.visual_field.size_y,
                                        background_luminance=self.background_luminance,
                                        first_relative_luminance=first_rel_lum,
                                        second_relative_luminance=second_rel_lum,
                                        orientation=angle,
                                        density=self.density,
                                        phase=self.parameters.phase,
                                        spatial_frequency=self.parameters.spatial_frequency,
                                        size=self.parameters.size,
                                        x=self.parameters.x
                                        + x
                                        + inverse * x_dir * direction,
                                        # inverse == 0 -> original start
                                        # inverse == 1 -> start from end
                                        y=self.parameters.y
                                        + y
                                        + inverse * y_dir * direction,
                                        location_x=0.0,
                                        location_y=0.0,
                                        trial=trial,
                                        stroke_time=self.parameters.stroke_time,
                                        x_direction=x_dir
                                        * direction
                                        * ((-1) ** inverse),
                                        # (-1)**inverse = 1 for original one
                                        # == -1 for the inverse movement
                                        y_direction=y_dir
                                        * direction
                                        * ((-1) ** inverse),
                                        grid=self.parameters.grid,
                                    )
                                )
                                # For the inverse movement we have to
                                # switch the luminances
                                first_rel_lum, second_rel_lum = (
                                    second_rel_lum,
                                    first_rel_lum,
                                )

    def do_analysis(self, data_store):
        pass


class MeasureGaborFlashDuration(VisualExperiment):
    """
    Experiment to measure the shortest duration of flashing a Gabor patch onto the RF of
    a neuron, which evokes a significant response. It tries flash durations in a range,
    and randomizes the order of durations.

    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------

    x : float
        x coordinate of the Gabor Patch

    y : float
        y coordinate of the Gabor Patch

    orientation : float
        Orientation of the Gabor Patch

    phase : float
        Phase of the Gabor Patch

    spatial_frequency : float
        Spatial frequency of the Gabor Patch

    sigma : float
        Standard deviation of the Gaussian in the Gabor Patch

    n_sigmas : float
        Number of standard deviations to which we sample the Gabor function.

    relative_luminance : float
        The scale of the stimulus. 0 is dark, 1.0 is double the background luminance.

    min_duration : float
        The minimum Gabor flash duration in ms

    max_duration : float
        The maximum Gabor flash duration in ms

    step : float
        Step size between flash durations.
        E.g. min_duration=1, max_duration=3.0, step=0.5 produces [1.0,1.5,2.0,2.5,3.0]

    blank_duration : float
        Duration of blank stimulus after Gabor patch presentation.

    neuron_id : int
        ID of measured neuron. Required to pair recordings to stimuli.

    num_trials : int
        Number of trials of showing the stimuli.
    """

    required_parameters = ParameterSet(
        {
            "x": float,
            "y": float,
            "orientation": float,
            "phase": float,
            "spatial_frequency": float,
            "sigma": float,
            "n_sigmas": float,
            "relative_luminance": float,
            "min_duration": float,
            "max_duration": float,
            "step": float,
            "blank_duration": float,
            "num_trials": int,
            "neuron_id": int,
        }
    )

    def generate_stimuli(self):
        common_params = {
            "size_x": self.model.visual_field.size_x,
            "size_y": self.model.visual_field.size_y,
            "location_x": 0.0,
            "location_y": 0.0,
            "background_luminance": self.background_luminance,
            "density": self.density,
            "frame_duration": self.frame_duration,
            "x": self.parameters.x,
            "y": self.parameters.y,
            "orientation": self.parameters.orientation,
            "phase": self.parameters.phase,
            "spatial_frequency": self.parameters.spatial_frequency,
            "size": 6 * self.parameters.sigma,
            "relative_luminance": self.parameters.relative_luminance,
            "grid": False,
            "neuron_id": self.parameters.neuron_id,
        }
        for trial in range(0, self.parameters.num_trials):
            trial_stims = [
                topo.SimpleGaborPatch(
                    flash_duration=flash_duration,
                    duration=flash_duration + self.parameters.blank_duration,
                    trial=trial,
                    **common_params
                )
                for flash_duration in np.arange(
                    self.parameters.min_duration,
                    self.parameters.max_duration
                    + 1,  # arange does not include final value
                    self.parameters.step,
                )
            ]
            random.shuffle(trial_stims)
            self.stimuli.extend(trial_stims)

    def do_analysis(self, data_store):
        pass


class CompareSlowVersusFastGaborMotion(VisualExperiment):
    """
    Present Gabor stimuli moving radially inward in either continuous or apparent
    motion.

    After each stimulus presentation, the center gabor patch is shown for an equal
    duration, to provide reference.

    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------

    x : float
        x coordinate of the Gabor Patch

    y : float
        y coordinate of the Gabor Patch

    orientation : float
        Orientation of the Gabor Patch

    phase : float
        Phase of the Gabor Patch

    spatial_frequency : float
        Spatial frequency of the Gabor Patch

    sigma : float
        Standard deviation of the Gaussian in the Gabor Patch

    n_sigmas : float
        Number of standard deviations to which we sample the Gabor function.

    center_relative_luminance : float
        The scale of the center stimulus.
        0 is dark, 1.0 is double the background luminance.

    surround_relative_luminance : float
        The scale of the surrounding stimuli.
        0 is dark, 1.0 is double the background luminance.

    movement_speeds : list(float)
        List of speeds to present with which the Gabor patch moves. For apparent motion,
        the flash duration is adjusted according to the patch radius, to match these
        speeds.

    angles : list
        List of angles from which the Gabor patches approach the center.

    moving_gabor_orientation_radial : bool
        If True, the orientation of the surround gabors is radial, otherwise tangential.

    n_circles : int
        Number of eccentricities at which Gabor patches are flashed. The movement range
        of the continuously moving Gabor is set to cover an equivalent distance.

    blank_duration : float
        Duration of blank stimulus after Gabor patch presentation.

    neuron_id : int
        ID of measured neuron. Required to pair recordings to stimuli.

    num_trials : int
        Number of trials of showing the stimuli.
    """

    required_parameters = ParameterSet(
        {
            "x": float,
            "y": float,
            "orientation": float,
            "phase": float,
            "spatial_frequency": float,
            "sigma": float,
            "n_sigmas": float,
            "center_relative_luminance": float,
            "surround_relative_luminance": float,
            "movement_speeds": list,
            "angles": list,
            "moving_gabor_orientation_radial": bool,
            "blank_duration": float,
            "n_circles": int,
            "neuron_id": int,
            "num_trials": int,
        }
    )

    def generate_stimuli(self):
        logger = mozaik.getMozaikLogger()
        common_params = {
            "size_x": self.model.visual_field.size_x,
            "size_y": self.model.visual_field.size_y,
            "location_x": 0.0,
            "location_y": 0.0,
            "background_luminance": self.background_luminance,
            "density": self.density,
            "frame_duration": self.frame_duration,
            "x": self.parameters.x,
            "y": self.parameters.y,
            "orientation": self.parameters.orientation,
            "phase": self.parameters.phase,
            "spatial_frequency": self.parameters.spatial_frequency,
            "sigma": self.parameters.sigma,
            "n_sigmas": self.parameters.n_sigmas,
            "center_relative_luminance": self.parameters.center_relative_luminance,
            "neuron_id": self.parameters.neuron_id,
        }

        am_specific_params = {
            "surround_relative_luminance": self.parameters.surround_relative_luminance,
            "surround_gabor_orientation_radial": self.parameters.moving_gabor_orientation_radial,
        }
        cont_mov_specific_params = {
            "moving_relative_luminance": self.parameters.surround_relative_luminance,
            "moving_gabor_orientation_radial": self.parameters.moving_gabor_orientation_radial,
        }
        center_specific_params = {
            "size": 2 * common_params["sigma"] * common_params["n_sigmas"],
            "relative_luminance": common_params["center_relative_luminance"],
        }

        am_params = common_params.copy()
        am_params.update(am_specific_params)
        cont_mov_params = common_params.copy()
        cont_mov_params.update(cont_mov_specific_params)
        center_params = common_params.copy()
        center_params.update(center_specific_params)
        del center_params["sigma"]
        del center_params["center_relative_luminance"]

        for trial in range(0, self.parameters.num_trials):
            for speed in self.parameters.movement_speeds:
                gabor_diameter = 2.0 * self.parameters.sigma * self.parameters.n_sigmas
                flash_duration = gabor_diameter / speed * 1000
                flash_duration_rounded = (
                    np.round(flash_duration / self.frame_duration) * self.frame_duration
                )
                if trial == 0:
                    logger.info(
                        "CompareSlowVersusFastGaborMotion: Calculated flash duration %.2f ms, must be integer multiple of the frame duration %.2f ms, modifiying it to %.2f"
                        % (flash_duration, self.frame_duration, flash_duration_rounded)
                    )
                flash_duration = flash_duration_rounded
                assert (
                    flash_duration >= self.frame_duration
                ), "Gabor flash duration must be at least as long as the frame duration"
                stim_duration = self.parameters.blank_duration + flash_duration * (
                    self.parameters.n_circles + 1
                )
                am_params["duration"] = stim_duration
                cont_mov_params["duration"] = stim_duration
                for angle in self.parameters.angles:
                    # Apparent Motion
                    am_stim = topo.RadialGaborApparentMotion(
                        flash_duration=flash_duration,
                        start_angle=angle,
                        end_angle=angle,
                        n_gabors=1,
                        n_circles=self.parameters.n_circles,
                        symmetric=False,
                        random=False,
                        flash_center=True,
                        centrifugal=False,
                        trial=trial,
                        **am_params
                    )

                    # Center-only stimulation
                    co_stim = topo.SimpleGaborPatch(
                        duration=stim_duration,
                        flash_duration=flash_duration,
                        trial=trial,
                        **center_params
                    )

                    # Continuous Motion
                    cont_mov_stim = topo.ContinuousGaborMovementAndJump(
                        movement_duration=flash_duration
                        * (self.parameters.n_circles - 1),
                        movement_length=gabor_diameter
                        * (self.parameters.n_circles - 1),
                        movement_angle=angle,
                        center_flash_duration=flash_duration,
                        trial=trial,
                        **cont_mov_params
                    )

                    # Center-only stimulation
                    co_stim_0 = topo.SimpleGaborPatch(
                        duration=stim_duration,
                        flash_duration=flash_duration,
                        trial=trial,
                        **center_params
                    )

                    self.stimuli.append(am_stim)
                    self.stimuli.append(co_stim)
                    self.stimuli.append(cont_mov_stim)
                    self.stimuli.append(co_stim_0)

    def do_analysis(self, data_store):
        pass


class RunApparentMotionConfigurations(VisualExperiment):
    """
    Apparent motion stimulus configurations from Benoit Le Bec, 2018.

    Benoit Le Bec, Lateral connectivity: propagation of network belief and
    hallucinatory-like states in the primary visual cortex.

    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------

    x : float
        x coordinate of the Gabor Patch

    y : float
        y coordinate of the Gabor Patch

    orientation : float
        Orientation of the Gabor Patch

    phase : float
        Phase of the Gabor Patch

    spatial_frequency : float
        Spatial frequency of the Gabor Patch

    sigma : float
        Standard deviation of the Gaussian in the Gabor Patch

    n_sigmas : float
        Number of standard deviations to which we sample the Gabor function.

    center_relative_luminance : float
        The scale of the center stimulus.
        0 is dark, 1.0 is double the background luminance.

    surround_relative_luminance : float
        The scale of the surrounding stimuli.
        0 is dark, 1.0 is double the background luminance.

    flash_duration : float
        Apparent motion flash duration

    blank_duration : float
        Blank duration after each stimulus

    configurations : list(string)
        List of configurations to present. The names and shapes of the configurations
        are described above.

    n_circles : int
        Number of eccentricities at which Gabor patches are flashed.

    random_order : bool
        Randomize the order of configurations between trials or not.

    flash_center : bool
        Flash in Gabor patch in the center or not.

    neuron_id : int
        ID of measured neuron. Required to pair recordings to stimuli.

    num_trials : int
        Number of trials of showing the stimuli.


    Possible configurations
    -----------------------

    SECTOR_ISO

       \ | /       .              .
                   .     \|/      .
                   .              .    |
                   .     /|\      .
       / | \       .              .

    SECTOR_CROSS

    /          \   .              .
                   .    /    \    .
   |            |  .   |      |   .    |
                   .    \    /    .
    \          /   .              .

    SECTOR_CF

                   .              .  \ | /
                   .     \|/      .
         |         .              .
                   .     /|\      .
                   .              .  / | \

    SECTOR_RND

      \            .      | /     .
       \           .      |/      .
                   .              .    |
         |         .     / \      .
       /   \       .      |       .

    FULL_ISO

       \ | /       .              .
     \       /     .    \ | /     .
   --         --   .   --   --    .    |
     /       \     .    / | \     .
       / | \       .              .

    FULL_CROSS
         __
      /      \
    /          \   .      __      .
                   .    /    \    .
   |            |  .   |      |   .    |
                   .    \ __ /    .
    \          /   .              .
      \  __  /

    FULL_RND

         |         .     \   /    .
     \     /       .     \ |   /  .
   --      -- --   .    --        .    |
     / / | \       .           \  .
       /   \       .       |      .

    CENTER_ONLY



        |



    """

    required_parameters = ParameterSet(
        {
            "x": float,
            "y": float,
            "orientation": float,
            "phase": float,
            "spatial_frequency": float,
            "sigma": float,
            "n_sigmas": float,
            "center_relative_luminance": float,
            "surround_relative_luminance": float,
            "flash_duration": float,
            "configurations": list,
            "n_circles": int,
            "random_order": bool,
            "flash_center": bool,
            "blank_duration": float,
            "neuron_id": int,
            "num_trials": int,
        }
    )

    def generate_stimuli(self):
        common_params = {
            "size_x": self.model.visual_field.size_x,
            "size_y": self.model.visual_field.size_y,
            "location_x": 0.0,
            "location_y": 0.0,
            "background_luminance": self.background_luminance,
            "density": self.density,
            "frame_duration": self.frame_duration,
            "flash_duration": self.parameters.flash_duration,
            "x": self.parameters.x,
            "y": self.parameters.y,
            "orientation": self.parameters.orientation,
            "phase": self.parameters.phase,
            "spatial_frequency": self.parameters.spatial_frequency,
            "sigma": self.parameters.sigma,
            "center_relative_luminance": self.parameters.center_relative_luminance,
            "surround_relative_luminance": self.parameters.surround_relative_luminance,
            "n_circles": self.parameters.n_circles,
            "surround_gabor_orientation_radial": True,
            "random": False,
            "centrifugal": False,
            "symmetric": True,
            "flash_center": self.parameters.flash_center,
            "duration": self.parameters.blank_duration + self.parameters.flash_duration
            # Center & 1 flash_duration blank
            * (self.parameters.n_circles + 1),
            "neuron_id": self.parameters.neuron_id,
        }

        valid_configs = [
            "SECTOR_ISO",
            "SECTOR_CROSS",
            "SECTOR_CF",
            "SECTOR_RND",
            "FULL_ISO",
            "FULL_CROSS",
            "FULL_RND",
            "CENTER_ONLY",
        ]
        for trial in range(0, self.parameters.num_trials):
            trial_stims = []
            for configuration in self.parameters.configurations:
                assert (
                    configuration in valid_configs
                ), "Configuration %s not in the set of valid configurations %s" % (
                    configuration,
                    valid_configs,
                )
                params = common_params.copy()
                c1, c2 = configuration.split("_")
                if c1 == "CENTER":
                    assert (
                        self.parameters.flash_center == True
                    ), "Cannot have center only stimulation without stimulating the center"
                    c1_params = {
                        "start_angle": 0,
                        "end_angle": 0,
                        "n_gabors": 0,
                        "n_circles": 0,
                    }
                elif c1 == "SECTOR":
                    c1_params = {
                        "start_angle": self.parameters.orientation - np.pi / 6.0,
                        "end_angle": self.parameters.orientation + np.pi / 6.0,
                        "n_gabors": 3,
                    }
                elif c1 == "FULL":
                    c1_params = {
                        "start_angle": self.parameters.orientation,
                        "end_angle": self.parameters.orientation + np.pi * 5.0 / 6.0,
                        "n_gabors": 6,
                    }
                params.update(c1_params)

                if c2 == "CROSS":
                    params["surround_gabor_orientation_radial"] = False
                    params["start_angle"] += np.pi / 2
                    params["end_angle"] += np.pi / 2
                elif c2 == "RND":
                    params["random"] = True
                elif c2 == "CF":
                    params["centrifugal"] = True
                params["identifier"] = configuration
                trial_stims.append(
                    topo.RadialGaborApparentMotion(trial=trial, **params)
                )

            if self.parameters.random_order:
                random.shuffle(trial_stims)

            self.stimuli.extend(trial_stims)

    def do_analysis(self, data_store):
        pass
