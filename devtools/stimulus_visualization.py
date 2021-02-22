import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import time


def show_experiment(experiment, merge_stimuli=True, frame_delay=None, grid=None):
    """
    Shows visual experiment stimuli either as a single animation or as a series
    of animations, one by one. The duration of each stimulus is set by their duration
    parameter.

    Parameters
    ----------

    experiment : VisualExperiment instance from mozaik.experiments.vision
    merge_stimuli : bool
                     Merge stimuli into a single animation or show them one by one
    frame_delay : float (ms)
                  How long to show each frame. Defaults to the frame_duration field of
                  each stimulus in the experiment. Does not change how many frames are
                  plotted.
    grid : None or int
           Plot a red grid with "grid" ticks for each visual angle. Disabled by default.
    """

    if merge_stimuli:
        # Check that parameters relevant for plotting (visual field size, etc.)
        # are equal across stimuli
        params_0 = get_stimulus_params(experiment.stimuli[0])
        for stimulus in experiment.stimuli:
            params = get_stimulus_params(stimulus)
            fields = ["size_x", "size_y", "location_x", "location_y", "frame_duration"]
            for field in fields:
                assert_message = (
                    field
                    + ' must be equal for all stimuli in the experiment to visualize them as a single animation. Set "merge_stimuli=False" to visualize such stimuli.'
                )
                assert params_0[field] == params[field], assert_message

        frames = []
        duration = 0.0
        for stimulus in experiment.stimuli:
            # Construct dictionary from stimulus parameters
            params = get_stimulus_params(stimulus)
            duration = duration + params["duration"]
            frames.extend(
                pop_frames(
                    stimulus, num_frames=params["duration"] / params["frame_duration"]
                )
            )
        if frame_delay is None:
            frame_delay = params_0["frame_duration"]
        show_frames(frames, params_0, True, grid, frame_delay, False)
    else:
        for stimulus in experiment.stimuli:
            show_stimulus(stimulus, None, frame_delay, grid, True, False)


def show_stimulus(
    stimulus, duration=None, frame_delay=None, grid=None, animate=True, repeat=True
):
    """
    Shows the stimulus as an animation, for a specified duration. Parameters
    of the extent of visual space, etc. are extracted from the stimulus instance.

    Parameters
    ----------

    stimulus : VisualStimulus instance from mozaik.stimuli.vision.visual_stimulus
    duration : float (ms)
               Stimulus plot duration in ms - we plot duration / frame_duration frames,
               where frame_duration is a field of the stimulus instance.
               Defaults to the duration field of the stimulus instance
    frame_delay : float (ms)
                  How long to show each frame. Defaults to the frame_duration field of
                  the stimulus instance. Does not change how many frames are plotted.
    grid : None or int
           Plot a red grid with "grid" ticks for each visual angle. Disabled by default.
    animate : bool
              Show a fluid animation of the stimulus, or each frame one by one.
    """
    # Construct dictionary from stimulus parameters
    params = get_stimulus_params(stimulus)
    if duration is None:
        duration = params["duration"]
    if frame_delay is None:
        frame_delay = params["frame_duration"]
    frames = pop_frames(stimulus, num_frames=duration / params["frame_duration"])
    show_frames(frames, params, animate, grid, frame_delay, repeat)


def show_frame(frame, params=None, grid=None):
    """
    Plot a frame of the stimulus.

    Parameters
    ----------

    frame : numpy ndarray
            The frame to plot
    params : dictionary of stimulus parameters. These parameters are given to stimulus
             instance on instantiation. The fields used for plotting are: size_x,
             size_y, location_x, location_y
    grid : None or int
           Plot a red grid with "grid" ticks for each visual angle. Disabled by default.
    """
    frame = frame.astype(float)
    if params is not None:
        plt.xlabel("x/$^\circ$")
        plt.ylabel("y/$^\circ$")
    plot_colorbar(plot_frame(frame, params, grid)[0])
    plt.gcf().canvas.set_window_title("")
    plt.show()


def show_frames(
    frames, params=None, animate=True, grid=None, frame_delay=16, repeat=True
):
    """
    Show a list of frames as an animation.

    Parameters
    ----------

    frames : list(numpy ndarray)
             List of frames to plot
    params : dictionary of stimulus parameters. These parameters are given to stimulus
             instance on instantiation. The fields used for plotting are: size_x,
             size_y, location_x, location_y, frame_duration
    animate : bool
              Show a fluid animation of the stimulus, or each frame one by one.
    grid : None or int
           Plot a red grid with "grid" ticks for each visual angle. Disabled by default.
    frame_delay : float (ms)
                  How long to show each frame. Defaults to 16ms, which is roughly
                  equivalent to 60Hz (framerate of most monitors nowadays)
    """

    vmin = np.array(frames).min()
    vmax = np.array(frames).max()

    if animate:
        fig = plt.figure(1)
        mpl.colors.Normalize(vmin=vmin, vmax=vmin)
        plot_colorbar(plot_frame(frames[0], params, grid, vmin, vmax)[0])
        blit_on = not grid  # Grid does not work with blit on
        if params is not None:
            plt.xlabel("x/$^\circ$")
            plt.ylabel("y/$^\circ$")

        ani = FuncAnimation(
            fig,
            func=plot_frame,
            frames=frames,
            interval=frame_delay,
            blit=blit_on,
            fargs=(params, grid, vmin, vmax),
            repeat=repeat,
        )

        try:  # Catch exception that crashes program on window close
            plt.show(block=True)
        except AttributeError:
            pass

        plt.close()
    else:
        for frame in frames:
            show_frame(frame, params, grid)


def get_stimulus_params(stimulus):
    """
    Returns the parameters with which the stimulus was initialized as a dictionary
    """
    return {
        key: getattr(stimulus, key)
        for key in vars(stimulus)["expanded_paramset_params_dict"]
    }


def pop_frames(stimulus, num_frames):
    return [stimulus._frames.next()[0] for i in range(num_frames)]


def get_xlim(params):
    """
    Retrieve minimal and maximal x values from parameters
    """
    return (
        params["location_x"] - params["size_x"] / 2.0,
        params["location_x"] + params["size_x"] / 2.0,
    )


def get_ylim(params):
    """
    Retrieve minimal and maximal y values from parameters
    """
    return (
        params["location_y"] - params["size_y"] / 2.0,
        params["location_y"] + params["size_y"] / 2.0,
    )


def plot_colorbar(pcm):
    cbar = plt.colorbar(pcm)
    cbar.ax.get_yaxis().labelpad = 20
    cbar.ax.set_ylabel("luminance (cd)", rotation=270)


# Ugly global variable to measure framerate inside animation
time_last = 0


def plot_frame(frame, params=None, grid=None, vmin=None, vmax=None):
    # Get minimal x and y values of the plot
    if params is not None:
        xmin, xmax = get_xlim(params)
        ymin, ymax = get_ylim(params)
    else:
        xmin, xmax = 0, frame.shape[0]
        ymin, ymax = 0, frame.shape[1]

    # Grid does not work with blit & this speeds up plotting without blit
    if grid is not None:
        plt.cla()

    p_frame = np.flip(frame, 0)  # flip image so origin is in lower left corner

    # Plot frame as a colormesh
    x = np.linspace(xmin, xmax, frame.shape[0])
    y = np.linspace(ymin, ymax, frame.shape[1])
    pcm = plt.pcolormesh(x, y, p_frame, cmap="gray", vmin=vmin, vmax=vmax)

    # Measure fps and show in title
    global time_last
    time_now = time.time()
    duration = time_now - time_last
    time_last = time_now
    params_duration_str = (  # Print what the framerate should be
        "" if params is None else " (in params %d ms)" % params["frame_duration"]
    )
    plt.gcf().canvas.set_window_title(
        "Frame duration: %3d ms%s, fps: %6.2f"
        % (duration * 1000, params_duration_str, 1 / duration)
    )

    # Set ticks given stimulus parameters and set equal axis
    if params is not None:
        xmin, xmax = get_xlim(params)
        ymin, ymax = get_ylim(params)
        tick_step = 1.0 if grid is None else 1.0 / grid
        plt.xticks(np.arange(round(xmin), round(xmax), 1))
        plt.yticks(np.arange(round(ymin), round(ymax), 1))
        plt.gca().set_xticks(np.arange(round(xmin), round(xmax), tick_step), "minor")
        plt.gca().set_yticks(np.arange(round(ymin), round(ymax), tick_step), "minor")
    plt.axis("equal")

    # Draw grid
    if grid is not None:
        plt.grid(True, color="r", which="minor")

    return (pcm,)
