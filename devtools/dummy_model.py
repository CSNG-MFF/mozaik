class DummyModel:
    """
    Dummy model to be able to create experiments without using any actual model,
    for example for visualization.
    """

    def __init__(
        self, density, background_luminance, frame_duration, size_x, size_y, **kwargs
    ):
        self.visual_field = DummyObject()
        self.visual_field.size_x = size_x
        self.visual_field.size_y = size_y
        self.input_space = DummyObject(1)
        self.input_space.background_luminance = background_luminance
        self.input_space.parameters.update_interval = frame_duration
        self.input_layer = DummyObject(2)
        self.input_layer.parameters.receptive_field.spatial_resolution = 1 / density


class DummyObject:
    """
    Optionally recursive dummy object, to mimic specific structure of the model, 
    such as input_layer.parameters.receptive_field.spatial_resolution
    """

    def __init__(self, recursion_level=0):
        self.size_x = 0
        self.size_y = 0
        self.background_luminance = 0
        self.parameters = self.recursion(recursion_level)
        self.update_interval = 0
        self.receptive_field = self.recursion(recursion_level)
        self.spatial_resolution = 0

    def recursion(self, level):
        if level:
            return DummyObject(recursion_level=level - 1)
