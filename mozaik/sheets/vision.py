# encoding: utf-8
"""
This module contains implementation of vision related sheets.
"""

import numpy
import mozaik
from parameters import ParameterSet
from pyNN import space
from pyNN.errors import NothingToWriteError
from mozaik.sheets import Sheet
        
logger = mozaik.getMozaikLogger()

class RetinalUniformSheet(Sheet):
    """
    Retinal sheet corresponds to a sheet of retinal cells (retinal ganglion cells or photoreceptors). 
    It implicitly assumes the coordinate systems is in degress in visual field.
    
    Other parameters
    ----------------
    
    sx : float (degrees)
       X size of the region.
        
    sy : float (degrees)
       Y size of the region.

    density : int
            Number of neurons along both axis.
    """
    required_parameters = ParameterSet({
        'sx': float,  # degrees, x size of the region
        'sy': float,  # degrees, y size of the region
        'density': int,  # neurons along each axis
    })
    
    def __init__(self, model, parameters):
        Sheet.__init__(self, model,parameters.sx, parameters.sy, parameters)
        logger.info("Creating %s with %d neurons." % (self.__class__.__name__, int(parameters.sx * parameters.sy * parameters.density)))
        rs = space.RandomStructure(boundary=space.Cuboid(self.size_x,self.size_y, 0),
                                   origin=(0.0, 0.0, 0.0),
                                   rng=mozaik.pynn_rng)
        
        #rs = space.Grid2D(aspect_ratio=1, dx=parameters.sx/parameters.density, dy=parameters.sy/parameters.density, x0=-parameters.sx/2,y0=-parameters.sy/2,z=0.0)
        
        self.pop = self.sim.Population(int(parameters.sx * parameters.sy * parameters.density),
                                            getattr(self.model.sim, self.parameters.cell.model),
                                            self.parameters.cell.params,
                                            structure=rs,
                                            initial_values=self.parameters.cell.initial_values,
                                            label=self.name)
        
        # Forces PyNN to generate the positions to ensure the reproducibility with multiprocessing
        self.pop.positions

    def size_in_degrees(self):
        return (self.parameters.sx, self.parameters.sy)


class SheetWithMagnificationFactor(Sheet):
    """
    A Sheet that has a magnification factor corresponding to cortical visual area.
    It interprets the coordinates system to be in degrees of visual field, but it allows
    for definition of the layer using parameters in cortical space. It offers 
    number of functions that facilitate conversion between the underlying visual degree
    coordinates and cortical space coordinate systems using the magnification factor. 
    
    Other parameters
    ----------------
    magnification_factor : float (μm/degree)
                         The magnification factor.
    
    sx : float (μm)
       X size of the region.
        
    sy : float (μm)
       Y size of the region.
    """
    required_parameters = ParameterSet({
        'magnification_factor': float,  # μm / degree
        'sx': float,      # μm, x size of the region
        'sy': float,      # μm, y size of the region
    })

    def __init__(self, model, parameters):
        """
        """
        logger.info("Creating %s with %d neurons." % (self.__class__.__name__, int(parameters.sx*parameters.sy/1000000*parameters.density)))
        Sheet.__init__(self, model, parameters.sx/ parameters.magnification_factor,parameters.sy/parameters.magnification_factor,parameters)
        self.magnification_factor = parameters.magnification_factor

    def vf_2_cs(self, degree_x, degree_y):
        """
        vf_2_cs converts the position (degree_x, degree_y) in visual field to
        position in cortical space (in μm) given the magnification_factor.
        
        Parameters
        ----------
        degree_x : float (degrees)
                 X coordinate of the position in degrees of visual field
        degree_y : float (degrees)
                 Y coordinate of the position in degrees of visual field
        
        Returns
        -------
        microm_meters_x,microm_meters_y : float,float (μm,μm)
                                          Tuple with the coordinates in cortical space (μm)
        
        """
        return (degree_x * self.magnification_factor,
                degree_y * self.magnification_factor)

    def cs_2_vf(self, micro_meters_x, micro_meters_y):
        """
        cs_2_vf converts the position (micro_meters_x, micro_meters_y) in
        cortical space to the position in the visual field (in degrees) given
        the magnification_factor
        
        Parameters
        ----------
        micro_meters_x : float (μm)
                 X coordinate of the position in μm of cortical space
        micro_meters_y : float (μm)
                 Y coordinate of the position in μm of cortical space
        
        Returns
        -------
        degrees_x,degrees_y : float,float (degrees,degrees)
                                          Tuple with the coordinates in visual space (degrees)
        """
        return (micro_meters_x / self.magnification_factor,
                micro_meters_y / self.magnification_factor)

    def dvf_2_dcs(self, distance_vf):
        """
        dvf_2_dcs converts the distance in visual space to the distance in
        cortical space given the magnification_factor
        
        Parameters
        ----------
        distance_vf : float (degrees)
                 The distance in visual field coordinates (degrees).
                 
        Returns
        -------
        distance_cs : float (μm)
                    Distance in cortical space.
        """
        return distance_vf * self.magnification_factor

    def size_in_degrees(self):
        """
        Returns the size of the sheet in cortical space (μm).
        """
        return self.cs_2_vf(self.parameters.sx, self.parameters.sy)


class VisualCorticalUniformSheet(SheetWithMagnificationFactor):
    """
    Represents a visual cortical sheet of neurons, randomly uniformly distributed in cortical space.
    
    Other parameters
    ----------------
    density : float (neurons/mm^2)
            The density of neurons per square milimeter.
    """
    
    required_parameters = ParameterSet({
        'density': float,  # neurons/(mm^2)
    })

    def __init__(self, model, parameters):
        SheetWithMagnificationFactor.__init__(self, model, parameters)
        dx, dy = self.cs_2_vf(parameters.sx, parameters.sy)
        rs = space.RandomStructure(boundary=space.Cuboid(dx, dy, 0),
                                   origin=(0.0, 0.0, 0.0),
                                   rng=mozaik.pynn_rng)


        if self.parameters.cell.model in set(["aeif_cond_alpha_multisynapse","aeif_cond_beta_multisynapse"]):
            self.multisynapse = True
            receptors= {}
            for (k,v) in self.parameters.cell.receptors.items():
                #name.append(v['name'])
                receptors[k] = getattr(self.model.sim, v.name)(**v.params)
                
                
            celltype = self.sim.PointNeuron(
                self.sim.AdExp(**self.parameters.cell.params),
                                **receptors)
                
                
            self.pop = self.sim.Population(int(parameters.sx * parameters.sy/1000000 * parameters.density), 
                                            celltype,structure=rs, initial_values=self.parameters.cell.initial_values,
                                            label= self.name)    
        
        else:
            self.pop = self.sim.Population(int(parameters.sx * parameters.sy/1000000 * parameters.density),
                                            getattr(self.model.sim, self.parameters.cell.model),
                                            self.parameters.cell.params,
                                            structure=rs,
                                            initial_values=self.parameters.cell.initial_values,
                                            label=self.name)

        '''self.pop = self.sim.Population(int(parameters.sx*parameters.sy/1000000*parameters.density),
                                       getattr(self.model.sim, self.parameters.cell.model),
                                       self.parameters.cell.params,
                                       structure=rs,
                                       initial_values=self.parameters.cell.initial_values,
                                       label=self.name)'''
        # Forces PyNN to generate the positions to ensure the reproducibility with multiprocessing
        self.pop.positions


class VisualCorticalUniformSheet3D(VisualCorticalUniformSheet):
    """
    Represents a visual cortical sheet of neurons, randomly uniformly distributed in cortical space.
    In addition to the VisualCorticalUniformSheet it adds 3rd dimension to the neurons that corresponds their depth 
    within cortical sheet (prepandicular to the cortical surface). 
    In the third dimensions, the neurons will be uniformly distributed between the *min_depth* and *max_depth* parameters.
    
    Notes
    -----
    Manny existing Mozaik components that take neural position into consideration will 
    ignore this 3rd dimension. Also unlike the first to dimensions, corresponding to the axis along
    the cortical surface, the third depth dimension is in μm!

    Also note the density is still calculated only per surface unit.
    
    Other parameters
    ----------------
    min_depth : float (μm)
            The mininmum depth of neurons.
    max_depth : float (μm)
            The maxinmum depth of neurons.

    """
    
    required_parameters = ParameterSet({
        'min_depth': float,  # μm
        'max_depth': float,  # μm
    })

    def __init__(self, model, parameters):
        SheetWithMagnificationFactor.__init__(self, model, parameters)
        dx, dy = self.cs_2_vf(parameters.sx, parameters.sy)

        origin_z = (self.parameters.min_depth + self.parameters.max_depth)/2.0
        width_z = (self.parameters.max_depth - self.parameters.min_depth)

        rs = space.RandomStructure(boundary=space.Cuboid(dx, dy, width_z),
                                   origin=(0.0, 0.0, origin_z),
                                   rng=mozaik.pynn_rng)

        self.pop = self.sim.Population(int(parameters.sx*parameters.sy/1000000*parameters.density),
                                       getattr(self.model.sim, self.parameters.cell.model),
                                       self.parameters.cell.params,
                                       structure=rs,
                                       initial_values=self.parameters.cell.initial_values,
                                       label=self.name)
        # Forces PyNN to generate the positions to ensure the reproducibility with multiprocessing
        self.pop.positions
