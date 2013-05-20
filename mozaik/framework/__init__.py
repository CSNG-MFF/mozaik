"""
The root framework module. 
"""

import logging

logger = logging.getLogger("mozaik")


def load_component(path):
    """
    This function loads a model component (represented by a class instance) located with the path varialble.
    
    Parameters
    ----------
        path : str
             The path to the module containing the component.   
             
    Returns
    -------
        component : object
                  The instance of the component class
    
    Note
    ----
    This function is primarily used to automatically load components based on configuration files during model construction.
    """
    path_parts = path.split('.')
    module_name = ".".join(path_parts[:-1])
    class_name = path_parts[-1]
    _module = __import__(module_name, globals(), locals(), [class_name], -1)
    logger.info("Loaded component %s from module %s" % (class_name, module_name))
    return getattr(_module, class_name)
