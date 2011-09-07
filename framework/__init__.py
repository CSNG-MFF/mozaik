import logging

logger = logging.getLogger("MozaikLite")

def load_component(path):
    path_parts = path.split('.')
    module_name = ".".join(path_parts[:-1])
    class_name = path_parts[-1]
    print module_name
    _module = __import__(module_name, globals(), locals(), [class_name], -1)
    logger.info("Loaded component %s from module %s" % (class_name, module_name))
    return getattr(_module, class_name)