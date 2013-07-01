import mozaik
import time

logger = mozaik.getMozaikLogger()


class timeit(object):
    """
    Decorator that times the execution time of the function it wraps and
    writes this time to the log.
    
    Parameters
        f : func
          The function to time
    """

    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        start_time = time.time()
        result = self.f(*args, **kwargs)
        #logging.debug("Execution time for %s: %g s" % (self.f.__name__, time.time()-start_time))
        logger.info("Execution time for %s: %g s" % (self.f.__name__, time.time() - start_time))
        return result
