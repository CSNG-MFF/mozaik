import unittest

"""
1. test the values supplied as parameters
    required_parameters = ParameterSet({
        'name': str,
        'results_dir': str,
        'store_stimuli' : bool,
        'reset': bool,
        'null_stimulus_period': float,
        'input_space': ParameterSet, # can be none - in which case input_space_type is ignored
        'input_space_type': str,  # defining the type of input space, visual/auditory/... it is the class path to the class representing it
    })

"""

class TestModel(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
