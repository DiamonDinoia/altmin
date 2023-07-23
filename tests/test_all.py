import test_layers
import test_losses
import test_derivatives
import test_altmin_functions 
import unittest 
from altmin import simpleNN, get_mods, Flatten, LeNet, load_dataset
import torch.nn as nn 
import fast_altmin


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromModule(test_layers)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromModule(test_losses)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromModule(test_derivatives)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromModule(test_altmin_functions)
    unittest.TextTestRunner(verbosity=2).run(suite)
