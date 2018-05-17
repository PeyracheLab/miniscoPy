#!/usr/bin/env python3

import unittest

class CTestImportPackages(unittest.TestCase):
    def test_import(self):
        self.assertTrue(self.is_import_OK())
    #
    def is_import_OK(self):
        try: import scipy
        except ImportError:
            print("\nThe NumPy package is missing. Please install it.")
            return False
        #
        try: import numpy
        except ImportError:
            print("\nThe SciPy package is missing. Please install it.")
            return False
        #
        try: import av
        except ImportError:
            print("\nERROR: The PyAV package is missing. Please install it.")
            return False
        #
        try: import cv2
        except ImportError:
            print("\nERROR: The OpenCV package is missing. Please install it.")
            return False
        #
        try: import yaml
        except ImportError:
            print("\nERROR: The PyYAML package is missing. Please install it.")
            return False
        #
        try: import pandas
        except ImportError:
            print("\nERROR: The Pandas package is missing. Please install it.")
            return False
        #
        try: import h5py
        except ImportError:
            print("\nERROR: The H5py package is missing. Please install it.")
            return False
        #
        try: import tqdm
        except ImportError:
            print("\nERROR: The Tqdm package is missing. Please install it.")
            return False
        #
        try: import skimage
        except ImportError:
            print("\nERROR: The PyYAML package is missing. Please install it.")
            return False
        #
        try: import sklearn
        except ImportError:
            print("\nERROR: The scikit-learn package is missing. Please install it.")
            return False
        #
        return True
    #
#

if __name__ == '__main__':
    unittest.main()
#
