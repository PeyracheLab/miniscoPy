#!/usr/bin/env python3

import unittest

class CTestNumpyDeadlockSingleProc(unittest.TestCase):
    def test_deadlock_sp(self):
        # for additional information regarding this issue see:
        # https://github.com/numpy/numpy/issues/11041
        # https://github.com/numpy/numpy/issues/4813
        # https://github.com/numpy/numpy/issues/654
        # 
        import numpy as np
        print("\nINFO: this test should be done in less than a minute...")
        n = 1024
        tmp1 = np.identity(n)
        tmp2 = np.linalg.inv(tmp1)
        print("INFO: Done! The test PASSED.")
    #
#

if __name__ == '__main__':
    unittest.main()
#
