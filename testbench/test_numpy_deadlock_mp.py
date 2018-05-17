#!/usr/bin/env python3

import unittest
import multiprocessing as mp
import numpy as np

def target_func(n):
    tmp1 = np.identity(n)
    tmp2 = np.linalg.inv(tmp1)
#

class CTestNumpyDeadlockMultiProc(unittest.TestCase):
    def test_deadlock_mp(self):
        # for additional information regarding this issue see:
        # https://github.com/numpy/numpy/issues/11041
        # https://github.com/numpy/numpy/issues/4813
        # https://github.com/numpy/numpy/issues/654
        # 
        print("\nINFO: this test should be done in less than a minute...")
        print("INFO: Numpy version: %s" % np.version.full_version)
        num_of_proc = mp.cpu_count()
        target_args = [1024] * num_of_proc
        pool = mp.Pool(num_of_proc)
        pool.map(target_func, target_args)
        print("INFO: Done! The test PASSED.")
    #
#

if __name__ == '__main__':
    unittest.main()
#
