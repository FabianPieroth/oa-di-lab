# ------------------------------------------------------------------------------
#  File: oa.py
#  Author: Jan Kukacka
#  Date: 11/2018
#  Email: jan.kukacka@tum.de
# ------------------------------------------------------------------------------
#  Unit test script for oa module
# ------------------------------------------------------------------------------



'''
Unit tests for linear unmixing
# TODO: test use_constant parameter

>>> from lib.oa import linear_unmixing
>>> import numpy as np
>>> data = np.random.rand(100,28)
>>> u = linear_unmixing(data)
>>> u.shape
(100, 4)
>>> data = np.random.rand(10,10,28)
>>> u = linear_unmixing(data)
>>> u.shape
(10, 10, 4)
>>> data = np.random.rand(100,28)
>>> u,e = linear_unmixing(data, return_error=True)
>>> u.shape
(100, 4)
>>> e.shape
(100,)
>>> u,e = linear_unmixing(data, non_negative=True, return_error=True)
>>> u.shape
(100, 4)
>>> e.shape
(100,)
>>> s = np.random.rand(3,28)
>>> u,e = linear_unmixing(data, spectra=s, non_negative=True, return_error=True)
>>> u.shape
(100, 3)
>>> e.shape
(100,)
>>> u,e = linear_unmixing(data.reshape((10,10,28)), spectra=s, non_negative=True, return_error=True)
>>> u.shape
(10, 10, 3)
>>> e.shape
(10, 10)


Unit tests for spectral_F_test

>>> from lib.oa import spectral_F_test
>>> import numpy as np
>>> data = np.random.rand(100,28)
>>> p = spectral_F_test(data)
>>> p.shape
(100,)
'''

if __name__ == "__main__":
    import doctest
    doctest.testmod()
