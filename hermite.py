from __future__ import division, print_function, absolute_import

import numpy as np
from numpy import poly1d

""" A stub for the Hermite polynomials for older numpy versions.
This uses _hermnorm recursion, a direct copy-paste from 
scipy/stats/morestats.py
"""

def _hermnorm(N):
    # return the negatively normalized hermite polynomials up to order N-1
    #  (inclusive)
    #  using the recursive relationship
    #  p_n+1 = p_n(x)' - x*p_n(x)
    #   and p_0(x) = 1
    plist = [None]*N
    plist[0] = poly1d(1)
    for n in range(1,N):
        plist[n] = plist[n-1].deriv() - poly1d([1,0])*plist[n-1]
    return plist


def _HE(coef):
    """Cook up an HermiteE replacement from _hermnorm."""
    h = _hermnorm(len(coef))
    def f(x):
        return sum((1 - 2*(j%2)) * c * h[j](x) for (j, c) in enumerate(coef))
    return f

try:
    from numpy.polynomial.hermite_e import HermiteE
except ImportError:  # numpy < 1.6
    HermiteE = _HE


if __name__ == "__main__":
    coef = np.random.random(8)
    xx = np.random.random(1000) * 20 - 10
    np.testing.assert_allclose(HermiteE(coef)(xx), _HE(coef)(xx))

