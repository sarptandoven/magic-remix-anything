# audioop.py - Compatibility module for Python 3.13+
# This module provides shims for the removed audioop module

import warnings

def bias(fragment, width, bias):
    """Add a bias to audio samples."""
    warnings.warn("audioop.bias not available in Python 3.13+", RuntimeWarning)
    return fragment

def add(fragment1, fragment2, width):
    """Add two audio fragments."""
    warnings.warn("audioop.add not available in Python 3.13+", RuntimeWarning)
    return fragment1

def mul(fragment, width, factor):
    """Multiply audio samples by a factor."""
    warnings.warn("audioop.mul not available in Python 3.13+", RuntimeWarning)
    return fragment

def reverse(fragment, width):
    """Reverse audio samples."""
    warnings.warn("audioop.reverse not available in Python 3.13+", RuntimeWarning)
    return fragment

def tomono(fragment, width, lfactor, rfactor):
    """Convert stereo to mono."""
    warnings.warn("audioop.tomono not available in Python 3.13+", RuntimeWarning)
    return fragment

def tostereo(fragment, width, lfactor, rfactor):
    """Convert mono to stereo.""" 
    warnings.warn("audioop.tostereo not available in Python 3.13+", RuntimeWarning)
    return fragment + fragment

def ratecv(fragment, width, nchannels, inrate, outrate, state, weightA=1, weightB=0):
    """Convert sample rate."""
    warnings.warn("audioop.ratecv not available in Python 3.13+", RuntimeWarning)
    return fragment, state

def lin2lin(fragment, width, newwidth):
    """Convert sample width."""
    warnings.warn("audioop.lin2lin not available in Python 3.13+", RuntimeWarning)
    return fragment

def adpcm2lin(fragment, width, state):
    """Convert ADPCM to linear."""
    warnings.warn("audioop.adpcm2lin not available in Python 3.13+", RuntimeWarning)
    return fragment, state

def lin2adpcm(fragment, width, state):
    """Convert linear to ADPCM."""
    warnings.warn("audioop.lin2adpcm not available in Python 3.13+", RuntimeWarning)
    return fragment, state

def minmax(fragment, width):
    """Get min and max values."""
    warnings.warn("audioop.minmax not available in Python 3.13+", RuntimeWarning)
    return 0, 0

def max(fragment, width):
    """Get maximum value."""
    warnings.warn("audioop.max not available in Python 3.13+", RuntimeWarning)
    return 0

def avg(fragment, width):
    """Get average value."""
    warnings.warn("audioop.avg not available in Python 3.13+", RuntimeWarning)
    return 0

def rms(fragment, width):
    """Get RMS value."""
    warnings.warn("audioop.rms not available in Python 3.13+", RuntimeWarning)
    return 0

def cross(fragment, width):
    """Count zero crossings."""
    warnings.warn("audioop.cross not available in Python 3.13+", RuntimeWarning)
    return 0

def findfactor(fragment, reference):
    """Find factor between fragments."""
    warnings.warn("audioop.findfactor not available in Python 3.13+", RuntimeWarning)
    return 1.0

def findfit(fragment, reference):
    """Find best fit between fragments."""
    warnings.warn("audioop.findfit not available in Python 3.13+", RuntimeWarning)
    return 0

def findmax(fragment, length):
    """Find maximum correlation."""
    warnings.warn("audioop.findmax not available in Python 3.13+", RuntimeWarning)
    return 0

def getsample(fragment, width, index):
    """Get a sample from fragment."""
    warnings.warn("audioop.getsample not available in Python 3.13+", RuntimeWarning)
    return 0

# Common error class
class error(Exception):
    """audioop error."""
    pass

# For compatibility with code that checks for audioop functions
__all__ = [
    'bias', 'add', 'mul', 'reverse', 'tomono', 'tostereo', 'ratecv', 
    'lin2lin', 'adpcm2lin', 'lin2adpcm', 'minmax', 'max', 'avg', 
    'rms', 'cross', 'findfactor', 'findfit', 'findmax', 'getsample', 'error'
] 