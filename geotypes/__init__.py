import warnings
from . import geometry
try:
    from . import rasters
    from . import features
except ImportError as e:
    warnings.warn("geotypes raster capability not loaded. Error {e}")
