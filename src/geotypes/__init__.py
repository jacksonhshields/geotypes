import warnings

# Import modules
from . import geometry

# Export main geometry classes for convenient access
from .geometry import (
    GeoType,
    LL,
    XY,
    GeoPoint,
    GeoPath,
    GeoPoints,
    GeoArea,
    LocalPath,
    LocalPoints,
    LocalArea,
)

# Try to import optional raster/feature modules
try:
    from . import rasters
    from . import features
    from .rasters import Raster
except ImportError as e:
    warnings.warn(f"geotypes raster capability not loaded. Error: {e}")

__all__ = [
    "geometry",
    "rasters",
    "features",
    "GeoType",
    "LL",
    "XY",
    "GeoPoint",
    "GeoPath",
    "GeoPoints",
    "GeoArea",
    "LocalPath",
    "LocalPoints",
    "LocalArea",
    "Raster",
]
