import numpy as np
import warnings
from .geometry import LL, GeoPoint, GeoPoints, GeoPath, check_crs_equal
from .rasters import Raster, RasterRegistry
from typing import Union, Iterable, Optional
from multiprocessing import Pool, cpu_count

class FeatureExtractorBase:
    def get_output_format(self):
        """
        Returns the output format of the feature vector, where the length corresponds to the length of the feature vector.
        i.e. if the feature vector contains [depth,slope,aspect,rugosity] this function would return ["depth","slope","aspect","rugosity"]


        Returns:
            [str]: maps index to feature vector item.
        """
        raise NotImplementedError()

    def get_z(self, p: Union[LL, GeoPoint]):
        """
        Gets the features at a given lat/lon

        Args:
            p: (LL,GeoPoint) Lat/lon or GeoPoint

        Returns:
            np.ndarray: Features at the given location.
        """
        raise NotImplementedError()

    def get_zs(self, ps: GeoPoints) -> np.ndarray:
        """
        Gets the features at a given set of lat/lon points

        Args:
            ps: (GeoPoints) The set of points

        Returns:
            np.ndarray: The features at the given location.
        """
        raise NotImplementedError()

    def check_valid(self, samples: Union[GeoPoint, LL, GeoPoints, GeoPath]):
        """
        Checks if a sample set is valid.

        Args:
            samples: (GeoPoint, LL, GeoPoints, GeoPath) The samples to check.

        Returns:
            bool: True if the samples are valid, False otherwise.
        """
        raise NotImplementedError()

    def get_named_zs(self, p: Union[LL, GeoPoint]):
        """
        Gets the features at a given lat/lon

        Args:
            p: (LL,GeoPoint) Lat/lon or GeoPoint

        Returns:
            dict: Features at the given location.
        """
        raise NotImplementedError()

class FeaturesFromRaster:
    """
    Gathers raw features from a raster.
    """
    def __init__(self, raster_registry: RasterRegistry):
        self.raster_registry = raster_registry

    def get_output_format(self):
        """
        Returns the output format of the feature vector, where the length corresponds to the length of the feature vector.
        i.e. if the feature vector contains [depth,slope,aspect,rugosity] this function would return ["depth","slope","aspect","rugosity"]
        Returns:
            [str]: maps index to feature vector item.
        """
        return list(self.raster_registry.keys())

    def get_z(self, p: Union[LL, GeoPoint]):
        """
        Gets the features at a given lat/lon

        Args:
            p: (LL,GeoPoint) Lat/lon or GeoPoint

        Returns:
            np.ndarray: Features at the given location.
        """

        # Checks for either LL or GeoPoint
        if isinstance(p, LL):
            use_ll = True
        elif isinstance(p, GeoPoint):
            use_ll = False
        else:
            raise ValueError("p must be LL or GeoPoint")

        feats = []
        data_ok = True
        # iterate over the rasters and get the values
        for rname,ritem in self.raster_registry.items():
            if use_ll:
                v = ritem.dataset.get_value_at_ll(p, band=ritem.band)  # check if the value is valid
            else:
                v = ritem.dataset.get_value_at_geo(p, band=ritem.band)
                # v = ritem.dataset.get_value(p)

            if v is None: # if not, then we can't use this point
                data_ok = False
                break
            feats.append(v)
        if not data_ok: # if we couldn't get the data, then return None
            return None

        feat_arr = np.array(feats)
        # feat_arr = np.concatenate(feats, axis=-1) # concatenate the features
        feat_arr = feat_arr.squeeze() # squeeze the features
        return feat_arr # return the features


    def get_zs(self, ps: GeoPoints) -> np.ndarray:
        """
        Gets the features at a given set of lat/lon points

        Args:
            ps: (GeoPoints) The set of points

        Returns:
            np.ndarray: The features at the given location.
        """
        features = []
        for rname, ritem in self.raster_registry.items():
            if check_crs_equal(ritem.dataset.dataset.crs, ps.crs):
                ps_crs = ps
            else:
                ps_crs = ps.to_crs(ritem.dataset.dataset.crs)

            features.append(np.array(ritem.dataset.get_value(ps_crs)).squeeze())

        if len(features) == 0:
            return None
        return np.asarray(features).T

    def check_valid(self, samples: Union[GeoPoint, LL, GeoPoints, GeoPath]):
        """
        Checks if a sample set is valid.

        Args:
            samples: (GeoPoint, LL, GeoPoints, GeoPath) The samples to check.

        Returns:
            bool: True if the samples are valid, False otherwise.
        """
        if isinstance(samples, (GeoPoint, LL)):
            samples = [samples]
        for p in samples:
            if self.get_z(p) is None:
                return False
        return True
