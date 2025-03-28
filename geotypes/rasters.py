import pyproj
from affine import Affine
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.windows import Window
import rasterio.features
from typing import Iterable, Union, Optional, List, Tuple, Dict
try:
    from osgeo import gdal
except ImportError:
    import gdal
from osgeo import osr
import utm
import warnings
import shapely
import shapely.geometry
import os
import geopandas as gpd
import pymap3d
from .geometry import GeoPoint, GeoPoints, GeoArea, LocalArea, LocalPoints, LocalPath, LL, XY, GeoPath, check_crs_equal
from pyproj import Transformer, Proj, transform

class Raster:
    """
    A wrapper around the rasterio raster class, to provide easier access.
    """
    def __init__(self, filepath: str, in_memory: bool = False):
        self.filepath = filepath
        if in_memory:
            # raise NotImplementedError("In memory rasters not supported")
            with rasterio.open(filepath) as src:
                profile = src.profile
                data = src.read()
                memfile = MemoryFile()
                with memfile.open(**profile) as dataset:
                    dataset.write(data)  # Write data to memfile
            self.dataset = memfile.open()
            # print(dataset)
            # self.filepath = None  # No filepath for in-memory rasters
        else:
            self.dataset = rasterio.open(filepath)
        self.transformer = pyproj.Transformer.from_crs("EPSG:4326", self.get_crs(), always_xy=True)


    def __repr__(self):
        """
        Returns a string representation of the raster.

        Returns:
            str: The string representation of the raster.
        """
        return f'Raster("Path={self.filepath}","crs={self.dataset.crs}",size({self.dataset.width},{self.dataset.height},))'

    def shape(self):
        """
        Returns the shape of the raster.

        Returns:
            (int,int): The shape of the raster.
        """
        return self.dataset.shape

    def bounds(self):
        """
        Returns the bounds of the raster.

        Returns:
            (float,float,float,float): The bounds of the raster.
        """
        return self.dataset.bounds

    def get_bounds_as_geoarea(self):
        """
        Returns the bounds of the raster as a GeoArea

        Returns:
            GeoArea: The bounds of the raster.
        """
        box = shapely.geometry.box(*self.bounds())
        s = gpd.GeoSeries(box, crs=self.get_crs())
        area = GeoArea(s)
        return area

    def valid_geoarea(self):
        """
        Returns the valid area of the raster as a GeoArea

        Returns:
            GeoArea: The valid area of the raster.
        """
        data = self.dataset.read(1, masked=True)
        shape_gen = ((shapely.geometry.shape(s), v) for s, v in rasterio.features.shapes(data, transform=self.dataset.transform))
        gdf = gpd.GeoDataFrame(dict(zip(["geometry", "class"], zip(*shape_gen))), crs=self.dataset.crs)
        return GeoArea(gdf.geometry)

    def get_geotransform(self):
        """
        Returns the geotransform of the raster.

        Returns:
            (float,float,float,float,float,float): The geotransform of the raster.
        """
        return self.dataset.transform

    def get_crs(self):
        """Return the crs of the raster."""
        return self.dataset.crs

    def get_nodata(self):
        """Return the nodata value of the raster."""
        return self.dataset.nodata

    def band_count(self):
        """
        Returns the number of bands in the raster.

        Returns:
            int: The number of bands in the raster.
        """
        return self.dataset.count
    

    def get_value(self, p: Union[GeoPoints, GeoPath, GeoPoint, LL]):
        if isinstance(p, LL):
            return self.get_value_at_ll(p)
        if check_crs_equal(self.get_crs(), p.crs):
            p_crs = p
        else:
            p_crs = p.to_crs(self.get_crs())
        if isinstance(p, (GeoPoints, GeoPath)):
            points = [(p.x, p.y) for p in p_crs]
            return [x for x in self.dataset.sample(points)]
        elif isinstance(p, GeoPoint):
            return list(self.dataset.sample([(p_crs.x, p_crs.y)]))[0]
        else:
            return ValueError("p must be GeoPoints, GeoPath or GeoPoint, LL")
        

    def get_value_at_ll(self, ll: LL, band=1):
        """
        Gets the vale of a raster when specified with lat,lon with crs EPSG:4326.

        Args:
            ll: (LL) the lat,lon in EPSG:4326
            band: (int) the band to access.

        Returns:
            float: The value at the lat,lon
        """
        x,y = self.transformer.transform(ll.lon, ll.lat)
        return self.get_value_at_location(XY(x,y),band=band)

    def get_value_at_geo(self, p: GeoPoint, band: int = 1):
        """
        Gets the value of a raster when specified with a GeoPoint.

        Args:
            p: (GeoPoint) The point to access.
            band: (int) the band to access.

        Returns:
            float: The value at the point.
        """
        if isinstance(p.crs, str):
            epsg = p.crs
        elif isinstance(p.crs, pyproj.CRS):
            epsg = p.crs.to_epsg()
        elif isinstance(p.crs, rasterio.CRS):
            epsg = p.crs.to_epsg()
        else:
            raise ValueError("unkown CRS type, {}".format(type(p.crs)))
        if epsg == self.get_crs().to_epsg():
            lp = p
        else:
            lp = p.to_crs(self.get_crs())
        return self.get_value_at_location(XY(lp.x,lp.y),band=band)

    def get_value_at_location(self, xy: XY, band: int = 1):
        """Return the value at a given location, where the location is in the same coordinate system as the raster.

        Args:
            XY: (situ.core.geometry.XY) Location in local coordinates
            band : (float) Band to read.

        Returns:
            float: Value at the given location.
        """
        # get pixel coordinates
        u,v = self.dataset.index(xy.x,xy.y)
        return self.get_value_at_pixel(u,v,band)

    def get_pixels_at_geo(self, p: GeoPoint):
        if isinstance(p.crs, str):
            epsg = p.crs
        elif isinstance(p.crs, pyproj.CRS):
            epsg = p.crs.to_epsg()
        elif isinstance(p.crs, rasterio.CRS):
            epsg = p.crs.to_epsg()
        else:
            raise ValueError("unkown CRS type")
        if epsg == self.get_crs().to_epsg():
            lp = p
        else:
            lp = p.to_crs(self.get_crs())
        return self.dataset.index(p.x,p.y)

    def get_value_at_pixel(self, u: int, v: int, band: int = 1):
        """Return the value at a given pixel.

        Args:
            u: (int) X coordinate of the pixel.
            v: (int)
            band : (int) Band to read.

        Returns:
            float: Value at the given pixel.
        """
        # Read the value from the raster
        r = self.dataset.read(band, window=Window(v, u, 1, 1))
        # ^ is equivalent to r.dataset.read(band)[u,v]
        # Returns empty if out of bounds
        if len(r) == 0:
            return None
        try:
            r = np.array(r[0]).item()
        except ValueError:
            return None
        # if no data value return None
        if r == self.dataset.nodata:
            return None
        return r
    
    def extract_patch(self, xy: XY, width: int, height: int, band: int = 1):
        """Read a patch of the raster.

        Args:
            xy: (XY) XY coordinate of the centre of the patch, in the same crs as the raster.
            width: (int) Width of the patch in pixels
            height: (int) Height of the patch in pixels
            band: (int) Band to read.

        Returns:
            np.ndarray: Patch of the raster.
        """
        u,v = self.dataset.index(xy.x,xy.y)
        return self.extract_patch_pixels(u,v,width,height,band)

    def extract_patch_pixels(self, u: int, v: int, width: int, height: int, band: int = 1):
        """
        Extracts a patch in pixels centred around (u,v)

        Args:
            u: x pixel coordinate
            v: y pixel coodinate
            width: patch size in pixel
            height: patch height in pixels
            band: the band to access.

        Returns:
            np.ndarray: The patch
        """
        uoff = u - width//2
        voff = v - height//2
        w = Window(voff,uoff,width,height)
        # Read the value from the raster
        r = self.dataset.read(band, window=w)
        # Returns empty if out of bounds
        if len(r) == 0:
            return None
        # if no data value return None
        if np.any(r == self.dataset.nodata):
            return None
        return r

    def extract_patch_ll(self, ll: LL, width: int, height: int, band: int = 1):
        """
        Extracts a patch centered around the Lat,Lon in EPSG:4326.

        Args:
            ll: (situ.core.geometry.LL) the centre of the patch in EPSG:4326
            width: (int) total patch size in pixels
            height: (int) total patch size in pixels
            band: (band) the band to access.

        Returns:
            np.array: the patch
        """
        x, y = self.transformer.transform(ll.lon, ll.lat)
        return self.extract_patch(XY(x, y),width, height, band=band)

    def extract_patch_geo(self, p: GeoPoint, width: int, height: int, band: int = 1):
        """
        Extracts a patch centered around a GeoPoint.

        Args:
            p: (situ.core.geometry.GeoPoint) the centre of the patch in the rasters CRS
            width: (int) total patch size in pixels
            height: (int) total patch size in pixels
            band: (band) the band to access.

        Returns:
            np.array: the patch
        """
        if isinstance(p.crs, str):
            epsg = p.crs
        elif isinstance(p.crs, pyproj.CRS):
            epsg = p.crs.to_epsg()
        elif isinstance(p.crs, rasterio.CRS):
            epsg = p.crs.to_epsg()
        else:
            raise ValueError("unkown CRS type")

        if epsg == self.get_crs().to_epsg():
            lp = p
        else:
            lp = p.to_crs(self.get_crs())

        return self.extract_patch(XY(lp.x, lp.y),width, height, band=band)

    def extract_multiband_patch(self, xy: XY, width: int, height: int, bands: Iterable[int]):
        """
        Extract a multiband patch (e.g. from a mosaic)

        Args:
            xy: (situ.core.geometry.XY)
            width: (int) patch size in pixels
            height: (int) patch size in pixels
            bands: [int] a list of bands to extract from. i.e. for a mosaic do list(range(1,3)).

        Returns:
            np.array: the patch
        """
        patches = []
        for b in bands:
            patch = self.extract_patch(xy, width, height, b)
            if patch is None:
                return None
            patches.append(patch)
        return np.array(patches)

    def extract_multiband_patch_ll(self, ll: LL, width: int, height: int, bands: Iterable[int]):
        """
        Extract a multiband patch (e.g. from a mosaic) with an LL input (a lat,lon in EPSG:4326)

        Args:
            ll: (situ.core.geometry.LL) the centre coordinate
            width: (int) width in pixels
            height: (int) height in pixels
            bands: [int] list of bands to form the patches from.

        Returns:
            np.array: the patch
        """
        x, y = self.transformer.transform(ll.lon, ll.lat)
        return self.extract_multiband_patch(XY(x,y), width, height, bands)


    def extract_multiband_patch_geo(self, p: GeoPoint, width: int, height: int, bands: Iterable[int]):
        """
        Extract a multiband patch (e.g. from a mosaic) with an GeoPoint input (a point with a CRS)

        Args:
            p: (situ.core.geometry.GeoPoint) the centre coordinate
            width: (int) width in pixels
            height: (int) height in pixels
            bands: [int] list of bands to form the patches from.

        Returns:
            np.array: the patch
        """
        if isinstance(p.crs, str):
            epsg = p.crs
        elif isinstance(p.crs, pyproj.CRS):
            epsg = p.crs.to_epsg()
        elif isinstance(p.crs, rasterio.CRS):
            epsg = p.crs.to_epsg()
        else:
            raise ValueError("unkown CRS type")
        if epsg == self.get_crs().to_epsg():
            lp = p
        else:
            lp = p.to_crs(self.get_crs())

        return self.extract_multiband_patch(XY(lp.x,lp.y), width, height, bands)

    def get_stats(self, band: int =1):
        """
        Gets the stats for the reward raster.

        Returns:
            (float, float, float, float) min, max, mean, stddev

        """
        stats = self.dataset.statistics(bidx=band, approx=True)  # min, max, mean, std
        return stats

    def grid_sample(self, spacing_pixels: int = None, spacing_distance: float = None) -> List[GeoPoint]:
        if spacing_pixels is None and spacing_distance is None:
            raise ValueError("Either spacing_pixels or spacing distance should be given")
        if spacing_pixels is not None and spacing_distance is not None:
            warnings.warn("spacing_pixels and spacing_distance both given, using spacing pixels")
        use_pixels = True
        if spacing_pixels is None:
            use_pixels = False
        rcrs = rasterio.crs.CRS(self.get_crs())
        is_geographic = rcrs.is_geographic
        is_projected = rcrs.is_projected
        if use_pixels:
            x_pixs = np.arange(0,self.dataset.shape[0], spacing_pixels)
            y_pixs = np.arange(0,self.dataset.shape[1], spacing_pixels)
            # pxl_list = []
            loc_list = []
            for u in x_pixs:
                for v in y_pixs:
                    # pxl = np.array([u,v])
                    p = GeoPoint(*self.dataset.xy(u,v), crs=self.get_crs())
                    # p = GeoPoint(*self.dataset.transform * (u, v), crs=self.get_crs())
                    loc_list.append(p)
        elif not is_geographic and is_projected:
            top_left = self.dataset.transform * (0.,0.)
            top_right = self.dataset.transform * (self.dataset.shape[0], 0.)
            bottom_left = self.dataset.transform * (0.,self.dataset.shape[1])
            loc_list = []
            for x in np.arange(top_left[0], top_right[0], spacing_distance):
                for y in np.arange(bottom_left[1], bottom_left[1], spacing_distance):
                    p = GeoPoint(x,y,crs=self.get_crs())
                    loc_list.append(p)
        else: # if geographic you need to convert to local
            origin_ll = GeoPoint(*self.dataset.transform * (self.dataset.shape[0]//2, self.dataset.shape[1]//2), crs=self.get_crs()).to_ll()
            top_left = GeoPoint(*self.dataset.transform * (0.,0.), crs=self.get_crs())
            top_right = GeoPoint(*self.dataset.transform * (self.dataset.shape[0], 0.), crs=self.get_crs())
            bottom_left = GeoPoint(*self.dataset.transform * (0., self.dataset.shape[1]), crs=self.get_crs())

            tl = top_left.to_local(origin_ll)
            tr = top_right.to_local(origin_ll)
            bl = bottom_left.to_local(origin_ll)
            if np.sqrt((tr[0] - tl[0])**2 + (tl[1] - bl[1])**2) > 2e4:
                warnings.warn("Grid sample with a distance in meters uses a local projection, will give unreliable results over large areas")
            loc_list = []
            for xl in np.arange(tl[0], tr[0], spacing_distance):
                for yl in np.arange(bl[1], tr[1], spacing_distance):
                    pl = XY(xl,yl)
                    p = pl.to_geo(origin_ll, self.get_crs())
                    loc_list.append(p)

        return loc_list


    def where(self, in_range=None, greater_than=None, less_than=None, equal_to=None, band: int = 1):
        """
        Find the indices where the condition is true.

        Args:
            condition: (str) The condition to check.
            band: (int) The band to check.

        Returns:
            np.array: The indices where the condition is true.
        """
        data = self.dataset.read(band)
        if in_range is not None:
            idxs = np.argwhere((data >= in_range[0]) & (data <= in_range[1]))
        elif greater_than is not None:
            idxs = np.argwhere(data > greater_than)
        elif less_than is not None:
            idxs = np.argwhere(data < less_than)
        elif equal_to is not None:
            idxs = np.argwhere(data == equal_to)
        else:
            raise ValueError("No condition given")
        points = []
        for pxl in idxs:
            pxl = XY(*pxl)
            p = GeoPoint(*self.dataset.xy(pxl.x,pxl.y), crs=self.get_crs())
            points.append(p)
        return points




    def explore(self, m=None):

        from localtileserver import get_folium_tile_layer, TileClient
        import folium
        # Create ipyleaflet tile layer from that server
        client = TileClient(self.filepath)

        t = get_folium_tile_layer(client)
        if m is None:
            m = folium.Map(center=client.center())#, zoom=client.default_zoom)
        m.add_child(t)
        return m, t, client


class RasterItem:
    def __init__(self, dataset, size: list, band: int = 1, in_memory: bool =False):
        self.dataset = Raster(dataset, in_memory=in_memory)
        self.size = size
        self.crs = self.dataset.get_crs()
        self.band=band
        self.band_count = self.dataset.band_count()
        self.no_data = self.dataset.get_nodata()
    def __repr__(self):
        return f"RasterItem({self.dataset.filepath}, {self.crs}, {self.size}, {self.band_count})"

class RasterRegistry:
    def __init__(self, raster_cfg):
        self.raster_lookup = {}
        for k,v in raster_cfg.items():
            self.raster_lookup[k] = RasterItem(v['path'], v['size'], band=v.get('band',1), in_memory=v.get('in_memory', False))
        self.leading_raster = self.raster_lookup[list(raster_cfg.keys())[0]]

    def __getitem__(self, item):
        return self.raster_lookup[item]

    def __repr__(self):
        return f"RasterRegistry({self.raster_lookup})"

    def __len__(self):
        return len(self.raster_lookup)

    def __iter__(self):
        return iter(self.raster_lookup)

    def __contains__(self, item):
        return item in self.raster_lookup

    def keys(self):
        return self.raster_lookup.keys()

    def values(self):
        return self.raster_lookup.values()

    def items(self):
        return self.raster_lookup.items()

    def get_bounds_as_geoarea(self):
        return self.leading_raster.dataset.get_bounds_as_geoarea()

    def get_patches(self, p: GeoPoint):
        for r in self.raster_lookup.values():
            z = r.dataset.extract_patch_geo(p, r.size[0], r.size[1])
            yield z

    def get_named_patches(self, p) -> Dict[str, np.ndarray]:
        d = {}
        for k,v in self.raster_lookup.items():
            d[k] = v.dataset.extract_patch_geo(p, v.size[0], v.size[1])
        return d

def load_raster_registry(raster_files):
    raster_lookup = {}
    for r in raster_files:
        if not os.path.exists(r):
            raise ValueError(f"Warning: Raster file {r} does not exist.")

        raster_lookup[os.path.splitext(os.path.basename(r))[0]] = {
            'path': r,
            'size': [1,1]
        }
    return RasterRegistry(raster_lookup)



def example_config() -> dict:
    return {
        "depth": {
            "path": "depth.tif",
            "size": [1,1]
            },
        "rugosity" : {
            "path": "rugosity.tif",
            "size": [1,1]
            },
        }
    
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('raster')
    parser.add_argument('--query', '-q', help="The query location")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    raster = Raster(args.raster)
    print(raster.get_crs())
    if args.query:
        # query_point = [float(x) for x in args.query.split(',')]

        ll = LL(-39.01794,143.59151)
        r = raster.get_value_at_ll(ll, band=1)
        print(r)
        patch = raster.extract_patch_ll(ll, width=11, height=11, band=1)
        print(patch)
        print(patch[patch.shape[0]//2, patch.shape[1]//2])



