import os.path
import numpy as np
import pyproj
import shapely.geometry
import shapely.affinity
from dataclasses import dataclass
import pandas as pd
import geopandas as gpd
import rasterio
import pymap3d
import geopy.distance
from typing import Union, List, Tuple, Optional, Iterable

class GeoType:
    """
    A base class to inherit from (makes typehinting easier)
    """
    def to_ll(self):
        raise NotImplementedError()
    def to_local(self, origin):
        raise NotImplementedError()


def get_local_to_geo_transformer(local_origin_ll, target_crs):
    """
    Gets the transform from a local projection to a geo one

    Args:
        local_origin_ll: (LL) the origin lat,lon
        target_crs: (pyproj.CRS, str) The target crs

    Returns:
        pyproj.Transformer: The transformer
    """
    # Define local projection (Transverse Mercator)
    local_proj = pyproj.Proj(proj='tmerc', lat_0=local_origin_ll.lat, lon_0=local_origin_ll.lon, k=1, x_0=0, y_0=0, ellps='WGS84', units='m')
    # Define a geo projection for the raster
    geo_proj = pyproj.Proj(target_crs)
    return pyproj.Transformer.from_proj(local_proj, geo_proj, always_xy=True)

def get_geo_to_local_transformer(src_crs, local_origin_ll):
    """
    Gets the transform from a global coordinate reference system to a local one.

    Args:
        src_crs: (pyproj.CRS, str)  The source coordinate reference system
        local_origin_ll: (LL) The local origin target.

    Returns:
        pyproj.Transformer: The transformer.
    """
    # Define local projection (Transverse Mercator)
    local_proj = pyproj.Proj(proj='tmerc', lat_0=local_origin_ll.lat, lon_0=local_origin_ll.lon, k=1, x_0=0, y_0=0, ellps='WGS84', units='m')
    # Define a geo projection for the raster
    geo_proj = pyproj.Proj(src_crs)
    return pyproj.Transformer.from_proj(geo_proj, local_proj, always_xy=True)

def check_crs_equal(crs_a, crs_b):
    def to_epsg_code(c):
        if isinstance(c, str):
            epsg_code = c.split(':')[-1]
        elif isinstance(c, (pyproj.CRS, rasterio.CRS)):
            epsg_code = c.to_epsg()
        else:
            raise ValueError("crs either has to be a epsg string or pyproj/rasterio CRS")
        return epsg_code
    epsg_a = to_epsg_code(crs_a)
    epsg_b = to_epsg_code(crs_b)
    if epsg_a == epsg_b:
        return True
    else:
        return False

@dataclass
class LL(GeoType):
    lat: float
    lon: float
    def __array__(self) -> np.ndarray:
        """
        To array

        Returns:
            np.ndarray
        """
        return np.array([self.lon, self.lat])
    def to_geopoint(self) -> 'GeoPoint':
        """
        To geopoint

        Returns:
            GeoPoint: The GeoPoint in crs EPSG:4326
        """
        return GeoPoint(x=self.lon, y=self.lat, crs="EPSG:4326")
    def to_local(self, origin_ll) -> 'XY':
        """
        To local

        Args:
            origin_ll: (LL) The local origin

        Returns:
            XY: The XY in NED
        """
        enu = pymap3d.geodetic2enu(lat=self.lat, lon=self.lon, h=0., lat0=origin_ll.lat, lon0=origin_ll.lon, h0=0.)
        return XY(enu[0], enu[1])


    def as_geoseries(self) -> gpd.GeoSeries:
        """
        Returns a geoseries of the point
        """
        return gpd.GeoSeries(shapely.geometry.Point(self.lon, self.lat), crs="EPSG:4326")

    def explore(self, **kwargs):
        """
        Views the point in an interactive map

        Returns:
            The folium map
        """
        gs = self.as_geoseries()
        return gs.explore(**kwargs)

    def plot(self):
        """
        Plots the point
        Returns:
            matplotlib.axes.Axes: The axes.
        """
        gs = self.as_geoseries()
        return gs.plot()

    @classmethod
    def from_degrees_minutes(cls, lat_deg: int, lat_min: float, lon_deg: int, lon_min: float):
        """
        Creates a LL from degrees and minutes

        Args:
            lat_deg: (int) The latitude degrees
            lat_min: (float) The latitude minutes
            lon_deg: (int) The longitude degrees
            lon_min: (float) The longitude minutes

        Returns:
            LL: The LL
        """
        if lat_deg < 0:
            lat = lat_deg - lat_min / 60
        else:
            lat = lat_deg + lat_min / 60
        if lon_deg < 0:
            lon = lon_deg - lon_min / 60
        else:
            lon = lon_deg + lon_min / 60
        return cls(lat=lat, lon=lon)

    @classmethod
    def from_degrees_minutes_seconds(cls, lat_deg: int, lat_min: int, lat_sec: float, lon_deg: int, lon_min: int, lon_sec: float):
        """
        Creates a LL from degrees, minutes and seconds

        Args:
            lat_deg: (int) The latitude degrees
            lat_min: (int) The latitude minutes
            lat_sec: (float) The latitude seconds
            lon_deg: (int) The longitude degrees
            lon_min: (int) The longitude minutes
            lon_sec: (float) The longitude seconds

        Returns:
            LL: The LL
        """
        lat = lat_deg + lat_min / 60 + lat_sec / 3600
        lon = lon_deg + lon_min / 60 + lon_sec / 3600
        return cls(lat=lat, lon=lon)

    def print_degrees_decimal(self):
        print(f"Latitude: {self.lat}, Longitude: {self.lon}")

    def print_degrees_minutes(self):
        lat_deg = int(self.lat)
        lat_min = abs((self.lat - lat_deg) * 60)
        lon_deg = int(self.lon)
        lon_min = abs((self.lon - lon_deg) * 60)
        print(f"Latitude: {lat_deg}° {lat_min}', Longitude: {lon_deg}° {lon_min}'")

    def print_degrees_minutes_second(self):
        lat_deg = int(self.lat)
        lat_min = int((self.lat - lat_deg) * 60)
        lat_sec = ((self.lat - lat_deg) * 60 - lat_min) * 60
        lon_deg = int(self.lon)
        lon_min = int((self.lon - lon_deg) * 60)
        lon_sec = ((self.lon - lon_deg) * 60 - lon_min) * 60
        print(f"Latitude: {lat_deg}° {lat_min}' {lat_sec}'', Longitude: {lon_deg}° {lon_min}' {lon_sec}''")



@dataclass
class XY:
    """
    A local XY point
    """
    x: float
    y: float

    def __array__(self) -> np.ndarray:
        """
        To array

        Returns:
            np.ndarray: The array [x,y]
        """
        return np.array([self.x, self.y])

    def to_ll(self, origin_ll : LL) -> LL:
        """
        To LL

        Args:
            origin_ll: (LL) The origin LL

        Returns:
            LL: The LL
        """
        llh = pymap3d.enu2geodetic(self.x, self.y, 0., lat0=origin_ll.lat, lon0=origin_ll.lon, h0=0.)
        return LL(lat=llh[0], lon=llh[1])

    def to_geo(self, origin_ll : LL, target_crs : str) -> 'GeoPoint':
        """
        To geo

        Args:
            origin_ll: (LL) The origin LL
            target_crs: (pyproj.CRS, str) The target crs

        Returns:
            GeoPoint: The GeoPoint
        """
        t = get_local_to_geo_transformer(origin_ll, target_crs)
        ret = t.transform(self.x, self.y)
        return GeoPoint(x=ret[0], y=ret[1], crs=target_crs)
    def __getitem__(self, item : int) -> float:
        """
        Gets the item

        Args:
            item: int

        Returns:
            float: the item
        """
        if item == 0:
            return self.x
        elif item == 1:
            return self.y
        else:
            raise ValueError("Invalid index")

    def translate(self, a: Union['XY', np.ndarray, List]) -> 'XY':
        """
        Translate the XY point

        Returns:
            XY: The translated XY
        """
        return XY(self.x + a[0], self.y + a[1])



class GeoPoint(GeoType):
    """
    A class to represent a point in geographical space.
    """
    def __init__(self, x: float, y: float, crs: Union[str, pyproj.CRS]):
        """
        Constructs all the necessary attributes for the GeoPoint object.

        Args:
            x: (float) The x coordinate
            y: (float) The y coordinate
            crs: (str, pyproj.CRS) The coordinate reference system of the point.

        """
        self.x = x
        self.y = y
        self.crs = crs

    def __getitem__(self, item):
        if item == 0:
            return self.x
        elif item == 1:
            return self.y
        else:
            raise IndexError("Index out of scope for 2D point")

    def __repr__(self) -> str:
        """
        Returns the representation of the GeoPoint object.

        Returns:
            str: The representation of the GeoPoint object.
        """
        return f"GeoPoint(x={self.x},y={self.y},crs={self.crs})"

    def to_crs(self, new_crs: Union[str, pyproj.CRS]) -> 'GeoPoint':
        """
        Converts the GeoPoint to a new coordinate reference system.

        Args:
            new_crs: (str, pyproj.CRS) The new coordinate reference system.

        Returns:
            GeoPoint: The GeoPoint in the new coordinate reference system.
        """
        transformer = pyproj.Transformer.from_crs(self.crs, new_crs, always_xy=True)
        x,y = transformer.transform(self.x, self.y)
        return GeoPoint(x,y,new_crs)

    def to_ll(self) -> LL:
        """
        Converts the GeoPoint to a LL.

        Returns:
            LL: The LL.
        """
        p = self.to_crs("EPSG:4326")
        return LL(lat=p.y, lon=p.x)

    def to_local(self, origin: LL) -> XY:
        """
        Converts the GeoPoint to a local XY.

        Args:
            origin: (LL) The origin

        Returns:
            XY: The XY in ENU
        """
        t = get_geo_to_local_transformer(src_crs=self.crs, local_origin_ll=origin)
        enu = t.transform(self.x, self.y)
        return XY(enu[0],enu[1])

    def as_geoseries(self) -> gpd.GeoSeries:
        """
        Returns a geoseries of the point
        """
        return gpd.GeoSeries(shapely.geometry.Point(self.x, self.y), crs=self.crs)

    def explore(self, **kwargs):
        """
        Views the point in an interactive map

        Returns:
            The folium map
        """
        gs = self.as_geoseries()
        return gs.explore(**kwargs)

    def plot(self):
        """
        Plots the point
        Returns:
            matplotlib.axes.Axes: The axes.
        """
        gs = self.as_geoseries()
        return gs.plot()

    def distance(self, other: GeoType):
        if not check_crs_equal(self.crs, other.crs):
            other = other.to_crs(self.crs)
        if isinstance(self.crs, str):
            crs = pyproj.CRS(self.crs)
        else:
            crs = self.crs

        if crs.is_geographic:
            this_ll = self.to_ll()
            if isinstance(other, GeoPoint):
                other_ll = other.to_ll()
                d = geopy.distance.distance((this_ll.lat, this_ll.lon), (other_ll.lat, other_ll.lon))
                return d.meters
            else:
                ds = []
                for o in other:
                    oll = o.to_ll()
                    d = geopy.distance.distance((this_ll.lat, this_ll.lon), (oll.lat, oll.lon))
                    ds.append( d.meters )
                return np.min(ds)
        else:
            return self.as_geoseries().distance(other.as_geoseries())







class LocalPath:
    """
    A class to represent a path in local space.
    """
    def __init__(self, points: Union[shapely.geometry.LineString, List[XY], List[np.ndarray]]):
        """
        Constructs all the necessary attributes for the LocalPath object.

        Args:
            points (shapely.geometry.LineString, [XY], [np.ndarray]): The points defining the path.
        """
        if isinstance(points, shapely.geometry.LineString):
            ls = points
        elif isinstance(points, list):
            if isinstance(points[0], XY):
                ls = shapely.geometry.LineString([(xy.x, xy.y) for xy in points])
            elif isinstance(points[0], (np.ndarray, list)):
                ls = shapely.geometry.LineString(points)
            else:
                raise ValueError("invalid type contained in list")
        else:
            raise ValueError("Invalid input - needs to be a LineString, list of XY or list of list/ndarray")

        self.ls = ls

    def __repr__(self) -> str:
        """
        The string representation of the LocalPath object.

        Returns:
            str: The string representation of the LocalPath object.

        """
        return self.ls.__repr__()

    def __getitem__(self, item: int) -> XY:
        """
        Gets the item from the LocalPath object.

        Args:
            item: (int) the item index

        Returns:
            XY: The XY at the given index.
        """
        ret = self.ls.coords[item]
        return XY(ret[0], ret[1])

    def __len__(self) -> int:
        """
        Gets the length of the LocalPath object.

        Returns:
            int: The length of the LocalPath object.
        """
        return len(self.ls.coords)

    def to_ll(self, origin: LL) -> 'GeoPath':
        """
        Converts to LL

        Args:
            origin: (LL) The origin

        Returns:
            GeoPath: The GeoPath in crs EPSG:4326
        """
        points = self.ls.coords
        lls = [pymap3d.enu2geodetic(p[0],p[1], 0, origin.lat, origin.lon, 0) for p in points]
        return GeoPath(lls)

    def to_geo(self, origin: LL, target_crs: Union[str, pyproj.CRS]) -> 'GeoPath':
        """
        Converts to geo

        Args:
            origin: (LL) The origin
            target_crs: (str, pyproj.CRS) The target crs

        Returns:
            GeoPath: The GeoPath in the target crs
        """
        points = self.ls.coords
        t = get_local_to_geo_transformer(origin, target_crs)
        points = [t.transform(p[0],p[1]) for p in points]  # in ENU = EAST NORTH UP
        return GeoPath(points=points, crs=target_crs)

    def length(self) -> float:
        """
        Gets the distance of the LocalPath object.

        Returns:
            float: The distance of the LocalPath object.
        """
        return np.sum(self.ls.length)

    def rotate(self, theta: float, use_midpoint: bool=True) -> 'LocalPath':
        """
        Rotate the path around its centroid. Returns a new LocalPath object.

        Args:
            theta: (float) indexed from north, clockwise.
            use_midpoint: (bool) whether to rotate around the midpoint or the centroid.
        Returns:
            LocalPath: the rotated path.
        """
        if use_midpoint:
            rotate_point = self.ls.interpolate(0.5, normalized=True)
        else:
            rotate_point = self.ls.centroid

        rotated = shapely.affinity.rotate(self.ls, theta, origin=rotate_point, use_radians=True)
        return LocalPath(rotated)

    def translate(self, t: Union[XY, List[float], np.ndarray]) -> 'LocalPath':
        """
        Translate the path by a vector t. Returns a new LocalPath object.

        Args:
            t: (XY) the translation vector.

        Returns:
            LocalPath: the translated path.
        """
        translated = shapely.affinity.translate(self.ls, xoff=t[0], yoff=t[1])
        return LocalPath(translated)

    def plot(self):
        """
        """
        return self.ls.plot()

    def asarray(self) -> np.ndarray:
        return np.array(self.ls.coords)

    def sample_points_along_path(self, d: float) -> 'LocalPath':
        """
        Subsamples along the path with interval d

        Args:
            d: (float) the interval distance in m

        Returns:
            LocalPath: the subsampled LocalPath.
        """
        # Sample the path at a fixed spatial interval
        new_points = []
        travelled = 0.
        while travelled < self.ls.length:
            new_points.append(self.ls.interpolate(travelled))
            travelled += d
        return LocalPath(shapely.geometry.LineString(new_points))

    def to_csv(self, filename):
        """
        """
        df = pd.DataFrame(self.ls.coords, columns=["x","y"])
        df.to_csv(filename, index=False)
        return df

class LocalPoints:
    """
    A collection of points in a local reference frame
    """
    def __init__(self, points: Union[shapely.geometry.MultiPoint, List[XY], List[Union[np.ndarray,list,tuple,np.ndarray]]]):
        """
        """
        if len(points) == 0:
            raise ValueError("Empty points")
        if isinstance(points, shapely.geometry.MultiPoint):
            self.points = points
        elif isinstance(points, np.ndarray):
            input = []
            for n in range(points.shape[0]):
                input.append((points[n,0], points[n,1]))
            self.points = shapely.geometry.MultiPoint(input)
        elif isinstance(points, list):

            if not isinstance(points[0], (np.ndarray, list, tuple, XY)):
                raise ValueError("points contains invalid type ", type(points[0]))
            self.points = shapely.geometry.MultiPoint(
                [(p[0],p[1]) for p in points]
            )
        else:
            raise ValueError("Unsupported type ", type(points))


    def __repr__(self) -> str:
        return self.points.__repr__()

    def __getitem__(self, item) -> XY:
        ret = self.points.coords[item]
        return XY(ret[0], ret[1])

    def __len__(self) -> int:
        return len(self.points.coords)

    def to_ll(self, origin: LL) -> 'GeoPoints':
        """
        """
        points = self.points.coords
        lls = [pymap3d.enu2geodetic(p[0],p[1], 0, origin.lat, origin.lon, 0) for p in points]
        return GeoPoints(lls)

    def to_geo(self, origin: LL, target_crs: str) -> 'GeoPoints':
        """
        """
        points = self.points.coords
        t = get_local_to_geo_transformer(origin, target_crs)
        points = [t.transform(p[0],p[1]) for p in points]
        return GeoPoints(points=points, crs=target_crs)

    def plot(self):
        """
        """
        return self.points.plot()

    def asarray(self) -> np.ndarray:
        return np.array(self.points.coords)



class LocalArea:
    """
    """
    def __init__(self, poly: Union[shapely.geometry.Polygon, List[XY], List[np.ndarray]]):
        """
        """
        if isinstance(poly, shapely.geometry.Polygon):
            self.poly = poly
        elif isinstance(poly, list):
            if len(poly) == 0:
                raise ValueError("Empty list")
            if isinstance(poly[0], XY):
                self.poly = shapely.geometry.Polygon([(xy.x, xy.y) for xy in poly])
            elif isinstance(poly[0], (np.ndarray, list)):
                self.poly = shapely.geometry.Polygon(poly)
            else:
                raise ValueError("invalid type contained in list")

        else:
            raise ValueError("Need to specify either poly, points or xys")

    def __getitem__(self, item: int) -> XY:
        ret = self.poly.exterior.coords[item]
        return XY(ret[0], ret[1])

    def __len__(self) -> int:
        return len(self.poly.exterior.coords)

    def __repr__(self) -> str:
        return self.poly.__repr__()

    def to_ll(self, origin: LL) -> 'GeoArea':
        """
        """
        points = self.poly.exterior.coords
        lls = [pymap3d.ned2geodetic(p[0], p[1], 0, origin.lat, origin.lon, 0) for p in points]
        return GeoArea(lls)

    def to_geo(self, origin: LL, target_crs: str) -> 'GeoArea':
        """
        """
        points = self.poly.exterior.coords
        t = get_local_to_geo_transformer(origin, target_crs)
        points = [t.transform(p[0],p[1]) for p in points]
        return GeoArea(points, crs=target_crs)

    def plot(self):
        """
        """
        return self.poly.plot()

    def asarray(self) -> np.ndarray:
        return np.array(self.poly.exterior.coords)

    def rotate(self, theta: float) -> 'LocalArea':
        """
        Rotate the path around its centroid. Returns a new LocalPath object.

        Args:
            theta: (float) indexed from north, clockwise.

        Returns:
            LocalPath: the rotated path.
        """
        centroid = self.poly.centroid
        rotated = shapely.affinity.rotate(self.poly, theta, origin=centroid, use_radians=True)
        return LocalArea(rotated)

    def translate(self, t: Union[XY, List[float], np.ndarray]) -> 'LocalArea':
        """
        Translate the path by a vector t. Returns a new LocalPath object.

        Args:
            t: (XY) the translation vector.

        Returns:
            LocalPath: the translated path.
        """
        translated = shapely.affinity.translate(self.poly, xoff=t[0], yoff=t[1])
        return LocalArea(translated)





class GeoPath(GeoType):
    """
    A class to represent a path in geographical space.

    Examples:

        # Create a path from an existing GeoSeries
        gpath = gpd.GeoSeries([shapely.geometry.LineString([(151, -30),(152, -31)]),
                               crs="EPSG:4326")
        path = GeoPath(gpath)

        # Create a path from a list of LLs
        lls = [LL(-30,151), LL(-31,151)]
        path = GeoPath(lls)

        # Create a path from a list of points
        points = [(151, -30), (152, -31)]
        path = GeoPath(points, crs="EPSG:4326")
    """
    def __init__(self, points, crs=None):
        """
        Constructs all the necessary attributes for the GeoPath object.

        Args:
            gpath: geopandas.GeoSeries, [LL], [(x,y)], [GeoPoint]
            crs: str
                The coordinate reference system of the points.
        """
        if isinstance(points, gpd.GeoSeries):
            gpath = points
            crs = gpath.crs
        elif isinstance(points, list):
            p = points[0]
            if isinstance(p, GeoPoint):
                crs = p.crs
                gpath = gpd.GeoSeries(shapely.geometry.LineString([(p[0], p[1]) for p in points]), crs=crs)
            elif isinstance(p, LL):
                if crs is None:
                    crs = "EPSG:4326"
                points = [(ll.lon, ll.lat) for ll in points]
                gpath = gpd.GeoSeries(shapely.geometry.LineString(points), crs=crs)
            elif isinstance(p, (np.ndarray, list, tuple)):
                if crs is None:
                    raise ValueError("Need to specify crs if points are not LL or GeoPoint")
                gpath = gpd.GeoSeries(shapely.geometry.LineString(points), crs=crs)
            else:
                raise ValueError("invalid type contained in list")
        else:
            raise ValueError("Invalid input")
        self.gpath = gpath
        self.crs = crs

    def __repr__(self) -> str:
        """
        The string representation of the GeoPath object.

        Returns:
            str: The string representation of the GeoPath object.

        """
        outstr = "GeoPath (crs= {}): \n".format(self.crs)
        for p in self.to_points():
            outstr += "\t({}, {})\n".format(p[0], p[1])
        return outstr

    def length(self) -> float:
        """
        Gets the distance of the GeoPath object.

        Returns:
            float: The distance of the GeoPath object.
        """
        if len(self.gpath) == 0:
            return 0.
        if isinstance(self.crs, str):
            crs = pyproj.CRS(self.crs)
        else:
            crs = self.crs
        if crs.is_geographic:
            lls = self.to_lls()
            d = 0.
            for i in range(len(lls)-1):
                d += geopy.distance.distance((lls[i].lat, lls[i].lon), (lls[i+1].lat, lls[i+1].lon)).meters
            return d
        else:
            return np.sum(self.gpath.length)

    def bounds(self):
        return self.gpath.geometry.iloc[0].bounds

    def rotate(self, theta: float) -> 'GeoPath':
        """
        Rotate the path around its centroid. Returns a new GeoPath object.

        Args:
            theta: (float) rotation angle, in radians, indexed from north, clockwise.

        Returns:
            GeoPath: the rotated path.
        """
        centroid = self.gpath.centroid.iloc[0]
        rotated = shapely.affinity.rotate(self.gpath.iloc[0], theta, origin=centroid, use_radians=True)

        return GeoPath(gpd.GeoSeries(rotated))

    def translate(self, t: Union[XY, List[float]]) -> 'GeoPath':
        """
        Translate the path by a vector t. Returns a new LocalPath object.

        Args:
            t: (XY) the translation vector.

        Returns:
            LocalPath: the translated path.
        """
        centroid = self.gpath.centroid.iloc[0]
        cgeo = GeoPoint(centroid.x, centroid.y, crs=self.crs)
        cll = cgeo.to_ll()
        local = self.to_local(cll)
        newlocal = local.translate(t)
        newgeo = newlocal.to_geo(cll, self.crs)
        return newgeo

    def __len__(self) -> int:
        """
        Gets the length of the GeoPath object.

        Returns:
            int: The length of the GeoPath object.
        """

        return len(self.gpath.get_coordinates())

    def __getitem__(self, item: int) -> GeoPoint:
        """
        Gets the item

        Args:
            item: (int) the item index

        Returns:
            GeoPoint: The GeoPoint at the given index.
        """
        ret = self.gpath[0].coords[item]
        return GeoPoint(ret[0], ret[1], crs=self.crs)


    def to_lls(self) -> List[LL]:
        """
        Converts the GeoPath to a list of LLs.

        Returns:
            [LL]: The list of LLs.
        """
        ll_path = self.gpath.to_crs("EPSG:4326")
        lls = []
        for p in ll_path[0].coords:
            lls.append(LL(p[1], p[0]))
        return lls
    def to_points(self) -> List[Tuple[float, float]]:
        """
        Converts the GeoPath to a list of points.

        Returns:
            [(x,y)]: The list of points.
        """
        points = []
        for p in self:
            points.append((p.x, p.y))
        return points

    def to_geopoints(self):
        """
        Converts the GeoPath to a list of GeoPoints.

        Returns:
            [GeoPoint]: The list of GeoPoints.
        """
        return [GeoPoint(p[0], p[1], crs=self.crs) for p in self.gpath[0].coords]

    def to_file(self, filename: str) -> None:
        """
        Writes the GeoPath to a file.

        Args:
            filename: (str) The filename to write to.

        Returns:
            None
        """
        return self.gpath.to_file(filename)

    def to_csv(self, filename: str) -> None:
        """
        Writes the GeoPath to a csv file.

        Args:
            filename: (str) The filename to write to.

        Returns:
            None
        """
        df = pd.DataFrame(self.to_points(), columns=["x","y"])
        df.to_csv(filename, index=False)

    @classmethod
    def from_file(cls, filename: str) -> 'GeoPath':
        """
        Reads the GeoPath from a file, constructs a class

        Args:
            filename: (str) The filename to read from.

        Returns:
            GeoPath: The GeoPath
        """
        assert os.path.isfile(filename)
        gdf = gpd.read_file(filename)
        assert gdf.geometry.geom_type[0] == "LineString"
        return cls(gdf.geometry)

    def plot(self):
        """
        Plots the GeoPath.
        Returns:
            matplotlib.axes.Axes: The axes.
        """
        return self.gpath.plot()

    def explore(self, **kwargs):
        """
        Views the path in an interactive map

        Returns:
            The folium map
        """
        return self.gpath.explore(**kwargs)

    def to_crs(self, new_crs: str) -> 'GeoPath':
        """
        To a new crs

        Args:
            new_crs: (str, pyproj.CRS) The new crs

        Returns:
            GeoPath: The GeoPath in the new crs
        """
        return GeoPath(self.gpath.to_crs(new_crs))

    def to_local(self, origin: LL) -> 'LocalPath':
        """
        Converts to a local projection

        Args:
            origin: (LL) The origin

        Returns:
            LocalPath: The LocalPath in NED
        """
        # lls = self.to_lls()
        # neds = [pymap3d.geodetic2ned(ll.lat, ll.lon, 0, origin.lat, origin.lon, 0) for ll in lls]
        # return LocalPath(xys=neds)
        t = get_geo_to_local_transformer(src_crs=self.crs, local_origin_ll=origin)
        enus = [t.transform(p[0],p[1]) for p in self.gpath[0].coords]
        xys = [XY(enu[0],enu[1]) for enu in enus]
        return LocalPath(xys)

    def asarray(self) -> np.ndarray:
        return np.array(self.gpath.coords)

    def sample_points_along_path(self, d: float) -> 'GeoPath':
        """
        Subsamples along the path with interval d

        Args:
            d: (float) the interval distance in m

        Returns:
            GeoPath: the subsampled GeoPath.
        """
        centroid = self.gpath.centroid.iloc[0]
        origin_ll = GeoPoint(centroid.x, centroid.y, crs=self.crs).to_ll()
        local_path = self.to_local(origin_ll)
        subsampled_path = local_path.sample_points_along_path(d)
        return subsampled_path.to_geo(origin_ll, self.crs)




class GeoPoints:
    """
    A class to represent a list of points in geographical space.

    Examples:

            # Create a list of points from an existing GeoSeries
            gpoints = gpd.GeoSeries([shapely.geometry.Point(151, -30),
                                    shapely.geometry.Point(152, -31)],
                                    crs="EPSG:4326")
            points = GeoPoints(gpoints)

            # Create a list of points from a list of LLs
            lls = [LL(-30,151), LL(-31,151)]
            points = GeoPoints(lls)

            # Create a list of points from a list of points
            points = GeoPoints(points=[(151, -30), (152, -31)], crs="EPSG:4326")
    """
    def __init__(self, points: Union[List[XY], List[Iterable], List[GeoPoint], List[LL]], crs=None):
        """
        Constructs all the necessary attributes for the GeoPoints object.

        Args:
            points: geopandas.GeoSeries
            points:  [(x,y),] | GeoPoint
            lls: [LL,]
            crs: str
                The coordinate reference system of the points.
        """
        if isinstance(points, gpd.GeoSeries):
            gpoints = points
        elif isinstance(points, list):
            if isinstance(points[0], GeoPoint):
                crs = points[0].crs
                gpoints = gpd.GeoSeries([shapely.geometry.Point(p.x, p.y) for p in points], crs=crs)
            elif isinstance(points[0], LL):
                if crs is None:
                    crs = "EPSG:4326"
                gpoints = gpd.GeoSeries([shapely.geometry.Point(ll.lon, ll.lat) for ll in points], crs=crs)
            elif isinstance(points[0], (np.ndarray, list)):
                if crs is None:
                    raise ValueError("Need to specify crs if points are not LL or GeoPoint")
                gpoints = gpd.GeoSeries([shapely.geometry.Point(p[0], p[1]) for p in points], crs=crs)
            else:
                raise ValueError("invalid type contained in list")
        else:
            raise ValueError("Invalid input")
        self.gpoints = gpoints
        self.crs = self.gpoints.crs

    @classmethod
    def from_file(cls, filename: str) -> 'GeoPoints':
        """
        Reads the GeoPoints from a file, constructs a class

        Args:
            filename: (str) The filename to read from.

        Returns:
            GeoPoints: The GeoPoints
        """
        assert os.path.isfile(filename)
        gdf = gpd.read_file(filename)
        assert gdf.geometry.geom_type[0] == "Point"
        return cls(gpd.GeoSeries(gdf.geometry))

    def __repr__(self) -> str:
        """
        The string representation of the GeoPoints object.
        Returns:
            str: The string representation of the GeoPoints object.
        """
        return self.gpoints.__repr__()

    def __len__(self) -> int:
        """
        The number of points
        Returns:
            int: The number of points
        """
        return len(self.gpoints)

    def __getitem__(self, item: int) -> GeoPoint:
        """
        Gets the item
        Args:
            item: (int) the item index

        Returns:
            GeoPoint: The GeoPoint at the given index.
        """
        # get the point from the geoseries
        try:
            ret = self.gpoints.geometry[item]
            return GeoPoint(ret.x, ret.y, crs=self.crs)
        except (IndexError, KeyError):
            raise IndexError("Index out of scope for GeoPoints")
    def to_lls(self) -> List[LL]:
        """
        Converts the GeoPoints to a list of LLs.
        Returns:
            [LL]: The list of LLs.
        """
        return [LL(p[0], p[1]) for p in self.gpoints]

    def to_points(self) -> List[Tuple[float, float]]:
        """
        Converts the GeoPoints to a list of points.
        Returns:
            [(x,y)]: The list of points.
        """
        points = self.gpoints.to_numpy()
        return [(p[0], p[1]) for p in points]

    def to_geopoints(self) -> List[GeoPoint]:
        """
        Converts the GeoPoints to a list of GeoPoints.
        Returns:
            [GeoPoint]: The list of GeoPoints.
        """
        return [GeoPoint(p.x, p.y, crs=self.crs) for p in self.gpoints]

    def to_file(self, filename: str) -> None:
        """
        Writes the GeoPoints to a file.
        Args:
            filename: (str)
        Returns:
            None
        """
        return self.gpoints.to_file(filename)

    def plot(self):
        """
        Plots the GeoPoints.
        Returns:
            matplotlib.axes.Axes: The axes.

        """
        return self.gpoints.plot()

    def to_crs(self, new_crs: str) -> 'GeoPoints':
        """
        Converts the GeoPoints to a new coordinate reference system.
        Args:
            new_crs: (str, pyproj.CRS) The new coordinate reference system.

        Returns:
            GeoPoints: The GeoPoints in the new coordinate reference system.
        """
        return GeoPoints(self.gpoints.to_crs(new_crs))

    def to_local(self, origin: LL) -> 'LocalArea':
        """
        Converts to a local projection
        Args:
            origin: (LL) The origin

        Returns:
            LocalPath: The LocalPath in NED
        """
        # lls = self.to_lls()
        # neds = [pymap3d.geodetic2ned(ll.lat, ll.lon, 0, origin.lat, origin.lon, 0) for ll in lls]
        # return LocalArea(xys=neds)
        t = get_geo_to_local_transformer(src_crs=self.crs, local_origin_ll=origin)
        enus = [t.transform(p[0], p[1]) for p in self.gpoints.geometry]
        xys = [XY(enu[0], enu[1]) for enu in enus]
        return LocalPath(xys)

    def asarray(self) -> np.ndarray:
        """
        Returns the points as an array
        Returns:
            np.ndarray: The array
        """
        return np.array(self.gpoints.coords)

    def explore(self, **kwargs):
        """
        Views the path in an interactive map

        Returns:
            The folium map
        """
        return self.gpoints.explore(**kwargs)




class GeoArea(GeoType):
    """A class to represent a geographical survey area.

    Examples:

        # Create an area from an existing polygon
        poly = shapely.geometry.Polygon([(151,-30), (151,-31), (150,-31), (150,-30)])
        gpoly = gpd.GeoSeries(poly, crs="EPSG:4326")
        area = GeoArea(gpoly)

        # Create an area from a list of LLs
        lls = [LL(-30,151), LL(-31,151), LL(-31,150), LL(-31,150)]
        area = GeoArea(lls)

        # Create an area from a list of points
        points = [(151,-30), (151,-31), (150,-31), (150,-30)]
        area = GeoArea(points, crs="EPSG:4326")

    """

    def __init__(self, data, crs=None):
        """Constructs all the necessary attributes for the area object.

        Parameters
        ----------
        gpoly : geopandas.GeoSeries
            The polygon defining the survey area.
        lls: [LL,]
        points: [(x,y),]
        crs: str
            The coordinate reference system of the points.
        """
        if isinstance(data, gpd.GeoSeries):
            gpoly = data
            if gpoly.crs is None:
                if crs is None:
                    raise ValueError("Need to specify crs if geoseries does not have one")
                gpoly.crs = crs
            crs = gpoly.crs

        elif isinstance(data, list):
            p0 = data[0]
            if isinstance(p0, GeoPoint):
                crs = p0.crs
                gpoly = gpd.GeoSeries(shapely.geometry.Polygon([(p.x, p.y) for p in data]), crs=crs)
            elif isinstance(p0, LL):
                if crs is None:
                    crs = "EPSG:4326"
                gpoly = gpd.GeoSeries(shapely.geometry.Polygon([(ll.lon, ll.lat) for ll in data]), crs=crs)
            elif isinstance(p0, (np.ndarray, list, tuple)):
                if crs is None:
                    raise ValueError("Need to specify crs if points are not LL or GeoPoint")
                gpoly = gpd.GeoSeries(shapely.geometry.Polygon(data), crs=crs)
            else:
                raise ValueError("invalid type contained in list")
        elif isinstance(data, np.ndarray):
            if crs is None:
                raise ValueError("Need to specify crs if points are not LL or GeoPoint")
            gpoly = gpd.GeoSeries(shapely.geometry.Polygon(data), crs=crs)
        elif isinstance(data, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)):
            if crs is None:
                raise ValueError("Need to specify crs if points are not LL or GeoPoint")
            gpoly = gpd.GeoSeries(data, crs=crs)
        else:
            raise ValueError("Invalid input - input must be either a geoseris, list of GeoPoints, LLs or list of points")
        self.gpoly = gpoly
        self.poly = gpoly.iloc[0]
        self.crs = gpoly.crs


    @classmethod
    def from_file(cls, filename: str, index: int = 0) -> 'GeoArea':
        """
        Reads the GeoArea from a file, constructs a class

        Args:
            filename: (str) The filename to read from.
            index: (int) the index of the geometry in the geodataframe.

        Returns:
            GeoArea: The GeoArea
        """
        assert os.path.isfile(filename)
        gdf = gpd.read_file(filename)
        assert gdf.geometry.geom_type[index] == "Polygon"
        return cls(gpd.GeoSeries(gdf.geometry))

    def __repr__(self) -> str:
        """
        The string representation of the GeoArea object.
        Returns:
            str: The string representation of the GeoArea object.
        """
        return self.gpoly.__repr__()

    def __len__(self) -> int:
        """
        The number of points
        Returns:
            int: The number of points
        """
        return len(self.gpoly)

    def __getitem__(self, item) -> GeoPoint:
        """
        Gets the item
        Args:
            item: (int) the item index

        Returns:
            GeoPoint: The GeoPoint at the given index.
        """
        # ret = self.gpoly.coords[item]
        poly  = self.gpoly.iloc[0]
        ret = poly.exterior.coords[item]
        return GeoPoint(ret[0], ret[1], crs=self.crs)

    def to_points(self) -> List[Tuple[float, float]]:
        """
        Converts the GeoArea to a list of points.
        Returns:
            [(x,y)]: The list of points.
        """
        return np.array(self.gpoly.iloc[0].exterior.coords)

    def to_lls(self) -> List[LL]:
        """
        Converts the GeoArea to a list of LLs defining the exterior coords
        Returns:
            [LL]: The list of LLs.
        """
        llarea = self.to_crs("EPSG:4326")
        return [LL(p[1], p[0]) for p in llarea.gpoly.iloc[0].exterior.coords]
    def to_file(self, filename) -> None:
        """
        Writes the GeoArea to a file.
        Args:
            filename: (str) The filename to write to.

        Returns:
            None
        """
        return self.gpoly.to_file(filename)

    def plot(self):
        """
        Plots the GeoArea.
        Returns:
            matplotlib.axes.Axes: The axes.
        """
        return self.gpoly.plot()

    def explore(self, **kwargs):
        """
        Views the path in an interactive map

        Returns:
            The folium map
        """
        return self.gpoly.explore(**kwargs)

    def to_crs(self, new_crs: Union[str, pyproj.CRS]) -> 'GeoArea':
        """Returns a new Area object with a different CRS."""
        return GeoArea(self.gpoly.to_crs(new_crs))

    def _get_coordinates(self) -> pd.DataFrame:
        """
        Gets the coordinates of the GeoArea object.
        Returns:
            np.ndarray: The coordinates of the GeoArea object.
        """
        coords = self.gpoly.get_coordinates()
        return coords

    def to_local(self, origin: LL) -> 'LocalArea':
        """
        Converts to a local projection
        Args:
            origin: (LL) The origin

        Returns:
            LocalArea: The LocalArea in NED
        """
        t = get_geo_to_local_transformer(src_crs=self.crs, local_origin_ll=origin)
        coords = np.array(self.gpoly.get_coordinates()).tolist()
        enus = [t.transform(p[0], p[1]) for p in coords]
        xys = [XY(enu[0], enu[1]) for enu in enus]  # swap to NED
        return LocalArea(xys)

    def asarray(self) -> np.ndarray:
        """
        Returns the points as an array
        Returns:
            np.ndarray: The array
        """
        return np.array(self.gpoly.exterior.coords)

    def random_point_geo(self) -> GeoPoint:
        """
        Returns a random point within the survey area.
        Returns:
            GeoPoint: The random point.
        """
        p = self.gpoly.sample_points(1).iloc[0]
        return GeoPoint(p.x, p.y, crs=self.crs)

    def random_point_ll(self) -> LL:
        """
        Returns a random point within the survey area as LL.
        Returns:
            LL: The random point.
        """
        p = self.random_point_geo()
        return p.to_ll()

    def centroid(self) -> GeoPoint:
        """
        Returns the centre of the survey area.

        Returns:
            GeoPoint: The centroid.
        """
        p = self.gpoly.centroid.iloc[0]
        return GeoPoint(p.x, p.y, crs=self.crs)

    def centroid_ll(self) -> LL:
        """
        Returns the centre of the survey area as LL

        Returns:
            LL: The centroid.
        """
        p = self.centroid()
        return p.to_ll()

    def contains(self, other: Union[GeoPoint, GeoPoints, GeoPath]) -> bool:
        """
        Checks if the GeoArea contains another GeoArea, GeoPoint or GeoPoints.
        Args:
            other: (GeoPoint, GeoPoints, GeoArea) The other object.

        Returns:
            bool: True if the GeoArea contains the other object.
        """
        if isinstance(other, GeoPoint):
            return self.gpoly.contains(shapely.geometry.Point(other.x, other.y)).iloc[0]
        elif isinstance(other, GeoPoints):
            return all(self.gpoly.contains(other.gpoints))
        elif isinstance(other, GeoPath):
            return all(self.gpoly.contains(other.gpath))

    def bounds(self):
        return self.gpoly.geometry.iloc[0].bounds


def combine_geopaths(paths: List[GeoPath]) -> gpd.GeoDataFrame:
    combined_geo_series = gpd.GeoSeries(pd.concat([c.gpath for c in paths], ignore_index=True))
    gdf = gpd.GeoDataFrame(geometry=combined_geo_series)
    return gdf
# GeoType = Union[GeoPoint, GeoPoints, GeoPath, GeoArea]


def load_geometry(path: str) -> 'GeoType':
    """
    Loads a geometry from a file. Can be any of GeoPoint, GeoPoints, GeoPath, GeoArea

    Args:
        path:

    Returns:

    """
    if not os.path.isfile(path):
        raise ValueError("Invalid path")

    gdf = gpd.read_file(path)
    if gdf.iloc[0].geometry.geom_type == "Point":
        if len(gdf) > 1:
            ps = [GeoPoint(gdf.iloc[i].geometry.x, gdf.iloc[i].geometry.y, crs=gdf.crs) for i in range(len(gdf))]
            return GeoPoints(ps)
        else:
            return GeoPoint(gdf.iloc[0].geometry.x, gdf.iloc[0].geometry.y, crs=gdf.crs)
    elif gdf.iloc[0].geometry.geom_type == "MultiPoint":
        return GeoPoints.from_file(path)
    elif gdf.iloc[0].geometry.geom_type == "LineString":
        return GeoPath.from_file(path)
    elif gdf.iloc[0].geometry.geom_type == "Polygon":
        return GeoArea.from_file(gdf)

def set_to_points(surveys, subsample_dist):
    """
    Converts a set of surveys to a set of points. This is useful for evaluating the performance of a planner.
    Args:
        surveys: The survey set
        subsample_dist: The distance along each path to subsample

    Returns:
        List[GeoPoint]: The set of points
    """
    points = []
    for survey in surveys:
        if isinstance(survey, GeoPoint):
            points.append(survey)
        elif isinstance(survey, GeoPath):
            subsample_path = survey.sample_points_along_path(subsample_dist)
            points.extend(subsample_path.to_geopoints())
        elif isinstance(survey, GeoPoints):
            sps = survey.to_points()
            points.extend(sps)
        else:
            raise ValueError("Not valid")
    return points