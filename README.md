# geotypes

geotypes is a python package for making it easy to work with geospatial data, particularly useful for geospatial planning and analysis. It wraps GeoPandas and rasterio with simple data types to make it easier and more pythonic to work with for some applications.

Visit the [documentation](https://jacksonhshields.github.io/geotypes/) for installation, getting started, tutorials and API reference.



Geotypes adds some high-level wrappers around geopandas and shapely to allow for easier use. They allow easy conversion between georeference schemes, better tracking of what coordinate system the variables are in and easy conversion between types.
These include:

* GeoPoint
* LL
* GeoPath
* GeoPoints
* GeoArea
* XY
* LocalPath
* LocalPoints
* LocalArea
* Raster



## Installation
Use pip to install geotypes. 

GDAL is not added to the requirements. If you already have libgdal-dev installed, you can use:
```
gdalversion=$(gdalinfo --version | cut -d\  -f2 | sed s'#,##') && pip3 install GDAL==$gdalversion
```



With GDAL installed or not, you can install geotypes with:
```bash
pip install git@github.com:jacksonhshields/geotypes.git

```



## Tutorials

To get started with geotypes, check out the tutorials, which run you through the data types and how to use them.


## Basic Usage

### GeoPoint
The GeoPoint is an x,y coordinate with an associated Coordinate Reference System (CRS). It allows easy conversion to LL and XY data types.

```
p = geotypes.geometry.GeoPoint(x=151.28826000, y=-33.79798000, crs="epsg:4326") # initialise a lat, lon centred in Manly.
print("GeoPoint", p)
ll = p.to_ll() # Converts to lat,lon data type
print("In LL", ll)
p_utm = p.to_crs("EPSG:32756")  # converts to utm zone 56S
print("In UTM Zone 56S", p_utm)
```

### LL

## LL
The lat,lon data type is there for convience and to make it explicit about what is latitude, what is longitude.
```
ll = geotypes.geometry.LL(-33.79798000,151.28826000)
print(ll)
# Convert to geopoint
p = ll.to_geopoint()
print(p)
```


### GeoPath
A path in geographic space, effectively a geopandas geoseries / shapely linestring with an associated CRS. It is commonly used to represent sampling transects (such as AUV transects). It has some convient functions to collect samples at set distances along the path - which can be done no-matter what coordinate system the path is in.
```
# Define a set of points. This is a simple broad grid in CRS "EPSG:32756"
points = [(342041.99358729343, 6259392.849573873),
 (342495.27623430215, 6259603.771833958),
 (342579.64503957686, 6259422.458960008),
 (342126.36269032373, 6259211.536449324),
 (342210.73189359176, 6259030.223443877),
 (342664.01394509675, 6259241.146205157)]

# create the geopath object
geopath = geotypes.geometry.GeoPath(points, crs="EPSG:32756")
print(geopath)

# Iterating over it returns a geopoint.
print("Geopoints:")
for p in geopath:
    print(p)
print("\n")

# Easily convert to other crs
geopath4326 = geopath.to_crs("EPSG:4326")
print(geopath4326)

# Sample points along the geopath
geopath_subsampled = geopath.sample_points_along_path(d=20) # sample a point every 20 meters.
print("Number of points in original {}, in subsampled {}".format(len(geopath), len(geopath_subsampled)))
```


### GeoArea

A GeoArea is a wrapper around a geopandas GeoSeries with just one element, a polygon or multipolygon. They are used within the situ framework to represent sampling bounds or focal areas. As such, they have easy access to random sampling.

```
# Define a geoarea
geoarea = geotypes.geometry.GeoArea([(151.28412747953516, -33.78781691040293),
 (151.30572252046485, -33.78781691040293),
 (151.30572478467255, -33.805848131225765),
 (151.28412521532746, -33.805848131225765),
 (151.28412747953516, -33.78781691040293)], crs='EPSG:4326')

# get some samples from within it - these are GeoPoints:
for n in range(5):
    print(geoarea.random_point_geo())
# you can get also get points directly as LLs, regardless of the CRS.
for n in range(5):
    print(geoarea.random_point_ll())
```

### Local Equivalents
There are local equivalents to the classes presented above. These include:
- XY: Equivalent of GeoPoint
- LocalPoints: Equivalent of GeoPoints
- LocalPath: Equivalent of GeoPath
- LocalArea: Equivalent of GeoArea
These are just wrappers around the shapely geometry classes, but offer some extra functionality, including being able to convert to and from the Geo equivalents.

```
origin_ll = geoarea.centroid_ll()
# Convert the position to local
xy = p.to_local(origin_ll)
# Now go back to GeoPoint, this time with a different CRS.
pll = xy.to_geo(origin_ll, "epsg:4326")
# Print all to compare
print(p, xy, pll)


# Convert a GeoPath to a LocalPath
localpath = geopath.to_local(origin_ll)
# And go back again
geopath2 = localpath.to_geo(origin_ll, geopath.crs)
```

### Rasters

There is an abstraction of rasterio Rasters to make it easier to work with rasters. You can use the Raster.get_value function with any geotypes geometry object to get the value or values at that point.

```
from geotypes.rasters import Raster
r = Raster('manly_uw.tif')
ll = LL(-33.79931, 151.29412)
from geotypes.geometry import LL
ll = LL(-33.79931, 151.29412)
r.get_value(ll)
>>> -7.871370315551758
```


## CLI: geoconvert
`geoconvert` converts geographic coordinates between decimal degrees, degrees+minutes, and degrees+minutes+seconds. It is based on GeoConvert, but also does degrees,minutes.

Examples:
```bash
# Degrees decimal -> degrees minutes
geoconvert "33.3 44.4" -m
geoconvert "33.3,44.4" -m

# Degrees minutes -> degrees decimal
geoconvert "33d18' 44d24'" -d

# Degrees minutes seconds -> degrees decimal
geoconvert "33d18'30\" 44d24'45\"" -d
```

Notes:
- Input accepts space or comma separated lat/lon.
- Hemisphere prefixes/suffixes are supported (e.g. `N33.3 E44.4`).
- Use `-w` to swap lon/lat order when coordinates are ambiguous.
