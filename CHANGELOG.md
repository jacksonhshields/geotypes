# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-07-02

### Fixed

- `LL.from_degrees_minutes` and `LL.from_degrees_minutes_seconds` dropped the
  minutes/seconds when the degrees component was 0 (e.g. any point within 1°
  of the equator or prime meridian), due to `np.sign(0)` returning 0.
- `LL.print_degrees_minutes` and `LL.print_degrees_minutes_second` lost the
  negative sign for coordinates between -1° and 0°.

### Added

- `dms_to_decimal` helper for converting degrees/minutes/seconds to decimal
  degrees. A negative sign on any component marks the whole coordinate as
  negative, so sub-degree southern/western coordinates can now be expressed,
  e.g. `LL.from_degrees_minutes(0, -30, 0, -15)` for 0°30'S 0°15'W.
- Test suite covering coordinate format conversions (decimal degrees,
  degrees-minutes, degrees-minutes-seconds) in the `LL` class and the
  `geoconvert` CLI, including round trips.

## [0.1.1] - 2026-01-23

### Added

- `geoconvert` command-line tool for converting coordinates between decimal
  degrees, degrees-minutes, and degrees-minutes-seconds formats, with support
  for hemisphere letters (N/S/E/W), file input/output, and configurable
  precision.
- `LL.from_degrees_minutes` and `LL.from_degrees_minutes_seconds`
  constructors, and `print_degrees_*` output methods.

## [0.1.0] - 2025-05-20

### Added

- Initial release: geometry types (`LL`, `XY`, `GeoPoint`, `GeoPoints`,
  `GeoPath`, `GeoArea`, `LocalPath`, `LocalPoints`, `LocalArea`), raster
  utilities, and feature helpers for working with geospatial data.
