"""Tests for coordinate conversions between decimal degrees, degrees-minutes,
and degrees-minutes-seconds, in both the LL class and the geoconvert CLI."""
import math
import re

import pytest

from geotypes.geometry import LL
from geotypes.cli import (
    _parse_token,
    _parse_lat_lon,
    _format_degrees_decimal,
    _format_degrees_minutes,
    _format_degrees_minutes_seconds,
    _format_output,
)


# ---------------------------------------------------------------------------
# LL.from_degrees_minutes / from_degrees_minutes_seconds (DM/DMS -> decimal)
# ---------------------------------------------------------------------------

class TestLLFromDegreesMinutes:
    def test_positive(self):
        ll = LL.from_degrees_minutes(33, 18.0, 151, 30.0)
        assert ll.lat == pytest.approx(33.3)
        assert ll.lon == pytest.approx(151.5)

    def test_negative(self):
        ll = LL.from_degrees_minutes(-33, 18.0, -151, 30.0)
        assert ll.lat == pytest.approx(-33.3)
        assert ll.lon == pytest.approx(-151.5)

    def test_mixed_hemispheres(self):
        ll = LL.from_degrees_minutes(-33, 51.6, 151, 12.6)
        assert ll.lat == pytest.approx(-33.86)
        assert ll.lon == pytest.approx(151.21)

    def test_zero_minutes(self):
        ll = LL.from_degrees_minutes(45, 0.0, -90, 0.0)
        assert ll.lat == pytest.approx(45.0)
        assert ll.lon == pytest.approx(-90.0)

    def test_zero_degrees_positive(self):
        ll = LL.from_degrees_minutes(0, 30.0, 0, 15.0)
        assert ll.lat == pytest.approx(0.5)
        assert ll.lon == pytest.approx(0.25)

    def test_zero_degrees_negative_minutes(self):
        # 0d30'S, 0d15'W expressed via negative minutes
        ll = LL.from_degrees_minutes(0, -30.0, 0, -15.0)
        assert ll.lat == pytest.approx(-0.5)
        assert ll.lon == pytest.approx(-0.25)


class TestLLFromDegreesMinutesSeconds:
    def test_positive(self):
        ll = LL.from_degrees_minutes_seconds(33, 18, 36.0, 151, 30, 18.0)
        assert ll.lat == pytest.approx(33.31)
        assert ll.lon == pytest.approx(151.505)

    def test_negative(self):
        ll = LL.from_degrees_minutes_seconds(-33, 18, 36.0, -151, 30, 18.0)
        assert ll.lat == pytest.approx(-33.31)
        assert ll.lon == pytest.approx(-151.505)

    def test_zero_seconds(self):
        ll = LL.from_degrees_minutes_seconds(10, 30, 0.0, 20, 45, 0.0)
        assert ll.lat == pytest.approx(10.5)
        assert ll.lon == pytest.approx(20.75)

    def test_zero_degrees_negative_components(self):
        # 0d30'0"S, 0d0'30"W expressed via a negative minutes/seconds component
        ll = LL.from_degrees_minutes_seconds(0, -30, 0.0, 0, 0, -30.0)
        assert ll.lat == pytest.approx(-0.5)
        assert ll.lon == pytest.approx(-30.0 / 3600.0)


# ---------------------------------------------------------------------------
# LL print methods (decimal -> DM/DMS, printed)
# ---------------------------------------------------------------------------

class TestLLPrintMethods:
    def test_print_degrees_decimal(self, capsys):
        LL(lat=-33.86, lon=151.21).print_degrees_decimal()
        out = capsys.readouterr().out
        assert "-33.86" in out
        assert "151.21" in out

    def test_print_degrees_minutes(self, capsys):
        LL(lat=-33.86, lon=151.21).print_degrees_minutes()
        out = capsys.readouterr().out
        # -33.86 => -33 deg 51.6 min ; 151.21 => 151 deg 12.6 min
        assert "-33°" in out
        assert "151°" in out
        minutes = [float(m) for m in re.findall(r"(\d+(?:\.\d+)?)'", out)]
        assert minutes[0] == pytest.approx(51.6)
        assert minutes[1] == pytest.approx(12.6)

    def test_print_degrees_minutes_seconds(self, capsys):
        LL(lat=33.31, lon=151.505).print_degrees_minutes_second()
        out = capsys.readouterr().out
        # 33.31 => 33 deg 18 min 36 sec ; 151.505 => 151 deg 30 min 18 sec
        assert "33°" in out
        assert "18'" in out
        assert "151°" in out
        assert "30'" in out


# ---------------------------------------------------------------------------
# CLI token parsing (DM/DMS strings -> decimal)
# ---------------------------------------------------------------------------

class TestParseToken:
    def test_decimal_degrees(self):
        value, hint = _parse_token("33.3")
        assert value == pytest.approx(33.3)
        assert hint is None

    def test_negative_decimal_degrees(self):
        value, _ = _parse_token("-33.3")
        assert value == pytest.approx(-33.3)

    def test_degrees_minutes(self):
        value, _ = _parse_token("33d18'")
        assert value == pytest.approx(33.3)

    def test_degrees_minutes_unicode(self):
        value, _ = _parse_token("33°18'")
        assert value == pytest.approx(33.3)

    def test_negative_degrees_minutes(self):
        value, _ = _parse_token("-33d18'")
        assert value == pytest.approx(-33.3)

    def test_degrees_minutes_seconds(self):
        value, _ = _parse_token("33d18'36\"")
        assert value == pytest.approx(33.31)

    def test_negative_degrees_minutes_seconds(self):
        value, _ = _parse_token("-33d18'36\"")
        assert value == pytest.approx(-33.31)

    def test_hemisphere_south_prefix(self):
        value, hint = _parse_token("S33d18'")
        assert value == pytest.approx(-33.3)
        assert hint == "lat"

    def test_hemisphere_east_suffix(self):
        value, hint = _parse_token("151d30'E")
        assert value == pytest.approx(151.5)
        assert hint == "lon"

    def test_hemisphere_west(self):
        value, hint = _parse_token("W71d03'")
        assert value == pytest.approx(-71.05)
        assert hint == "lon"

    def test_hemisphere_north(self):
        value, hint = _parse_token("N42d21'")
        assert value == pytest.approx(42.35)
        assert hint == "lat"

    def test_negative_zero_degrees_minutes(self):
        # 0d30' south of the equator
        value, _ = _parse_token("-0d30'")
        assert value == pytest.approx(-0.5)

    def test_invalid_token_raises(self):
        with pytest.raises(ValueError):
            _parse_token("abc")


class TestParseLatLon:
    def test_space_separated_decimal(self):
        ll = _parse_lat_lon("33.3 44.4")
        assert ll.lat == pytest.approx(33.3)
        assert ll.lon == pytest.approx(44.4)

    def test_comma_separated_decimal(self):
        ll = _parse_lat_lon("-33.3,151.5")
        assert ll.lat == pytest.approx(-33.3)
        assert ll.lon == pytest.approx(151.5)

    def test_degrees_minutes_pair(self):
        ll = _parse_lat_lon("33d18' 44d24'")
        assert ll.lat == pytest.approx(33.3)
        assert ll.lon == pytest.approx(44.4)

    def test_degrees_minutes_seconds_pair(self):
        ll = _parse_lat_lon("33d18'30\" 44d24'45\"")
        assert ll.lat == pytest.approx(33 + 18 / 60 + 30 / 3600)
        assert ll.lon == pytest.approx(44 + 24 / 60 + 45 / 3600)

    def test_swap(self):
        ll = _parse_lat_lon("151.5 -33.3", swap=True)
        assert ll.lat == pytest.approx(-33.3)
        assert ll.lon == pytest.approx(151.5)

    def test_hemisphere_letters_reordered(self):
        # lon first, lat second: hints should sort them correctly
        ll = _parse_lat_lon("151d30'E S33d18'")
        assert ll.lat == pytest.approx(-33.3)
        assert ll.lon == pytest.approx(151.5)

    def test_single_hemisphere_hint(self):
        ll = _parse_lat_lon("151d30'E 33.3")
        assert ll.lat == pytest.approx(33.3)
        assert ll.lon == pytest.approx(151.5)

    def test_ambiguous_hints_raise(self):
        with pytest.raises(ValueError):
            _parse_lat_lon("N33d18' S44d24'")

    def test_lat_out_of_range_raises(self):
        with pytest.raises(ValueError):
            _parse_lat_lon("91.0 100.0")

    def test_lon_out_of_range_raises(self):
        with pytest.raises(ValueError):
            _parse_lat_lon("45.0 181.0")

    def test_wrong_token_count_raises(self):
        with pytest.raises(ValueError):
            _parse_lat_lon("33.3")


# ---------------------------------------------------------------------------
# CLI formatting (decimal -> DM/DMS strings)
# ---------------------------------------------------------------------------

class TestFormatting:
    def test_format_decimal(self):
        assert _format_degrees_decimal(-33.3, 4) == "-33.3000"

    def test_format_degrees_minutes_positive(self):
        assert _format_degrees_minutes(33.3, 2) == "33d18.00'"

    def test_format_degrees_minutes_negative(self):
        assert _format_degrees_minutes(-33.3, 2) == "-33d18.00'"

    def test_format_degrees_minutes_small_negative(self):
        # between -1 and 0: the sign must survive
        assert _format_degrees_minutes(-0.5, 2) == "-0d30.00'"

    def test_format_degrees_minutes_rounding_carry(self):
        # 33.99999999 deg: minutes round to 60, must carry into degrees
        assert _format_degrees_minutes(33.9999999999, 2) == "34d0.00'"

    def test_format_dms_positive(self):
        assert _format_degrees_minutes_seconds(33.31, 1) == "33d18'36.0\""

    def test_format_dms_negative(self):
        assert _format_degrees_minutes_seconds(-33.31, 1) == "-33d18'36.0\""

    def test_format_dms_rounding_carry(self):
        # seconds round to 60 -> carry to minutes -> carry to degrees
        assert _format_degrees_minutes_seconds(33.9999999999, 1) == "34d00'0.0\""

    def test_format_output_degrees(self):
        out = _format_output(LL(lat=-33.3, lon=151.5), "degrees", 4)
        assert out == "-33.3000 151.5000"

    def test_format_output_unknown_raises(self):
        with pytest.raises(ValueError):
            _format_output(LL(lat=0, lon=0), "nope", 4)


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------

ROUND_TRIP_POINTS = [
    (33.3, 44.4),
    (-33.86, 151.21),
    (42.360278, -71.057778),
    (-0.5, -0.25),          # sub-degree negatives
    (0.0, 0.0),
    (89.999, 179.999),
    (-89.999, -179.999),
]


class TestRoundTrips:
    @pytest.mark.parametrize("lat,lon", ROUND_TRIP_POINTS)
    def test_decimal_to_dm_and_back(self, lat, lon):
        dm = _format_output(LL(lat=lat, lon=lon), "degrees_minutes", 6)
        ll = _parse_lat_lon(dm)
        assert ll.lat == pytest.approx(lat, abs=1e-6)
        assert ll.lon == pytest.approx(lon, abs=1e-6)

    @pytest.mark.parametrize("lat,lon", ROUND_TRIP_POINTS)
    def test_decimal_to_dms_and_back(self, lat, lon):
        dms = _format_output(LL(lat=lat, lon=lon), "degrees_minutes_seconds", 6)
        ll = _parse_lat_lon(dms)
        assert ll.lat == pytest.approx(lat, abs=1e-6)
        assert ll.lon == pytest.approx(lon, abs=1e-6)

    @pytest.mark.parametrize("lat,lon", ROUND_TRIP_POINTS)
    def test_ll_from_degrees_minutes_round_trip(self, lat, lon):
        """decimal -> (deg, min) components -> LL.from_degrees_minutes -> decimal"""
        def split(v):
            deg = math.trunc(v)
            minutes = abs(v - deg) * 60
            if deg == 0 and v < 0:
                minutes = -minutes  # sign carried by minutes when degrees is 0
            return deg, minutes

        lat_deg, lat_min = split(lat)
        lon_deg, lon_min = split(lon)
        ll = LL.from_degrees_minutes(lat_deg, lat_min, lon_deg, lon_min)
        assert ll.lat == pytest.approx(lat, abs=1e-9)
        assert ll.lon == pytest.approx(lon, abs=1e-9)

    @pytest.mark.parametrize("lat,lon", ROUND_TRIP_POINTS)
    def test_ll_from_dms_round_trip(self, lat, lon):
        """decimal -> (deg, min, sec) components -> LL.from_degrees_minutes_seconds"""
        def split(v):
            deg = math.trunc(v)
            minutes_full = abs(v - deg) * 60
            minutes = math.trunc(minutes_full)
            seconds = (minutes_full - minutes) * 60
            if deg == 0 and v < 0:
                # sign carried by the first nonzero component when degrees is 0
                if minutes:
                    minutes = -minutes
                else:
                    seconds = -seconds
            return deg, minutes, seconds

        lat_deg, lat_min, lat_sec = split(lat)
        lon_deg, lon_min, lon_sec = split(lon)
        ll = LL.from_degrees_minutes_seconds(lat_deg, lat_min, lat_sec, lon_deg, lon_min, lon_sec)
        assert ll.lat == pytest.approx(lat, abs=1e-9)
        assert ll.lon == pytest.approx(lon, abs=1e-9)
