import argparse
import re
import textwrap
from typing import Iterable, Optional, Tuple

from .geometry import LL


def _split_lines(text: str, line_separator: str) -> Iterable[str]:
    if line_separator and line_separator != "\n":
        lines = text.split(line_separator)
    else:
        lines = text.splitlines()
    return [line.strip() for line in lines if line.strip()]


def _parse_token(token: str) -> Tuple[float, Optional[str]]:
    token = token.strip()
    hemisphere = None
    if token[:1] in "NSEW":
        hemisphere = token[:1]
        token = token[1:]
    elif token[-1:] in "NSEW":
        hemisphere = token[-1:]
        token = token[:-1]

    axis_hint = None
    if hemisphere in ("N", "S"):
        axis_hint = "lat"
    elif hemisphere in ("E", "W"):
        axis_hint = "lon"

    sign = 1
    if token[:1] in "+-":
        sign = -1 if token[:1] == "-" else 1
        token = token[1:]

    token = token.replace("°", "d")
    numbers = re.findall(r"\d+(?:\.\d+)?", token)
    if not numbers:
        raise ValueError(f"Invalid coordinate token: '{token}'")

    deg = abs(float(numbers[0]))
    minutes = float(numbers[1]) if len(numbers) > 1 else 0.0
    seconds = float(numbers[2]) if len(numbers) > 2 else 0.0

    value = deg + minutes / 60.0 + seconds / 3600.0
    if hemisphere in ("S", "W"):
        value = -value
    elif hemisphere in ("N", "E"):
        value = value
    else:
        value = value * sign
    return value, axis_hint


def _parse_lat_lon(line: str, swap: bool = False) -> LL:
    tokens = re.split(r"[,\s]+", line.strip())
    tokens = [token for token in tokens if token]
    if len(tokens) != 2:
        raise ValueError(f"Expected two tokens, got {len(tokens)}: '{line}'")

    value_a, hint_a = _parse_token(tokens[0])
    value_b, hint_b = _parse_token(tokens[1])

    if hint_a and hint_b:
        if hint_a == hint_b:
            raise ValueError(f"Ambiguous coordinates: '{line}'")
        lat = value_a if hint_a == "lat" else value_b
        lon = value_a if hint_a == "lon" else value_b
    elif hint_a or hint_b:
        if hint_a == "lat" or hint_b == "lat":
            lat = value_a if hint_a == "lat" else value_b
            lon = value_b if hint_a == "lat" else value_a
        else:
            lon = value_a if hint_a == "lon" else value_b
            lat = value_b if hint_a == "lon" else value_a
    else:
        if swap:
            lon, lat = value_a, value_b
        else:
            lat, lon = value_a, value_b

    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Latitude out of range: {lat}")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"Longitude out of range: {lon}")
    return LL(lat=lat, lon=lon)


def _format_degrees_decimal(value: float, precision: int) -> str:
    return f"{value:.{precision}f}"


def _format_degrees_minutes(value: float, precision: int) -> str:
    abs_val = abs(value)
    deg = int(abs_val)
    minutes = (abs_val - deg) * 60.0
    minutes = round(minutes, precision)
    if minutes >= 60.0:
        minutes = 0.0
        deg += 1
    sign = "-" if value < 0 else ""
    return f"{sign}{deg}d{minutes:.{precision}f}'"


def _format_degrees_minutes_seconds(value: float, precision: int) -> str:
    abs_val = abs(value)
    deg = int(abs_val)
    minutes_full = (abs_val - deg) * 60.0
    minutes = int(minutes_full)
    seconds = (minutes_full - minutes) * 60.0
    seconds = round(seconds, precision)
    if seconds >= 60.0:
        seconds = 0.0
        minutes += 1
    if minutes >= 60:
        minutes = 0
        deg += 1
    sign = "-" if value < 0 else ""
    return f"{sign}{deg}d{minutes:02d}'{seconds:.{precision}f}\""


def _format_output(ll: LL, output_format: str, precision: int) -> str:
    if output_format == "degrees":
        return f"{_format_degrees_decimal(ll.lat, precision)} {_format_degrees_decimal(ll.lon, precision)}"
    if output_format == "degrees_minutes":
        return f"{_format_degrees_minutes(ll.lat, precision)} {_format_degrees_minutes(ll.lon, precision)}"
    if output_format == "degrees_minutes_seconds":
        return (
            f"{_format_degrees_minutes_seconds(ll.lat, precision)} "
            f"{_format_degrees_minutes_seconds(ll.lon, precision)}"
        )
    raise ValueError(f"Unknown output format: {output_format}")


def get_args():
    examples = textwrap.dedent(
        """\
        Examples:
          # Degrees decimal -> degrees minutes
          geoconvert "33.3 44.4" -m
          geoconvert "33.3,44.4" -m

          # Degrees minutes -> degrees decimal
          geoconvert "33d18' 44d24'" -d

          # Degrees minutes seconds -> degrees decimal
          geoconvert "33d18'30\" 44d24'45\"" -d
        """
    )
    parser = argparse.ArgumentParser(
        "geoconvert",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples,
    )
    parser.add_argument("inputstr", nargs="?", help="Input coordinate string")
    parser.add_argument(
        "--input-string",
        dest="input_string",
        help="Input coordinate string (overrides positional input)",
    )
    parser.add_argument("--input-file", dest="input_file", help="Input file")
    parser.add_argument(
        "--output-degrees-minutes",
        "-m",
        action="store_true",
        help="Output degrees and decimal minutes",
    )
    parser.add_argument(
        "--output-degrees-minutes-seconds",
        "-s",
        action="store_true",
        help="Output degrees, minutes, and seconds",
    )
    parser.add_argument(
        "--output-degrees-decimal",
        "-d",
        action="store_true",
        help="Output decimal degrees",
    )
    parser.add_argument("--swap", "-w", action="store_true", help="Swap lon/lat order")
    parser.add_argument(
        "--precision",
        "-p",
        type=int,
        default=6,
        help="Decimal precision for minutes/seconds/decimal output",
    )
    parser.add_argument("--output-file", dest="output_file", help="Output file")
    parser.add_argument(
        "--line-separator",
        dest="line_separator",
        default="\n",
        help="Line separator used for input/output",
    )
    return parser.parse_args()


def geoconvert_main():
    args = get_args()

    output_flags = [
        args.output_degrees_decimal,
        args.output_degrees_minutes,
        args.output_degrees_minutes_seconds,
    ]
    if sum(bool(flag) for flag in output_flags) > 1:
        raise ValueError("Select only one output format")

    if args.output_degrees_minutes:
        output_format = "degrees_minutes"
    elif args.output_degrees_minutes_seconds:
        output_format = "degrees_minutes_seconds"
    else:
        output_format = "degrees"

    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as handle:
            input_text = handle.read()
    elif args.input_string is not None:
        input_text = args.input_string
    elif args.inputstr is not None:
        input_text = args.inputstr
    else:
        input_text = ""

    if not input_text:
        raise ValueError("No input provided")

    lines = _split_lines(input_text, args.line_separator)
    outputs = []
    for line in lines:
        ll = _parse_lat_lon(line, swap=args.swap)
        outputs.append(_format_output(ll, output_format, args.precision))

    output_text = args.line_separator.join(outputs)
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as handle:
            handle.write(output_text)
    else:
        print(output_text)


if __name__ == "__main__":
    geoconvert_main()