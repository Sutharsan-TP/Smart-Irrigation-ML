import math

import pytest

from src.data_collection import ReadingAssembler, make_row, parse_line


@pytest.mark.parametrize(
    "line, expected",
    [
        ("Soil raw: 812", {"soil_raw": 812.0}),
        ("Temperature: 29.4 C", {"temperature_c": 29.4}),
        ("Humidity: 61.0 %", {"humidity_pct": 61.0}),
        ("soil raw:812", {"soil_raw": 812.0}),
        ("SOIL_RAW = 455", {"soil_raw": 455.0}),
        ("Soil moisture raw: 455", {"soil_raw": 455.0}),
        ("Soil: 455", {"soil_raw": 455.0}),
        ("Temp: 30 C", {"temperature_c": 30.0}),
        ("temperature=29.40", {"temperature_c": 29.4}),
        ("Hum: 55%", {"humidity_pct": 55.0}),
        ("  Humidity :  61.5 %  \r\n", {"humidity_pct": 61.5}),
        ("Temperature: -2.0 C", {"temperature_c": -2.0}),
    ],
)
def test_parse_single_field(line, expected):
    assert parse_line(line) == pytest.approx(expected)


def test_parse_multiple_fields_on_one_line():
    got = parse_line("Soil raw: 512, Temperature: 28.1 C, Humidity: 70 %")
    assert got == pytest.approx({"soil_raw": 512, "temperature_c": 28.1, "humidity_pct": 70})


def test_parse_nan_from_failed_dht():
    got = parse_line("Temperature: nan C")
    assert math.isnan(got["temperature_c"])


@pytest.mark.parametrize("line", ["", "# sensor_logger started", "garbage ###", "Soil raw: abc",
                                  "Pump ON", "Temperature:"])
def test_parse_malformed_returns_nothing(line):
    assert parse_line(line) == {}


def feed_all(lines):
    asm = ReadingAssembler()
    out = [r for r in (asm.feed(l) for l in lines) if r is not None]
    return asm, out


def test_assembler_standard_block():
    asm, out = feed_all(["Soil raw: 812", "Temperature: 29.4 C", "Humidity: 61.0 %", "---"])
    assert out == [pytest.approx({"soil_raw": 812, "temperature_c": 29.4, "humidity_pct": 61.0})]
    assert asm.complete_count == 1 and asm.dropped_count == 0


def test_assembler_field_order_does_not_matter():
    _, out = feed_all(["Humidity: 60 %", "Soil raw: 500", "Temp: 25 C"])
    assert len(out) == 1 and out[0]["soil_raw"] == 500


def test_assembler_drops_incomplete_block_and_recovers():
    lines = [
        "Soil raw: 800", "Temperature: 29 C", "---",         # humidity missing -> dropped
        "Soil raw: 810", "Temperature: 29.1 C", "Humidity: 60 %", "---",
    ]
    asm, out = feed_all(lines)
    assert len(out) == 1 and out[0]["soil_raw"] == 810
    assert asm.dropped_count == 1


def test_assembler_repeated_field_starts_new_reading():
    lines = ["Soil raw: 800", "Soil raw: 805", "Temperature: 29 C", "Humidity: 60 %"]
    asm, out = feed_all(lines)
    assert len(out) == 1 and out[0]["soil_raw"] == 805
    assert asm.dropped_count == 1


def test_assembler_discards_nan_reading():
    asm, out = feed_all(["Soil raw: 800", "Temperature: nan C", "Humidity: nan %"])
    assert out == [] and asm.dropped_count == 1


def test_assembler_counts_malformed_lines_without_crashing():
    asm, out = feed_all(["\x00\xff junk", "# comment", "Soil raw: 1", "Temperature: 2", "Humidity: 3"])
    assert len(out) == 1
    assert asm.malformed_lines == 2


def test_make_row_formats_values():
    row = make_row({"soil_raw": 812.4, "temperature_c": 29.4, "humidity_pct": 61.0},
                   plant_id="plant_1", condition="dry", label="WATER", session_id="s1")
    assert row["soil_raw"] == 812
    assert row["irrigation_label"] == "WATER" and row["label_source"] == "manual"
    unlabeled = make_row({"soil_raw": 1, "temperature_c": 2, "humidity_pct": 3},
                         plant_id="p", condition="", label="", session_id="s1")
    assert unlabeled["label_source"] == ""
