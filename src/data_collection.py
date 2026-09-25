"""
Serial data logger: reads the Arduino's Serial Monitor-style output and saves
timestamped observations to CSV in data/raw/.

The Arduino (firmware/arduino/sensor_logger) prints one block per sample:

    Soil raw: 812
    Temperature: 29.4 C
    Humidity: 61.0 %
    ---

Small formatting differences are tolerated (case, "Temp"/"Hum", "=" instead of
":", missing units, several values on one line). Malformed lines are counted
and skipped, never crash the logger.

Usage:
    python -m src.data_collection --port COM3 --plant-id plant_1 --condition dry
    python -m src.data_collection --port COM3 --label WATER --max-samples 30
    python -m src.data_collection --list-ports

Stop with Ctrl+C; every row is flushed to disk as soon as it is complete.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import CSV_COLUMNS, LABELS, PROJECT_ROOT

_NUMBER = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)|nan)"
_SEP = r"\s*[:=]?\s*"

# key -> regex. Each regex captures one number (or "nan" from a failed DHT read).
FIELD_PATTERNS = {
    "soil_raw": re.compile(r"\b(?:soil(?:[ _]*moisture)?(?:[ _]*raw)?|moisture(?:[ _]*raw)?)" + _SEP + _NUMBER, re.I),
    "temperature_c": re.compile(r"\b(?:temperature|temp|t)\b" + _SEP + _NUMBER, re.I),
    "humidity_pct": re.compile(r"\b(?:humidity|hum|rh|h)\b" + _SEP + _NUMBER, re.I),
}
FIELDS = tuple(FIELD_PATTERNS)
BLOCK_SEPARATOR = re.compile(r"^\s*-{3,}\s*$")


def parse_line(line: str) -> dict[str, float]:
    """
    Extract any sensor fields present in one line of serial output.

    Returns a dict with zero or more of: soil_raw, temperature_c, humidity_pct.
    Values that read as "nan" (DHT failure) are returned as float('nan') so
    the caller can decide to discard the sample.
    """
    found: dict[str, float] = {}
    for key, pattern in FIELD_PATTERNS.items():
        m = pattern.search(line)
        if m:
            try:
                found[key] = float(m.group(1))
            except ValueError:
                continue
    return found


@dataclass
class ReadingAssembler:
    """
    Collects fields across lines until a full reading (all three sensors) is
    available, then emits it. A field that repeats before the reading is
    complete starts a new reading (the previous partial one is dropped as
    malformed).
    """

    current: dict[str, float] = field(default_factory=dict)
    complete_count: int = 0
    dropped_count: int = 0
    malformed_lines: int = 0

    def feed(self, line: str) -> Optional[dict[str, float]]:
        line = line.strip()
        if not line:
            return None
        if BLOCK_SEPARATOR.match(line):
            if self.current:
                self.dropped_count += 1
            self.current = {}
            return None

        fields = parse_line(line)
        if not fields:
            self.malformed_lines += 1
            return None

        if any(k in self.current for k in fields):
            self.dropped_count += 1
            self.current = {}
        self.current.update(fields)

        if all(k in self.current for k in FIELDS):
            reading = self.current
            self.current = {}
            if any(math.isnan(v) for v in reading.values()):
                self.dropped_count += 1
                return None
            self.complete_count += 1
            return reading
        return None


def make_row(reading: dict[str, float], *, plant_id: str, condition: str, label: str,
             session_id: str, timestamp: Optional[datetime] = None) -> dict[str, object]:
    ts = (timestamp or datetime.now()).isoformat(timespec="seconds")
    return {
        "timestamp": ts,
        "soil_raw": int(round(reading["soil_raw"])),
        "temperature_c": reading["temperature_c"],
        "humidity_pct": reading["humidity_pct"],
        "plant_id": plant_id,
        "soil_condition": condition,
        "irrigation_label": label,
        "session_id": session_id,
        "label_source": "manual" if label else "",
    }


def default_output_path(session_id: str) -> Path:
    return PROJECT_ROOT / "data" / "raw" / f"{session_id}.csv"


def run_logger(args) -> int:
    try:
        import serial  # pyserial, imported lazily so the parser is testable without it
    except ImportError:
        print("pyserial is not installed: pip install pyserial", file=sys.stderr)
        return 1

    session_id = args.session_id or datetime.now().strftime("session_%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else default_output_path(session_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_path.exists()

    assembler = ReadingAssembler()
    print(f"Opening {args.port} @ {args.baud} baud. Writing to {out_path}")
    print("Press Ctrl+C to stop.")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as exc:
        print(f"Could not open serial port {args.port}: {exc}", file=sys.stderr)
        return 1

    started = time.monotonic()
    with ser, open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        time.sleep(2)  # the Uno resets when the port opens
        ser.reset_input_buffer()
        try:
            while True:
                if args.max_samples and assembler.complete_count >= args.max_samples:
                    break
                if args.duration and time.monotonic() - started >= args.duration:
                    break
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace")
                if args.echo:
                    print(f"  < {line.rstrip()}")
                reading = assembler.feed(line)
                if reading is None:
                    continue
                row = make_row(reading, plant_id=args.plant_id, condition=args.condition,
                               label=args.label, session_id=session_id)
                writer.writerow(row)
                f.flush()
                print(f"[{assembler.complete_count:4d}] {row['timestamp']} soil_raw={row['soil_raw']} "
                      f"T={row['temperature_c']}C H={row['humidity_pct']}%")
        except KeyboardInterrupt:
            print("\nStopped by user.")

    print(f"Saved {assembler.complete_count} readings to {out_path} "
          f"(dropped {assembler.dropped_count} incomplete/failed readings, "
          f"skipped {assembler.malformed_lines} unrecognised lines).")
    return 0


def list_ports() -> int:
    try:
        from serial.tools import list_ports as lp
    except ImportError:
        print("pyserial is not installed: pip install pyserial", file=sys.stderr)
        return 1
    ports = list(lp.comports())
    if not ports:
        print("No serial ports found.")
    for p in ports:
        print(f"{p.device}\t{p.description}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Log Arduino sensor readings to CSV (offline).")
    p.add_argument("--port", help="Serial port, e.g. COM3 or /dev/ttyACM0")
    p.add_argument("--baud", type=int, default=9600, help="Baud rate (must match the sketch)")
    p.add_argument("--plant-id", default="plant_1")
    p.add_argument("--condition", default="", help="Observed soil condition: dry / moist / wet")
    p.add_argument("--label", default="", choices=("",) + LABELS,
                   help="Irrigation label decided by the team for this whole session. "
                        "Leave empty to label afterwards.")
    p.add_argument("--session-id", default=None, help="Defaults to session_<date>_<time>")
    p.add_argument("--out", default=None, help="Output CSV (default data/raw/<session_id>.csv)")
    p.add_argument("--max-samples", type=int, default=0, help="Stop after N readings (0 = no limit)")
    p.add_argument("--duration", type=float, default=0, help="Stop after N seconds (0 = no limit)")
    p.add_argument("--echo", action="store_true", help="Print every raw serial line")
    p.add_argument("--list-ports", action="store_true", help="List serial ports and exit")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_ports:
        return list_ports()
    if not args.port:
        print("--port is required (use --list-ports to find it).", file=sys.stderr)
        return 2
    return run_logger(args)


if __name__ == "__main__":
    sys.exit(main())
