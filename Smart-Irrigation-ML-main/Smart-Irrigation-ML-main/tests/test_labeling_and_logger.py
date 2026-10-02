import sys
import types

import pandas as pd
import pytest

from src import data_collection
from src.config import ConfigError, load_config
from src.labeling import apply_threshold_labels, threshold_label


def test_threshold_label_boundary():
    assert threshold_label([10, 34.9, 35, 80], 35).tolist() == ["WATER", "WATER", "NO_WATER", "NO_WATER"]


def test_apply_threshold_fills_only_empty_labels_by_default():
    cfg = load_config(demo=True)   # dry=600, wet=280, threshold=35 (INVENTED demo values)
    df = pd.DataFrame({"soil_raw": [590, 300, 590], "irrigation_label": ["", "", "NO_WATER"],
                       "label_source": ["", "", "manual"]})
    out, n = apply_threshold_labels(df, cfg)
    assert n == 2
    assert out["irrigation_label"].tolist() == ["WATER", "NO_WATER", "NO_WATER"]   # manual label kept
    assert out["label_source"].tolist() == ["threshold_rule", "threshold_rule", "manual"]
    out2, n2 = apply_threshold_labels(df, cfg, overwrite=True)
    assert n2 == 3 and out2["irrigation_label"].tolist() == ["WATER", "NO_WATER", "WATER"]


def test_labeling_refuses_without_team_threshold():
    cfg = load_config(demo=False)  # real config ships with null calibration/threshold
    with pytest.raises(ConfigError):
        apply_threshold_labels(pd.DataFrame({"soil_raw": [500]}), cfg)


class FakeSerial:
    """Stands in for serial.Serial: yields scripted lines, then raises KeyboardInterrupt."""
    lines = []

    def __init__(self, *a, **kw):
        self._it = iter(self.lines)

    def __enter__(self): return self
    def __exit__(self, *a): return False
    def reset_input_buffer(self): pass

    def readline(self):
        try:
            return next(self._it)
        except StopIteration:
            raise KeyboardInterrupt


def install_fake_serial(monkeypatch, lines):
    FakeSerial.lines = lines
    mod = types.ModuleType("serial")
    mod.Serial = FakeSerial
    mod.SerialException = OSError
    monkeypatch.setitem(sys.modules, "serial", mod)
    monkeypatch.setattr(data_collection.time, "sleep", lambda s: None)


def test_logger_writes_csv_and_survives_garbage(monkeypatch, tmp_path):
    lines = [b"# sensor_logger started\r\n",
             b"Soil raw: 812\r\n", b"Temperature: 29.4 C\r\n", b"Humidity: 61.0 %\r\n", b"---\r\n",
             b"\xff\xfe garbage\r\n",
             b"Soil raw: 700\r\n", b"Temperature: nan C\r\n", b"Humidity: nan %\r\n", b"---\r\n",   # DHT failure
             b"Soil raw: 690\r\n", b"Temperature: 30.0 C\r\n", b"Humidity: 58.5 %\r\n", b"---\r\n"]
    install_fake_serial(monkeypatch, lines)
    out = tmp_path / "log.csv"
    rc = data_collection.main(["--port", "COMX", "--out", str(out), "--plant-id", "p1",
                               "--condition", "dry", "--label", "WATER", "--session-id", "s_test"])
    assert rc == 0
    df = pd.read_csv(out)
    assert list(df.columns)[:7] == ["timestamp", "soil_raw", "temperature_c", "humidity_pct",
                                    "plant_id", "soil_condition", "irrigation_label"]
    assert df["soil_raw"].tolist() == [812, 690]
    assert set(df["irrigation_label"]) == {"WATER"} and set(df["session_id"]) == {"s_test"}


def test_logger_appends_without_duplicating_header(monkeypatch, tmp_path):
    block = [b"Soil raw: 1\n", b"Temperature: 2\n", b"Humidity: 3\n"]
    out = tmp_path / "log.csv"
    for _ in range(2):
        install_fake_serial(monkeypatch, block)
        data_collection.main(["--port", "COMX", "--out", str(out)])
    assert len(pd.read_csv(out)) == 2
    assert out.read_text().count("timestamp,") == 1


def test_logger_reports_missing_port_cleanly(monkeypatch, tmp_path, capsys):
    mod = types.ModuleType("serial")
    mod.SerialException = OSError
    def boom(*a, **k): raise OSError("no such port")
    mod.Serial = boom
    monkeypatch.setitem(sys.modules, "serial", mod)
    assert data_collection.main(["--port", "NOPE", "--out", str(tmp_path / "x.csv")]) == 1
    assert "Could not open serial port" in capsys.readouterr().err
