from pathlib import Path

import pytest

from hmmusic.midi import Note, Track, _read_varlen, _varlen, read_midi, write_midi


@pytest.mark.parametrize("value", [0, 1, 127, 128, 480, 16383, 16384, 2**21, 2**28 - 1])
def test_varlen_round_trip(value):
    encoded = _varlen(value)
    decoded, pos = _read_varlen(encoded, 0)
    assert decoded == value
    assert pos == len(encoded)


def test_varlen_known_bytes():
    # examples from the MIDI specification
    assert _varlen(0x40) == bytes([0x40])
    assert _varlen(0x80) == bytes([0x81, 0x00])
    assert _varlen(0x3FFF) == bytes([0xFF, 0x7F])


def test_write_then_read(tmp_path: Path):
    melody = Track("MELODY", [Note(0.0, 0.5, 60, 90), Note(0.5, 1.0, 64, 80), Note(1.0, 2.0, 67, 70)])
    pad = Track("PAD", [Note(0.0, 2.0, 48, 60), Note(0.0, 2.0, 55, 60)])
    path = tmp_path / "x.mid"
    write_midi(path, [melody, pad], bpm=90)
    midi = read_midi(path)
    assert [t.name for t in midi.tracks] == ["", "MELODY", "PAD"]  # first is the tempo track
    for original in (melody, pad):
        got = midi.track(original.name).notes
        assert [(n.pitch, n.velocity) for n in got] == [(n.pitch, n.velocity) for n in sorted(original.notes)]
        for a, b in zip(got, sorted(original.notes)):
            assert a.start == pytest.approx(b.start, abs=1e-3)
            assert a.end == pytest.approx(b.end, abs=1e-3)


def _raw_midi(track_body: bytes, ticks_per_beat: int = 480) -> bytes:
    header = b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big") + (1).to_bytes(2, "big")
    header += ticks_per_beat.to_bytes(2, "big")
    return header + b"MTrk" + len(track_body).to_bytes(4, "big") + track_body


def test_running_status_and_velocity_zero_note_off(tmp_path: Path):
    # note on 60, then (running status) note on 60 with velocity 0 one beat later = note off
    body = bytes([0x00, 0x90, 60, 100, 0x83, 0x60, 60, 0x00, 0x00, 0xFF, 0x2F, 0x00])
    path = tmp_path / "r.mid"
    path.write_bytes(_raw_midi(body))
    notes = read_midi(path).tracks[0].notes
    assert notes == [Note(0.0, 0.5, 60, 100)]  # 480 ticks at default 120 bpm = 0.5 s


def test_tempo_change(tmp_path: Path):
    # tempo 60 bpm (1e6 us per beat) then a note lasting 2 beats -> 2 seconds
    body = bytes([0x00, 0xFF, 0x51, 0x03, 0x0F, 0x42, 0x40])
    body += bytes([0x00, 0x90, 64, 90, 0x87, 0x40, 0x80, 64, 0x00, 0x00, 0xFF, 0x2F, 0x00])
    path = tmp_path / "t.mid"
    path.write_bytes(_raw_midi(body))
    assert read_midi(path).tracks[0].notes == [Note(0.0, 2.0, 64, 90)]


def test_rejects_non_midi(tmp_path: Path):
    path = tmp_path / "bad.mid"
    path.write_bytes(b"RIFF....")
    with pytest.raises(ValueError):
        read_midi(path)
