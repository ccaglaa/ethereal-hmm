"""Minimal MIDI file reader and writer in pure Python

Only what this project needs: notes, track names and tempo
A MIDI file is a header chunk (MThd) followed by track chunks (MTrk)
Each track is a list of events, each preceded by a delta time in ticks
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, order=True)
class Note:
    """One note, times in seconds"""

    start: float
    end: float
    pitch: int  # MIDI pitch, 60 = middle C
    velocity: int


@dataclass
class Track:
    name: str
    notes: list[Note] = field(default_factory=list)


@dataclass
class MidiFile:
    ticks_per_beat: int
    tracks: list[Track]

    def track(self, name: str) -> Track:
        """Return the first track with this name"""
        for t in self.tracks:
            if t.name == name:
                return t
        raise KeyError(f"no track named {name!r}")


def _read_varlen(data: bytes, pos: int) -> tuple[int, int]:
    """Read a variable length quantity, return (value, new position)

    7 bits per byte, the high bit set means another byte follows
    """
    value = 0
    while True:
        byte = data[pos]
        pos += 1
        value = (value << 7) | (byte & 0x7F)
        if byte < 0x80:
            return value, pos


def _parse_track(data: bytes) -> tuple[str, list[tuple[int, int, int, int]], list[tuple[int, int]]]:
    """Parse one MTrk chunk body

    Returns the track name, raw notes as (start_tick, end_tick, pitch, velocity)
    and tempo changes as (tick, microseconds per beat)
    """
    pos, tick, status = 0, 0, 0
    name = ""
    notes: list[tuple[int, int, int, int]] = []
    tempos: list[tuple[int, int]] = []
    open_notes: dict[tuple[int, int], list[tuple[int, int]]] = {}  # (channel, pitch) -> [(start, vel)]

    while pos < len(data):
        delta, pos = _read_varlen(data, pos)
        tick += delta
        byte = data[pos]
        if byte >= 0x80:
            status = byte
            pos += 1
        # else running status: reuse the previous status byte

        if status == 0xFF:  # meta event
            kind = data[pos]
            length, pos = _read_varlen(data, pos + 1)
            body = data[pos : pos + length]
            pos += length
            if kind == 0x03:
                name = body.decode("latin-1")
            elif kind == 0x51:
                tempos.append((tick, int.from_bytes(body, "big")))
            elif kind == 0x2F:
                break
        elif status in (0xF0, 0xF7):  # sysex, skipped
            length, pos = _read_varlen(data, pos)
            pos += length
        else:
            kind, channel = status & 0xF0, status & 0x0F
            n_data = 1 if kind in (0xC0, 0xD0) else 2
            args = data[pos : pos + n_data]
            pos += n_data
            if kind == 0x90 and args[1] > 0:
                open_notes.setdefault((channel, args[0]), []).append((tick, args[1]))
            elif kind == 0x80 or (kind == 0x90 and args[1] == 0):
                stack = open_notes.get((channel, args[0]))
                if stack:
                    start, vel = stack.pop(0)  # first in first out
                    notes.append((start, tick, args[0], vel))
    return name, notes, tempos


def _tick_to_seconds(tempos: list[tuple[int, int]], ticks_per_beat: int):
    """Build a function mapping ticks to seconds from a tempo map"""
    tempos = sorted(tempos) or [(0, 500_000)]  # default 120 bpm
    if tempos[0][0] != 0:
        tempos.insert(0, (0, 500_000))
    # seconds elapsed at the start of each tempo segment
    starts = [0.0]
    for (t0, us), (t1, _) in zip(tempos, tempos[1:]):
        starts.append(starts[-1] + (t1 - t0) * us / 1e6 / ticks_per_beat)

    def convert(tick: int) -> float:
        i = len(tempos) - 1
        while tempos[i][0] > tick:
            i -= 1
        t0, us = tempos[i]
        return starts[i] + (tick - t0) * us / 1e6 / ticks_per_beat

    return convert


def read_midi(path: str | Path) -> MidiFile:
    data = Path(path).read_bytes()
    if data[:4] != b"MThd":
        raise ValueError(f"{path} is not a MIDI file")
    header_len = struct.unpack(">I", data[4:8])[0]
    _fmt, n_tracks, ticks_per_beat = struct.unpack(">HHH", data[8:14])
    if ticks_per_beat & 0x8000:
        raise ValueError("SMPTE time division not supported")

    pos = 8 + header_len
    parsed = []
    for _ in range(n_tracks):
        if data[pos : pos + 4] != b"MTrk":
            raise ValueError("expected MTrk chunk")
        length = struct.unpack(">I", data[pos + 4 : pos + 8])[0]
        parsed.append(_parse_track(data[pos + 8 : pos + 8 + length]))
        pos += 8 + length

    # in format 1 files the tempo map lives in any track, usually the first
    all_tempos = [t for _, _, tempos in parsed for t in tempos]
    to_sec = _tick_to_seconds(all_tempos, ticks_per_beat)

    tracks = []
    for name, raw, _ in parsed:
        notes = sorted(Note(to_sec(s), to_sec(e), p, v) for s, e, p, v in raw)
        tracks.append(Track(name, notes))
    return MidiFile(ticks_per_beat, tracks)


def _varlen(value: int) -> bytes:
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(out))


def write_midi(path: str | Path, tracks: list[Track], bpm: float = 120.0, ticks_per_beat: int = 480) -> None:
    """Write tracks as a format 1 MIDI file, one channel per track"""
    sec_to_tick = lambda s: round(s * bpm / 60 * ticks_per_beat)  # noqa: E731
    tempo = round(60e6 / bpm)
    chunks = [b"\x00\xff\x51\x03" + tempo.to_bytes(3, "big") + b"\x00\xff\x2f\x00"]
    for channel, track in enumerate(tracks):
        events = []  # (tick, order, bytes), note offs before note ons at the same tick
        for n in track.notes:
            events.append((sec_to_tick(n.start), 1, bytes([0x90 | channel, n.pitch, n.velocity])))
            events.append((sec_to_tick(n.end), 0, bytes([0x80 | channel, n.pitch, 0])))
        events.sort(key=lambda e: (e[0], e[1]))
        name = track.name.encode("latin-1")
        body = b"\x00\xff\x03" + _varlen(len(name)) + name
        last = 0
        for tick, _, msg in events:
            body += _varlen(tick - last) + msg
            last = tick
        chunks.append(body + b"\x00\xff\x2f\x00")
    out = b"MThd" + struct.pack(">IHHH", 6, 1, len(chunks), ticks_per_beat)
    for c in chunks:
        out += b"MTrk" + struct.pack(">I", len(c)) + c
    Path(path).write_bytes(out)
