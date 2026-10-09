"""Frames -> MIDI notes

Pitch classes become MIDI pitches by picking the octave closest to the previous note,
which keeps the melody smooth, inside a fixed range
"""

from __future__ import annotations

import numpy as np

from hmmusic.frames import HOLD, REST, SUBDIVISIONS
from hmmusic.midi import Note


def nearest_pitch(pc: int, previous: int, low: int = 60, high: int = 84) -> int:
    """MIDI pitch with pitch class pc closest to previous, kept inside [low, high]"""
    candidates = [p for p in range(low, high + 1) if p % 12 == pc]
    return min(candidates, key=lambda p: (abs(p - previous), p))


def obs_to_notes(obs: np.ndarray, bpm: float = 80.0, start_pitch: int = 72, velocity: int = 80) -> list[Note]:
    frame = 60.0 / bpm / SUBDIVISIONS
    notes: list[Note] = []
    previous = start_pitch
    current: tuple[int, int] | None = None  # (start frame, pitch) of the note being held

    def close(end: int) -> None:
        if current is not None:
            notes.append(Note(current[0] * frame, end * frame, current[1], velocity))

    for t, x in enumerate(obs):
        if x == HOLD:
            continue  # a hold after a rest just extends the silence
        close(t)
        current = None
        if x != REST:
            previous = nearest_pitch(int(x), previous)
            current = (t, previous)
    close(len(obs))
    return notes
