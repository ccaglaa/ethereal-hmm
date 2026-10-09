"""Load POP909 songs: melody notes, chord labels, beat grid and key

Dataset layout, one folder per song (001 to 909)
  001.mid          tracks MELODY, BRIDGE, PIANO
  chord_midi.txt   start, end, label  (label like B:maj7/3 or N for no chord)
  beat_midi.txt    time, beat flag, downbeat flag  (one row per beat, melody onsets fall on quarter beats)
  key_audio.txt    start, end, key  (like Gb:maj)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hmmusic.midi import Note, read_midi

PITCH_CLASSES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def pitch_class(name: str) -> int:
    """Note name with optional sharps or flats -> 0..11, for example Gb -> 6"""
    pc = PITCH_CLASSES[name[0]]
    for accidental in name[1:]:
        pc += {"#": 1, "b": -1}[accidental]
    return pc % 12


@dataclass(frozen=True)
class Segment:
    """A labelled time span, used for chords and keys"""

    start: float
    end: float
    label: str


@dataclass
class Song:
    song_id: str
    melody: list[Note]
    chords: list[Segment]
    keys: list[Segment]
    grid: np.ndarray  # time in seconds of each beat
    on_beat: np.ndarray  # bool per grid row
    downbeat: np.ndarray  # bool per grid row


def _read_segments(path: Path) -> list[Segment]:
    segments = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 3:
            segments.append(Segment(float(parts[0]), float(parts[1]), parts[2]))
    return segments


def load_song(folder: str | Path) -> Song:
    folder = Path(folder)
    song_id = folder.name
    midi = read_midi(folder / f"{song_id}.mid")
    beats = np.loadtxt(folder / "beat_midi.txt", ndmin=2)
    return Song(
        song_id=song_id,
        melody=midi.track("MELODY").notes,
        chords=_read_segments(folder / "chord_midi.txt"),
        keys=_read_segments(folder / "key_audio.txt"),
        grid=beats[:, 0],
        on_beat=beats[:, 1] > 0.5,
        downbeat=beats[:, 2] > 0.5,
    )


def song_folders(root: str | Path) -> list[Path]:
    """All song folders under the POP909 directory, sorted by id"""
    return sorted(p for p in Path(root).iterdir() if p.is_dir() and p.name.isdigit())
