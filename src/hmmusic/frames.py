"""Turn a song into frames: one melody symbol and one chord per 16th note

Melody symbols (the HMM observations)
  0..11  a new note starts, value = pitch class (0 = C, 1 = C#, ..., 11 = B)
  12     HOLD, the previous note keeps sounding
  13     REST, silence
Octaves are dropped here and chosen again when decoding (nearest to the previous note)

Chords (the HMM hidden states, used in M2)
  root * 2 + 0 for major-like chords, root * 2 + 1 for minor-like chords, 24 = no chord

Every song is transposed so that its key is C major or A minor (the relative minor of C)
Both use the white keys only, so pitch classes mean the same thing across songs
Songs that change key are skipped
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hmmusic.pop909 import Song, pitch_class

HOLD = 12
REST = 13
N_OBS = 14
NO_CHORD = 24
N_CHORDS = 25
SUBDIVISIONS = 4  # frames per beat, 4 = 16th notes

PITCH_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
OBS_NAMES = PITCH_NAMES + ["-", "."]  # "-" hold, "." rest


@dataclass
class Frames:
    song_id: str
    obs: np.ndarray  # int, one melody symbol per frame
    chord: np.ndarray  # int, one chord per frame
    shift: int  # semitones added to the original pitches to reach C
    first_beat: int = 0  # frame index of the first beat (> 0 when notes start before it)

    def beat_phase(self, frame: np.ndarray | int) -> np.ndarray | int:
        """Position of a frame inside its beat: 0 on the beat, 1, 2, 3 for the following 16ths"""
        return (np.asarray(frame) - self.first_beat) % SUBDIVISIONS


def key_shift(label: str) -> int:
    """Semitones to add so that this key becomes C major or A minor

    For a major key the tonic goes to C, for a minor key the tonic goes to A
    """
    tonic, mode = label.split(":")
    target = 0 if mode == "maj" else 9
    return (target - pitch_class(tonic)) % 12


def chord_symbol(label: str, shift: int) -> int:
    """POP909 chord label -> chord symbol after transposition

    Quality is reduced to major-like or minor-like, inversions (/3, /5) are ignored
    min, min7, dim, hdim7 -> minor-like; maj, 7, maj7, sus2, sus4, aug ... -> major-like
    """
    if label == "N":
        return NO_CHORD
    root, quality = label.split(":")
    quality = quality.split("/")[0]
    minor = quality.startswith(("min", "dim", "hdim"))
    return ((pitch_class(root) + shift) % 12) * 2 + int(minor)


def chord_name(symbol: int) -> str:
    if symbol == NO_CHORD:
        return "N"
    return PITCH_NAMES[symbol // 2] + ("m" if symbol % 2 else "")


def _time_to_frame(song: Song, times: np.ndarray) -> np.ndarray:
    """Seconds -> fractional frame index, following the beat grid

    Inside the grid we interpolate between beats, outside it we extend with the
    first and last beat lengths
    """
    beats = np.arange(len(song.grid), dtype=float)
    grid = song.grid
    first_step = grid[1] - grid[0]
    last_step = grid[-1] - grid[-2]
    pos = np.interp(times, grid, beats)
    pos = np.where(times < grid[0], (times - grid[0]) / first_step, pos)
    pos = np.where(times > grid[-1], beats[-1] + (times - grid[-1]) / last_step, pos)
    return pos * SUBDIVISIONS


def _frame_to_time(song: Song, frames: np.ndarray) -> np.ndarray:
    """Frame index -> seconds, the inverse of _time_to_frame"""
    grid = song.grid
    beat = frames / SUBDIVISIONS
    t = np.interp(beat, np.arange(len(grid)), grid)
    t = np.where(beat < 0, grid[0] + beat * (grid[1] - grid[0]), t)
    last = len(grid) - 1
    return np.where(beat > last, grid[-1] + (beat - last) * (grid[-1] - grid[-2]), t)


def to_frames(song: Song) -> Frames | None:
    """Quantize the melody and chords of a song to the 16th grid

    Returns None for songs that change key
    """
    if len(song.keys) != 1:
        return None
    shift = key_shift(song.keys[0].label)

    starts = np.round(_time_to_frame(song, np.array([n.start for n in song.melody]))).astype(int)
    ends = np.round(_time_to_frame(song, np.array([n.end for n in song.melody]))).astype(int)
    ends = np.maximum(ends, starts + 1)  # every note lasts at least one frame
    offset = min(0, starts.min())  # in case a note starts before the first beat
    starts, ends = starts - offset, ends - offset
    n_frames = ends.max()

    # which note sounds in each frame, -1 = none
    # notes are in start order, so a later note overwrites the tail of an overlapping earlier one
    owner = np.full(n_frames, -1)
    for i, (s, e) in enumerate(zip(starts, ends)):
        owner[s:e] = i

    obs = np.full(n_frames, REST)
    sounding = owner >= 0
    first = sounding & np.r_[True, owner[1:] != owner[:-1]]  # first frame of each note
    pitches = np.array([(n.pitch + shift) % 12 for n in song.melody])
    obs[sounding] = HOLD
    obs[first] = pitches[owner[first]]

    # chord at each frame: the chord segment that contains the frame time
    frame_times = _frame_to_time(song, np.arange(n_frames) + offset)
    chord_starts = np.array([c.start for c in song.chords])
    idx = np.clip(np.searchsorted(chord_starts, frame_times + 1e-6, side="right") - 1, 0, len(song.chords) - 1)
    chord = np.array([chord_symbol(song.chords[i].label, shift) for i in idx])
    chord[frame_times < chord_starts[0]] = NO_CHORD
    chord[frame_times >= song.chords[-1].end] = NO_CHORD

    return Frames(song.song_id, obs, chord, shift, first_beat=-offset)


def split_songs(ids: list[str], seed: int = 0, val: float = 0.1, test: float = 0.1) -> tuple[list[str], list[str], list[str]]:
    """Random split by song, so no song appears in two sets"""
    rng = np.random.default_rng(seed)
    ids = sorted(ids)
    order = rng.permutation(len(ids))
    n_test, n_val = round(test * len(ids)), round(val * len(ids))
    test_ids = [ids[i] for i in order[:n_test]]
    val_ids = [ids[i] for i in order[n_test : n_test + n_val]]
    train_ids = [ids[i] for i in order[n_test + n_val :]]
    return train_ids, val_ids, test_ids


def phrase_bounds(frames: Frames, min_gap: int = 32) -> list[tuple[int, int]]:
    """(first, last) frame index of each phrase, last included; see phrases"""
    sounding = np.flatnonzero(frames.obs != REST)
    if len(sounding) == 0:
        return []
    gaps = np.diff(sounding) - 1
    cut = np.flatnonzero(gaps >= min_gap)
    first = np.r_[sounding[0], sounding[cut + 1]]
    last = np.r_[sounding[cut], sounding[-1]]
    return list(zip(first.tolist(), last.tolist()))


def phrases(frames: Frames, min_gap: int = 32) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cut a song at silences of at least min_gap frames (32 = 2 bars of 4/4)

    Returns (obs, chord) pairs without leading or trailing rests
    Long silences are instrumental parts, not melody, so the model should not learn them
    """
    return [(frames.obs[a : b + 1], frames.chord[a : b + 1]) for a, b in phrase_bounds(frames, min_gap)]


def note_sequences(frames: Frames, min_gap: int = 32) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """One entry per note instead of per frame, for each phrase

    Returns (pitch classes, chord at each note onset, frames until the next onset) per phrase
    The last note's gap runs to the end of the phrase; phrases with fewer than 2 notes are dropped
    """
    out = []
    for obs, chord in phrases(frames, min_gap):
        onsets = np.flatnonzero(obs < 12)
        if len(onsets) < 2:
            continue
        gaps = np.diff(np.r_[onsets, len(obs)])
        out.append((obs[onsets], chord[onsets], gaps))
    return out


def onset_phases(frames: Frames, min_gap: int = 32) -> list[np.ndarray]:
    """Beat phase (0..3) of every note onset, phrase by phrase, aligned with note_sequences"""
    out = []
    for a, b in phrase_bounds(frames, min_gap):
        onsets = a + np.flatnonzero(frames.obs[a : b + 1] < 12)
        if len(onsets) >= 2:
            out.append(frames.beat_phase(onsets))
    return out


def beat_chords(frames: Frames) -> list[np.ndarray]:
    """Chord at each beat of the song, split where there is no chord (N), runs of 2+ beats kept"""
    per_beat = frames.chord[frames.first_beat :: SUBDIVISIONS]
    runs, current = [], []
    for c in per_beat:
        if c == NO_CHORD:
            if len(current) >= 2:
                runs.append(np.array(current))
            current = []
        else:
            current.append(int(c))
    if len(current) >= 2:
        runs.append(np.array(current))
    return runs
