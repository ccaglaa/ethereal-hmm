"""Generate a melody with chords

Three small models, each trained on the training songs
  rhythm   P(gap to next onset | beat phase of this onset, previous gap)      gaps in 16ths, 1..16
  chords   Markov chain over the chord at each beat (chords change on beats, like in the songs)
  notes    P(pitch class | chord, previous pitch class) = emission table B of the ar1 chord HMM (M2)

Together this is an HMM whose hidden chord moves once per beat and emits a note at each onset
Sampling order: rhythm -> onset times; chords -> one chord per beat; notes -> pitch at each onset
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hmmusic.decode import nearest_pitch
from hmmusic.frames import N_CHORDS, NO_CHORD, SUBDIVISIONS
from hmmusic.hmm import HMM
from hmmusic.markov import MarkovChain
from hmmusic.midi import Note

MAX_GAP = 16  # longest gap between onsets, in 16ths (1 bar of 4/4); longer gaps are clipped


class RhythmModel:
    """P(gap | phase, previous gap) with additive smoothing

    phase = position of the onset inside its beat (0..3), previous gap = 1..16 or 0 at the start
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.counts = np.zeros((SUBDIVISIONS, MAX_GAP + 1, MAX_GAP))

    def fit(self, phases: list[np.ndarray], gaps: list[np.ndarray]) -> RhythmModel:
        self.counts[:] = 0
        for ph, g in zip(phases, gaps):
            g = np.minimum(g[:-1], MAX_GAP)  # the last gap runs to the phrase end, not to a note
            previous = np.r_[0, g[:-1]]
            np.add.at(self.counts, (ph[: len(g)], previous, g - 1), 1)
        return self

    def probs(self) -> np.ndarray:
        smoothed = self.counts + self.alpha
        return smoothed / smoothed.sum(axis=-1, keepdims=True)

    def sample_onsets(self, n_frames: int, rng: np.random.Generator) -> np.ndarray:
        """Onset frames in [0, n_frames), the first one on the first beat"""
        probs = self.probs()
        onsets, t, previous = [], 0, 0
        while t < n_frames:
            onsets.append(t)
            gap = 1 + rng.choice(MAX_GAP, p=probs[t % SUBDIVISIONS, previous])
            t, previous = t + gap, gap
        return np.array(onsets)


@dataclass
class Piece:
    onsets: np.ndarray  # frame of each note onset
    pitch_classes: np.ndarray  # one per onset
    pitches: np.ndarray  # MIDI pitch per onset, octave chosen by nearest_pitch
    beat_chords: np.ndarray  # chord symbol per beat
    n_frames: int

    def chord_at_onsets(self) -> np.ndarray:
        return self.beat_chords[self.onsets // SUBDIVISIONS]

    def lead_notes(self, bpm: float, max_beats: float = 2.0, velocity: int = 85) -> list[Note]:
        """Legato: each note lasts until the next onset, at most max_beats beats"""
        frame = 60.0 / bpm / SUBDIVISIONS
        ends = np.minimum(np.r_[self.onsets[1:], self.n_frames], self.onsets + max_beats * SUBDIVISIONS)
        return [Note(s * frame, e * frame, int(p), velocity) for s, e, p in zip(self.onsets, ends, self.pitches)]

    def chord_segments(self, bpm: float) -> list[tuple[float, float, int]]:
        beat = 60.0 / bpm
        return [(i * beat, (i + 1) * beat, int(c)) for i, c in enumerate(self.beat_chords)]


def sample_beat_chords(chain: MarkovChain, n_beats: int, rng: np.random.Generator) -> np.ndarray:
    """Chord per beat from the chain, never the no-chord symbol (resampled if drawn)"""
    probs = chain.probs()
    context = [chain.start] * chain.order
    out = np.empty(n_beats, dtype=int)
    for t in range(n_beats):
        p = probs[tuple(context)].copy()
        p[NO_CHORD] = 0.0
        out[t] = rng.choice(N_CHORDS, p=p / p.sum())
        if chain.order:
            context = context[1:] + [int(out[t])]
    return out


def sample_notes(emission: np.ndarray, chords: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Pitch class at each onset from B[chord, previous pitch class], B of shape (K, V + 1, V)"""
    V = emission.shape[-1]
    out = np.empty(len(chords), dtype=int)
    previous = V  # "no previous note" row
    for t, c in enumerate(chords):
        out[t] = rng.choice(V, p=emission[c, previous])
        previous = out[t]
    return out


def generate(rhythm: RhythmModel, chord_chain: MarkovChain, note_hmm: HMM, n_bars: int,
             rng: np.random.Generator, low: int = 67, high: int = 86, start_pitch: int = 76) -> Piece:
    """Sample a piece of n_bars bars of 4/4; lead range G4 to D6 by default (high and airy)"""
    if note_hmm.emission_order != 1:
        raise ValueError("the note model must be an autoregressive (ar1) HMM")
    n_beats = 4 * n_bars
    n_frames = n_beats * SUBDIVISIONS
    onsets = rhythm.sample_onsets(n_frames, rng)
    beat_chords = sample_beat_chords(chord_chain, n_beats, rng)
    pcs = sample_notes(note_hmm.B, beat_chords[onsets // SUBDIVISIONS], rng)
    pitches, previous = [], start_pitch
    for pc in pcs:
        previous = nearest_pitch(int(pc), previous, low, high)
        pitches.append(previous)
    return Piece(onsets, pcs, np.array(pitches), beat_chords, n_frames)
