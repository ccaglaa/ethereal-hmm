"""Generate a melody with chords

Three small models, each trained on the training songs
  rhythm   P(gap to next onset | beat phase of this onset, previous gap)      gaps in 16ths, 1..16
  chords   Markov chain over the chord at each beat (chords change on beats, like in the songs)
  notes    P(pitch class | chord, previous pitch class) = emission table B of the ar1 chord HMM (M2)

Together this is an HMM whose hidden chord moves once per beat and emits a note at each onset
Sampling order: rhythm -> onset times; chords -> one chord per beat; notes -> pitch at each onset

generate_form adds structure on top: phrases that come back (AABA), varied, with a final cadence
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hmmusic.decode import nearest_pitch
from hmmusic.frames import N_CHORDS, NO_CHORD, SUBDIVISIONS
from hmmusic.hmm import HMM
from hmmusic.markov import MarkovChain
from hmmusic.midi import Note

HOME_CHORDS = (0, 19)  # C major, A minor
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
        """Legato: each note lasts until the next onset, at most max_beats beats

        The last note is held to the end of the piece (a final note cut short sounds unfinished)
        """
        frame = 60.0 / bpm / SUBDIVISIONS
        ends = np.minimum(np.r_[self.onsets[1:], self.n_frames], self.onsets + max_beats * SUBDIVISIONS)
        if len(ends):
            ends[-1] = self.n_frames
        return [Note(s * frame, e * frame, int(p), velocity) for s, e, p in zip(self.onsets, ends, self.pitches)]

    def chord_segments(self, bpm: float) -> list[tuple[float, float, int]]:
        beat = 60.0 / bpm
        return [(i * beat, (i + 1) * beat, int(c)) for i, c in enumerate(self.beat_chords)]


def sample_beat_chords(chain: MarkovChain, n_beats: int, rng: np.random.Generator,
                       context: list[int] | None = None) -> np.ndarray:
    """Chord per beat from the chain, never the no-chord symbol (resampled if drawn)

    context: chords that come just before, so a new section continues the previous one
    """
    probs = chain.probs()
    ctx = _chain_context(chain, context)
    out = np.empty(n_beats, dtype=int)
    for t in range(n_beats):
        p = probs[tuple(ctx)].copy()
        p[NO_CHORD] = 0.0
        out[t] = rng.choice(N_CHORDS, p=p / p.sum())
        if chain.order:
            ctx = ctx[1:] + [int(out[t])]
    return out


def _chain_context(chain: MarkovChain, previous: list[int] | None) -> list[int]:
    """Last chain.order chords, padded on the left with START when there are fewer"""
    previous = [int(c) for c in (previous or [])][-chain.order :] if chain.order else []
    return [chain.start] * (chain.order - len(previous)) + previous


def sample_notes(emission: np.ndarray, chords: np.ndarray, rng: np.random.Generator,
                 previous: int | None = None) -> np.ndarray:
    """Pitch class at each onset from B[chord, previous pitch class], B of shape (K, V + 1, V)

    previous: pitch class just before the first onset (None = start of a melody)
    """
    V = emission.shape[-1]
    out = np.empty(len(chords), dtype=int)
    prev = V if previous is None else int(previous)  # row V = "no previous note"
    for t, c in enumerate(chords):
        out[t] = rng.choice(V, p=emission[c, prev])
        prev = out[t]
    return out


def _pitches(pcs: np.ndarray, start: int, low: int, high: int) -> np.ndarray:
    out, previous = [], start
    for pc in pcs:
        previous = nearest_pitch(int(pc), previous, low, high)
        out.append(previous)
    return np.array(out, dtype=int)


def generate(rhythm: RhythmModel, chord_chain: MarkovChain, note_hmm: HMM, n_bars: int,
             rng: np.random.Generator, low: int = 67, high: int = 86, start_pitch: int = 76) -> Piece:
    """Sample a piece of n_bars bars of 4/4, no form; lead range G4 to D6 by default (high and airy)"""
    if note_hmm.emission_order != 1:
        raise ValueError("the note model must be an autoregressive (ar1) HMM")
    n_beats = 4 * n_bars
    n_frames = n_beats * SUBDIVISIONS
    onsets = rhythm.sample_onsets(n_frames, rng)
    beat_chords = sample_beat_chords(chord_chain, n_beats, rng)
    pcs = sample_notes(note_hmm.B, beat_chords[onsets // SUBDIVISIONS], rng)
    return Piece(onsets, pcs, _pitches(pcs, start_pitch, low, high), beat_chords, n_frames)


# ---- form: phrases that come back ----

def loop_score(chain: MarkovChain, chords: np.ndarray) -> float:
    """Log probability that the chain goes from the end of this progression back to its first chord

    High = the progression can repeat naturally (its last chords lead to its first)
    """
    ctx = _chain_context(chain, list(chords))
    return float(np.log(chain.probs()[tuple(ctx)][chords[0]]))


def chord_changes(chords: np.ndarray) -> int:
    """Number of beats where the chord differs from the previous beat"""
    return int((chords[1:] != chords[:-1]).sum())


def vary(piece: Piece, note_hmm: HMM, keep_beats: int, rng: np.random.Generator,
         low: int = 67, high: int = 86, start_pitch: int = 76) -> Piece:
    """Same rhythm and chords; notes before keep_beats unchanged, the rest sampled again

    Like a musical answer: the phrase starts the same way and ends differently
    """
    keep = piece.onsets < keep_beats * SUBDIVISIONS
    pcs = piece.pitch_classes.copy()
    chords = piece.chord_at_onsets()
    previous = int(pcs[keep][-1]) if keep.any() else None
    pcs[~keep] = sample_notes(note_hmm.B, chords[~keep], rng, previous)
    return Piece(piece.onsets.copy(), pcs, _pitches(pcs, start_pitch, low, high),
                 piece.beat_chords.copy(), piece.n_frames)


def dominant_of(home: int) -> int:
    """Major chord a fifth above the home chord's root: G for C, E (major) for A minor

    The dominant leads strongly back home; in minor keys pop and classical music use the major
    dominant (E, not Em, before Am) because its third (G#) pulls up to the root A
    """
    return ((home // 2 + 7) % 12) * 2


def cadence(piece: Piece, home: int, note_hmm: HMM, rng: np.random.Generator,
            low: int = 67, high: int = 86) -> Piece:
    """A full close: dominant on the last 2 beats of the second to last bar, home chord on the last bar

    Notes under the dominant are sampled again from that chord; the last of them is the scale step above
    the root (D before C, B before A), a chord tone of the dominant that falls by a step to the root;
    the root lands on the downbeat of the last bar and is the only note there, held to the end
    """
    chords = piece.beat_chords.copy()
    dominant = dominant_of(home)
    chords[-6:-4] = dominant
    chords[-4:] = home
    root = home // 2
    step = (root + 2) % 12
    last_bar = piece.n_frames - 4 * SUBDIVISIONS
    approach = last_bar - 2 * SUBDIVISIONS  # start of the dominant

    keep = piece.onsets < last_bar
    onsets, pcs = piece.onsets[keep].copy(), piece.pitch_classes[keep].copy()
    under_dominant = onsets >= approach
    if under_dominant.any():
        before = pcs[~under_dominant]
        previous = int(before[-1]) if len(before) else None
        pcs[under_dominant] = sample_notes(note_hmm.B, np.full(under_dominant.sum(), dominant), rng, previous)
        pcs[-1] = step
    else:
        onsets, pcs = np.r_[onsets, approach + SUBDIVISIONS], np.r_[pcs, step]  # on the last beat
    onsets, pcs = np.r_[onsets, last_bar], np.r_[pcs, root]
    start = int(piece.pitches[0]) if len(piece.pitches) else 76
    return Piece(onsets, pcs, _pitches(pcs, start, low, high), chords, piece.n_frames)


def concat(pieces: list[Piece]) -> Piece:
    onsets, pcs, pitches, chords, offset = [], [], [], [], 0
    for p in pieces:
        onsets.append(p.onsets + offset)
        pcs.append(p.pitch_classes)
        pitches.append(p.pitches)
        chords.append(p.beat_chords)
        offset += p.n_frames
    return Piece(np.concatenate(onsets), np.concatenate(pcs), np.concatenate(pitches), np.concatenate(chords), offset)


def generate_form(rhythm: RhythmModel, chord_chain: MarkovChain, note_hmm: HMM, rng: np.random.Generator,
                  form: str = "AABA", phrase_bars: int = 4, candidates: int = 50, min_changes: int = 3,
                  low: int = 67, high: int = 86, start_pitch: int = 76) -> tuple[Piece, list[str]]:
    """A piece made of phrases: A = the theme, B = contrast, repeated A = varied, last A = cadence

    A: its chords are the most loopable of `candidates` sampled progressions, starting at home if possible
    Repeats of A keep the first half of the phrase and answer differently in the second half
    Returns the piece and the label of each phrase (A, A', B, A'' ...)
    """
    if note_hmm.emission_order != 1:
        raise ValueError("the note model must be an autoregressive (ar1) HMM")
    n_beats = 4 * phrase_bars
    n_frames = n_beats * SUBDIVISIONS

    def new_phrase(context, previous_pc, start):
        onsets = rhythm.sample_onsets(n_frames, rng)
        chords = sample_beat_chords(chord_chain, n_beats, rng, context)
        pcs = sample_notes(note_hmm.B, chords[onsets // SUBDIVISIONS], rng, previous_pc)
        return Piece(onsets, pcs, _pitches(pcs, start, low, high), chords, n_frames)

    options = [sample_beat_chords(chord_chain, n_beats, rng) for _ in range(candidates)]
    # prefer progressions with at least min_changes chord changes (the most loopable ones are often
    # static: staying on a chord is the likeliest move), then starting on a home chord
    # (C or Am, as every song is in C major or A minor), then the most loopable
    a_chords = max(options, key=lambda c: (chord_changes(c) >= min_changes, int(c[0]) in HOME_CHORDS,
                                           loop_score(chord_chain, c)))
    onsets = rhythm.sample_onsets(n_frames, rng)
    pcs = sample_notes(note_hmm.B, a_chords[onsets // SUBDIVISIONS], rng)
    theme = Piece(onsets, pcs, _pitches(pcs, start_pitch, low, high), a_chords, n_frames)

    phrases, labels, seen_a = [], [], 0
    for i, part in enumerate(form):
        last = i == len(form) - 1
        if part == "A":
            p = theme if seen_a == 0 else vary(theme, note_hmm, n_beats // 2, rng, low, high, start_pitch)
            labels.append("A" + "'" * seen_a)
            seen_a += 1
        else:
            prev = phrases[-1]
            p = new_phrase(list(prev.beat_chords), int(prev.pitch_classes[-1]), int(prev.pitches[-1]))
            labels.append(part)
        if last:
            p = cadence(p, int(a_chords[0]), note_hmm, rng, low, high)
        phrases.append(p)
    return concat(phrases), labels


def ring_out(lead: list[Note], chords: list[tuple[float, float, int]], seconds: float = 4.0
             ) -> tuple[list[Note], list[tuple[float, float, int]]]:
    """Hold the final note and the final chord `seconds` longer, whatever the tempo

    At fast tempos the last bar lasts only ar. 2 s, too short for an ending to settle
    """
    if lead:
        last = lead[-1]
        lead = lead[:-1] + [Note(last.start, last.end + seconds, last.pitch, last.velocity)]
    if chords:
        start, end, symbol = chords[-1]
        chords = chords[:-1] + [(start, end + seconds, symbol)]
    return lead, chords
