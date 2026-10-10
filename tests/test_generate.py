import numpy as np
import pytest

from hmmusic.frames import NO_CHORD
from hmmusic.generate import MAX_GAP, Piece, RhythmModel, generate, sample_beat_chords, sample_notes
from hmmusic.hmm import HMM
from hmmusic.markov import MarkovChain


def test_rhythm_counts():
    # onsets on phases 0, 2, 0 with gaps 2, 2, then the final gap (to the phrase end) is ignored
    model = RhythmModel(alpha=0.0).fit([np.array([0, 2, 0])], [np.array([2, 2, 7])])
    assert model.counts.sum() == 2
    assert model.counts[0, 0, 1] == 1  # phase 0, no previous gap, gap 2
    assert model.counts[2, 2, 1] == 1  # phase 2, previous gap 2, gap 2


def test_rhythm_clips_long_gaps():
    model = RhythmModel(alpha=0.0).fit([np.array([0, 0, 0])], [np.array([40, 4, 1])])
    assert model.counts[0, 0, MAX_GAP - 1] == 1
    assert model.counts[0, MAX_GAP, 3] == 1


def test_sample_onsets():
    model = RhythmModel(alpha=1.0)
    onsets = model.sample_onsets(64, np.random.default_rng(0))
    assert onsets[0] == 0
    assert onsets.max() < 64
    assert np.all(np.diff(onsets) >= 1) and np.all(np.diff(onsets) <= MAX_GAP)


def test_sample_onsets_follows_a_deterministic_rhythm():
    model = RhythmModel(alpha=1e-12)
    model.counts[:, :, 3] = 1.0  # always a gap of 4 = one note per beat
    assert model.sample_onsets(20, np.random.default_rng(0)).tolist() == [0, 4, 8, 12, 16]


def test_beat_chords_never_no_chord():
    chain = MarkovChain(25, 1, 1.0, "witten_bell").fit([np.array([NO_CHORD, NO_CHORD, 0, NO_CHORD])])
    chords = sample_beat_chords(chain, 200, np.random.default_rng(0))
    assert NO_CHORD not in chords


def test_sample_notes_uses_chord_and_previous_note():
    V = 3
    B = np.zeros((2, V + 1, V))
    B[0, :, 0] = 1.0  # chord 0 always plays note 0
    B[1, V, 1] = 1.0  # chord 1: first note 1, then note 2 after note 1, note 1 otherwise
    B[1, 1, 2] = 1.0
    B[1, 0, 1] = B[1, 2, 1] = 1.0
    notes = sample_notes(B, np.array([1, 1, 1, 0, 1]), np.random.default_rng(0))
    assert notes.tolist() == [1, 2, 1, 0, 1]


def _models(rng):
    rhythm = RhythmModel(alpha=1.0)
    chain = MarkovChain(25, 1, 1.0, "witten_bell").fit([rng.integers(0, 24, 50)])
    hmm = HMM.random(25, 12, rng, emission_order=1)
    return rhythm, chain, hmm


def test_generate_piece_is_consistent():
    rng = np.random.default_rng(0)
    piece = generate(*_models(rng), n_bars=4, rng=rng, low=60, high=80)
    assert piece.n_frames == 64 and len(piece.beat_chords) == 16
    assert len(piece.onsets) == len(piece.pitches) == len(piece.pitch_classes)
    assert np.all(piece.pitches % 12 == piece.pitch_classes)
    assert piece.pitches.min() >= 60 and piece.pitches.max() <= 80


def test_generate_needs_autoregressive_hmm():
    rng = np.random.default_rng(0)
    rhythm, chain, _ = _models(rng)
    with pytest.raises(ValueError):
        generate(rhythm, chain, HMM.random(25, 12, rng), n_bars=1, rng=rng)


def test_piece_notes_and_chords():
    piece = Piece(onsets=np.array([0, 2, 12]), pitch_classes=np.array([0, 4, 7]), pitches=np.array([72, 76, 79]),
                  beat_chords=np.array([0, 0, 19, 19]), n_frames=16)
    assert piece.chord_at_onsets().tolist() == [0, 0, 19]
    notes = piece.lead_notes(bpm=60, max_beats=2)  # one 16th = 0.25 s
    assert [(n.start, n.end, n.pitch) for n in notes] == [(0.0, 0.5, 72), (0.5, 2.5, 76), (3.0, 4.0, 79)]
    assert piece.chord_segments(bpm=60)[2] == (2.0, 3.0, 19)
