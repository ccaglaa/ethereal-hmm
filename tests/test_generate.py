import numpy as np
import pytest

from hmmusic.frames import NO_CHORD
from hmmusic.generate import (
    MAX_GAP,
    Piece,
    RhythmModel,
    cadence,
    chord_changes,
    concat,
    generate,
    generate_form,
    loop_score,
    sample_beat_chords,
    sample_notes,
    vary,
)
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


def _cycle_chain(order=2):
    # deterministic cycle of chords 0 -> 2 -> 4 -> 0 ...
    return MarkovChain(25, order, 1e-9, "witten_bell").fit([np.array([0, 2, 4] * 40)])


def test_beat_chords_continue_from_context():
    chords = sample_beat_chords(_cycle_chain(), 4, np.random.default_rng(0), context=[0, 2])
    assert chords.tolist() == [4, 0, 2, 4]


def test_sample_notes_with_previous():
    V = 2
    B = np.zeros((1, V + 1, V))
    B[0, 0, 1] = B[0, 1, 0] = B[0, V, 0] = 1.0  # alternate, start with 0
    assert sample_notes(B, np.zeros(3, int), np.random.default_rng(0)).tolist() == [0, 1, 0]
    assert sample_notes(B, np.zeros(3, int), np.random.default_rng(0), previous=0).tolist() == [1, 0, 1]


def test_loop_score_prefers_progressions_that_return():
    chain = _cycle_chain()
    assert loop_score(chain, np.array([0, 2, 4])) > loop_score(chain, np.array([0, 4, 2]))


def test_chord_changes():
    assert chord_changes(np.array([0, 0, 5, 5, 0, 19])) == 3


def _piece():
    return Piece(onsets=np.array([0, 4, 8, 12, 20, 28, 40, 60]), pitch_classes=np.array([0, 2, 4, 5, 7, 9, 11, 0]),
                 pitches=np.array([72, 74, 76, 77, 79, 81, 83, 84]), beat_chords=np.array([0] * 8 + [10] * 8),
                 n_frames=64)


def test_vary_keeps_the_start():
    rng = np.random.default_rng(0)
    hmm = HMM.random(25, 12, rng, emission_order=1)
    p = _piece()
    v = vary(p, hmm, keep_beats=4, rng=rng)
    assert v.onsets.tolist() == p.onsets.tolist()
    assert v.beat_chords.tolist() == p.beat_chords.tolist()
    assert v.pitch_classes[:4].tolist() == p.pitch_classes[:4].tolist()  # onsets before frame 16


def test_cadence():
    c = cadence(_piece(), home=19)  # A minor
    assert c.beat_chords[-4:].tolist() == [19] * 4
    assert c.pitch_classes[-1] == 9  # root of A minor
    assert c.onsets.max() < 64 - 2 * 4 + 1  # nothing in the second half of the last bar
    assert c.onsets[-1] >= 48  # the final note is in the last bar


def test_concat():
    p = concat([_piece(), _piece()])
    assert p.n_frames == 128
    assert p.onsets[8] == 64 and len(p.onsets) == 16
    assert len(p.beat_chords) == 32


def test_generate_form():
    rng = np.random.default_rng(1)
    rhythm, chain, hmm = _models(rng)
    piece, labels = generate_form(rhythm, chain, hmm, rng, form="AABA", phrase_bars=4)
    assert labels == ["A", "A'", "B", "A''"]
    assert piece.n_frames == 4 * 64 and len(piece.beat_chords) == 64
    a = piece.onsets < 32  # first half of the first phrase
    a2 = (piece.onsets >= 64) & (piece.onsets < 96)  # first half of the second phrase
    assert piece.pitch_classes[a].tolist() == piece.pitch_classes[a2].tolist()
    assert piece.beat_chords[:16].tolist() == piece.beat_chords[16:32].tolist()
    home = piece.beat_chords[0]
    assert piece.beat_chords[-4:].tolist() == [home] * 4
    assert piece.pitch_classes[-1] == home // 2
