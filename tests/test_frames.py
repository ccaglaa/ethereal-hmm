import numpy as np
import pytest

from hmmusic.frames import (
    HOLD,
    NO_CHORD,
    REST,
    Frames,
    chord_name,
    chord_symbol,
    key_shift,
    note_sequences,
    phrases,
    split_songs,
    to_frames,
)
from hmmusic.midi import Note
from hmmusic.pop909 import Segment, Song


@pytest.mark.parametrize("label, shift", [("C:maj", 0), ("G:maj", 5), ("Gb:maj", 6), ("A:min", 0), ("E:min", 5), ("C:min", 9)])
def test_key_shift(label, shift):
    assert key_shift(label) == shift


@pytest.mark.parametrize(
    "label, shift, symbol",
    [("C:maj", 0, 0), ("A:min7/b7", 0, 19), ("G:sus4", 0, 14), ("B:hdim7", 0, 23), ("D:maj", 5, 14), ("N", 3, NO_CHORD)],
)
def test_chord_symbol(label, shift, symbol):
    assert chord_symbol(label, shift) == symbol


def test_chord_name():
    assert [chord_name(s) for s in (0, 19, NO_CHORD)] == ["C", "Am", "N"]


def _song(melody, chords, key="C:maj", n_beats=8, beat=0.5):
    grid = np.arange(n_beats) * beat
    return Song(
        song_id="x",
        melody=melody,
        chords=[Segment(*c) for c in chords],
        keys=[Segment(0.0, n_beats * beat, key)],
        grid=grid,
        on_beat=np.ones(n_beats, bool),
        downbeat=np.zeros(n_beats, bool),
    )


def test_to_frames_basic():
    # beat = 0.5 s, so one 16th frame = 0.125 s
    melody = [Note(0.0, 0.25, 60, 80), Note(0.5, 0.625, 64, 80)]  # C for 2 frames, rest 2, E for 1
    chords = [(0.0, 0.5, "C:maj"), (0.5, 1.0, "A:min")]
    f = to_frames(_song(melody, chords))
    assert f.obs.tolist() == [0, HOLD, REST, REST, 4]
    assert f.chord.tolist() == [0, 0, 0, 0, 19]


def test_to_frames_transposes_to_c():
    melody = [Note(0.0, 0.125, 67, 80)]  # G in G major is the tonic -> C
    f = to_frames(_song(melody, [(0.0, 4.0, "G:maj")], key="G:maj"))
    assert f.shift == 5
    assert f.obs.tolist() == [0]
    assert f.chord.tolist() == [0]


def test_overlap_truncates_earlier_note():
    melody = [Note(0.0, 0.375, 60, 80), Note(0.25, 0.5, 62, 80)]  # D starts while C still sounds
    f = to_frames(_song(melody, [(0.0, 4.0, "C:maj")]))
    assert f.obs.tolist() == [0, HOLD, 2, HOLD]


def test_same_note_repeated_is_two_onsets():
    melody = [Note(0.0, 0.125, 60, 80), Note(0.125, 0.25, 60, 80)]
    f = to_frames(_song(melody, [(0.0, 4.0, "C:maj")]))
    assert f.obs.tolist() == [0, 0]


def test_time_outside_chords_is_no_chord():
    melody = [Note(0.0, 0.125, 60, 80), Note(1.0, 1.125, 60, 80)]
    f = to_frames(_song(melody, [(0.5, 0.75, "F:maj")]))
    assert f.chord[0] == NO_CHORD and f.chord[4] == 10 and f.chord[-1] == NO_CHORD


def test_key_change_is_skipped():
    song = _song([Note(0.0, 0.125, 60, 80)], [(0.0, 4.0, "C:maj")])
    song.keys.append(Segment(4.0, 8.0, "D:maj"))
    assert to_frames(song) is None


def test_phrases_cut_at_long_silence():
    obs = np.array([REST, 0, HOLD] + [REST] * 4 + [2, REST, 4, REST, REST])
    chord = np.arange(len(obs))
    parts = phrases(Frames("x", obs, chord, 0), min_gap=4)
    assert [p[0].tolist() for p in parts] == [[0, HOLD], [2, REST, 4]]
    assert [p[1].tolist() for p in parts] == [[1, 2], [7, 8, 9]]


def test_phrases_keep_short_silence():
    obs = np.array([0, REST, REST, 2])
    assert len(phrases(Frames("x", obs, obs, 0), min_gap=4)) == 1


def test_split_songs_is_a_partition_and_reproducible():
    ids = [f"{i:03d}" for i in range(100)]
    train, val, test = split_songs(ids, seed=3)
    assert sorted(train + val + test) == ids
    assert (len(train), len(val), len(test)) == (80, 10, 10)
    assert split_songs(ids, seed=3) == (train, val, test)
    assert split_songs(ids, seed=4) != (train, val, test)


def test_note_sequences():
    obs = np.array([0, HOLD, REST, 4, 7, HOLD, HOLD] + [REST] * 5 + [9])
    chord = np.array([0, 0, 0, 0, 19, 19, 19, 19, 19, 19, 19, 19, 10])
    seqs = note_sequences(Frames("x", obs, chord, 0), min_gap=4)
    assert len(seqs) == 1  # the last phrase has a single note and is dropped
    pitches, chords, gaps = seqs[0]
    assert pitches.tolist() == [0, 4, 7]
    assert chords.tolist() == [0, 0, 19]
    assert gaps.tolist() == [3, 1, 3]
