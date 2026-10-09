from pathlib import Path

import numpy as np
import pytest

from hmmusic.midi import Note, Track, write_midi
from hmmusic.pop909 import load_song, pitch_class, song_folders


@pytest.mark.parametrize("name, pc", [("C", 0), ("C#", 1), ("Db", 1), ("Gb", 6), ("B", 11), ("Cb", 11), ("B#", 0)])
def test_pitch_class(name, pc):
    assert pitch_class(name) == pc


def _fake_song(root: Path, song_id: str = "042") -> Path:
    folder = root / song_id
    folder.mkdir()
    melody = Track("MELODY", [Note(0.0, 0.5, 72, 90), Note(0.5, 1.0, 74, 90)])
    piano = Track("PIANO", [Note(0.0, 1.0, 48, 60)])
    write_midi(folder / f"{song_id}.mid", [melody, Track("BRIDGE"), piano], bpm=120)
    (folder / "chord_midi.txt").write_text("0.0\t0.5\tN\n0.5\t1.0\tC:maj7/3\n")
    (folder / "beat_midi.txt").write_text("0.0 1.0 1.0\n0.5 0.0 0.0\n")
    (folder / "key_audio.txt").write_text("0.0\t1.0\tC:maj\n")
    return folder


def test_load_song(tmp_path: Path):
    song = load_song(_fake_song(tmp_path))
    assert song.song_id == "042"
    assert [n.pitch for n in song.melody] == [72, 74]
    assert [c.label for c in song.chords] == ["N", "C:maj7/3"]
    assert song.keys[0].label == "C:maj"
    np.testing.assert_allclose(song.grid, [0.0, 0.5])
    assert song.downbeat.tolist() == [True, False]


def test_song_folders_skips_non_song_entries(tmp_path: Path):
    _fake_song(tmp_path, "002")
    _fake_song(tmp_path, "001")
    (tmp_path / "index.xlsx").write_text("")
    (tmp_path / "notes").mkdir()
    assert [p.name for p in song_folders(tmp_path)] == ["001", "002"]
