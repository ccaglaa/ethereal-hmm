"""Do generated melodies look like real ones? Simple statistics, generated vs held-out test songs

Compared
  test songs                  real melodies with their annotated chords
  chord HMM (ours)            notes from B[chord, previous note], chords per beat from the chord chain
  chord HMM + AABA form       the same models arranged in phrases that come back (generate_form)
  Markov order 2, no chords   same rhythm and chords, but notes ignore the chord (control)

Statistics per note onset
  chord tone      the pitch class is in the triad of the chord at that onset
  repeat / step / leap   size of the move from the previous note: 0, 1 to 2, 3 or more semitones
  notes per bar
  repeated    share of 8-note pitch class patterns already heard earlier in the same piece
              (real songs: their first 90 notes, about the length of a 16 bar generated piece)

Usage: python scripts/evaluate_generation.py [--pieces=50] [--bars=16] [--root=../POP909-Dataset/POP909]
"""

import json
import sys
from pathlib import Path

import numpy as np

from hmmusic.frames import N_CHORDS, beat_chords, note_sequences, onset_phases, phrases, split_songs, to_frames
from hmmusic.generate import RhythmModel, generate, generate_form
from hmmusic.hmm import HMM
from hmmusic.markov import MarkovChain
from hmmusic.pop909 import load_song, song_folders
from hmmusic.synth import chord_pitch_classes

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
root = Path(opts.get("root", "../POP909-Dataset/POP909"))
n_pieces = int(opts.get("pieces", 50))
n_bars = int(opts.get("bars", 16))

frames = {f.song_id: f for f in (to_frames(load_song(d)) for d in song_folders(root)) if f is not None}
train_ids, _, test_ids = split_songs(list(frames), seed=0)
phases = [p for i in train_ids for p in onset_phases(frames[i])]
train_notes = [s for i in train_ids for s in note_sequences(frames[i])]
rhythm = RhythmModel().fit(phases, [s[2] for s in train_notes])
chord_chain = MarkovChain(N_CHORDS, 4, 1.0, "witten_bell").fit([c for i in train_ids for c in beat_chords(frames[i])])
note_hmm = HMM.fit_supervised([s[0] for s in train_notes], [s[1] for s in train_notes], N_CHORDS, 12, 0.1, 1)
markov2 = MarkovChain(12, 2, 1.0, "witten_bell").fit([s[0] for s in train_notes])


def repeated(pieces, window=8):
    """Share of windows of `window` pitch classes that already occurred earlier in the same piece"""
    hits, total = 0, 0
    for pcs in pieces:
        seen = set()
        for i in range(len(pcs) - window + 1):
            w = tuple(int(x) for x in pcs[i : i + window])
            hits += w in seen
            total += 1
            seen.add(w)
    return hits / max(total, 1)


def stats(seqs, notes_per_bar=None):
    """seqs: list of (pitch classes, chords at onsets)"""
    tone, moves = [], []
    for pcs, chords in seqs:
        ok = chords < 24
        tone += [pc in chord_pitch_classes(c) for pc, c in zip(pcs[ok], chords[ok])]
        step = np.abs((np.diff(pcs) + 6) % 12 - 6)
        moves.append(step)
    m = np.concatenate(moves)
    out = {"chord tone": np.mean(tone), "repeat": np.mean(m == 0), "step 1-2": np.mean((m >= 1) & (m <= 2)),
           "leap 3+": np.mean(m >= 3)}
    if notes_per_bar is not None:
        out["notes/bar"] = notes_per_bar
    return out


rows = {}
test = [(s[0], s[1]) for i in test_ids for s in note_sequences(frames[i])]
# bars inside phrases (silences of 2+ bars excluded), 16 frames per bar of 4/4
test_bars = sum(len(obs) for i in test_ids for obs, _ in phrases(frames[i])) / 16
rows["test songs"] = stats(test, sum(len(s[0]) for s in test) / test_bars)
song_melodies = [np.concatenate([s[0] for s in note_sequences(frames[i])])[:90] for i in test_ids]
rows["test songs"]["repeated"] = repeated(song_melodies)

rng = np.random.default_rng(0)
ours, control, n_notes = [], [], 0
for _ in range(n_pieces):
    piece = generate(rhythm, chord_chain, note_hmm, n_bars, rng)
    chords = piece.chord_at_onsets()
    ours.append((piece.pitch_classes, chords))
    control.append((markov2.sample(len(chords), rng), chords))
    n_notes += len(chords)
rows["chord HMM (ours)"] = stats(ours, n_notes / (n_pieces * n_bars))
rows["chord HMM (ours)"]["repeated"] = repeated([p for p, _ in ours])
rows["Markov 2, no chords"] = stats(control, n_notes / (n_pieces * n_bars))
rows["Markov 2, no chords"]["repeated"] = repeated([p for p, _ in control])

formed, n_formed = [], 0
for _ in range(n_pieces):
    piece, _ = generate_form(rhythm, chord_chain, note_hmm, rng)  # 4 phrases of 4 bars = 16 bars
    formed.append((piece.pitch_classes, piece.chord_at_onsets()))
    n_formed += len(piece.onsets)
rows["chord HMM + AABA form"] = stats(formed, n_formed / (n_pieces * 16))
rows["chord HMM + AABA form"]["repeated"] = repeated([p for p, _ in formed])

keys = ["chord tone", "repeat", "step 1-2", "leap 3+", "notes/bar", "repeated"]
Path("outputs").mkdir(exist_ok=True)
with open("outputs/generation_eval.json", "w") as f:  # read by scripts/make_readme_figures.py
    json.dump({name: {k: float(v) for k, v in r.items()} for name, r in rows.items()}, f, indent=2)
print(f"{n_pieces} generated pieces of {n_bars} bars vs {len(test)} test phrases\n")
print(f"{'':<24}" + "".join(f"{k:>12}" for k in keys))
for name, r in rows.items():
    print(f"{name:<24}" + "".join(f"{(f'{100 * r[k]:.1f}%' if k != 'notes/bar' else f'{r[k]:.1f}') if k in r else '':>12}" for k in keys))
