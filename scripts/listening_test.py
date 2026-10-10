"""Blind listening test: real song excerpts vs our generator vs a chord-blind control

Step 1  python scripts/listening_test.py make
        writes outputs/listening/clip_01.wav ... clip_18.wav (shuffled) and ratings.csv to fill in
Step 2  listen to every clip and fill ratings.csv, without opening key.txt:
          rating  1 to 5, how much you like it as music (5 = best)
          guess   real or generated
Step 3  python scripts/listening_test.py score

Three sources, 6 clips each, 8 bars each
  real      excerpts of test songs (never seen in training), their real chords
  ours      chord HMM, AABA form of 2-bar phrases, no final cadence (real excerpts are cut mid-song,
            an ending would give ours away)
  control   the same generator, but notes depend only on the previous note, not on the chord
Everything else is identical: airy preset, tempo drawn in 99-136 bpm per clip, octave rule G4-D6,
legato notes, same fade out. So the notes are the only difference.

Usage: python scripts/listening_test.py make|score [--root=../POP909-Dataset/POP909]
"""

import base64
import csv
import json
import sys
from pathlib import Path

import numpy as np

from hmmusic.frames import N_CHORDS, beat_chords, note_sequences, onset_phases, phrase_bounds, split_songs, to_frames
from hmmusic.generate import RhythmModel, chord_blind_hmm, generate_form, real_excerpt
from hmmusic.hmm import HMM
from hmmusic.markov import MarkovChain
from hmmusic.pop909 import load_song, song_folders
from hmmusic.stats import permutation_test
from hmmusic.synth import PRESETS, render, write_wav

CLIPS_PER_SOURCE = 6
N_BARS = 8
OUT = Path("outputs/listening")


def make(root: Path) -> None:
    frames = {f.song_id: f for f in (to_frames(load_song(d)) for d in song_folders(root)) if f is not None}
    train_ids, _, test_ids = split_songs(list(frames), seed=0)
    train_notes = [s for i in train_ids for s in note_sequences(frames[i])]
    rhythm = RhythmModel().fit([p for i in train_ids for p in onset_phases(frames[i])], [s[2] for s in train_notes])
    chain = MarkovChain(N_CHORDS, 4, 1.0, "witten_bell").fit([c for i in train_ids for c in beat_chords(frames[i])])
    note_hmm = HMM.fit_supervised([s[0] for s in train_notes], [s[1] for s in train_notes], N_CHORDS, 12, 0.1, 1)
    control_hmm = chord_blind_hmm([s[0] for s in train_notes])

    rng = np.random.default_rng(2026)
    pieces = []

    # real: one excerpt per song, starting on a beat inside a phrase long enough for 8 bars
    n_frames = 16 * N_BARS
    for song_id in rng.permutation(sorted(test_ids)):
        f = frames[song_id]
        starts = []
        for a, b in phrase_bounds(f):
            first = f.first_beat + 4 * int(np.ceil((a - f.first_beat) / 4))  # first beat inside the phrase
            starts += list(range(first, b - n_frames + 2, 4))
        starts = [s for s in starts if (f.obs[s : s + n_frames] < 12).sum() >= 16]  # at least 16 notes
        if starts:
            pieces.append(("real", real_excerpt(f, int(rng.choice(starts)), N_BARS)))
        if len(pieces) == CLIPS_PER_SOURCE:
            break

    for source, hmm in (("ours", note_hmm), ("control", control_hmm)):
        for _ in range(CLIPS_PER_SOURCE):
            piece, _ = generate_form(rhythm, chain, hmm, rng, form="AABA", phrase_bars=2, min_changes=2,
                                     end_cadence=False)
            pieces.append((source, piece))

    OUT.mkdir(parents=True, exist_ok=True)
    order = rng.permutation(len(pieces))
    key = {}
    for k, i in enumerate(order, start=1):
        source, piece = pieces[i]
        bpm = int(rng.integers(99, 137))
        name = f"clip_{k:02d}"
        audio = render(piece.lead_notes(bpm), piece.chord_segments(bpm), bpm, PRESETS["airy"], tail=3.0, seed=k)
        write_wav(OUT / f"{name}.wav", audio)
        key[name] = source
    # encoded so that a glance at the file does not reveal the answers
    (OUT / "key.txt").write_text(base64.b64encode(json.dumps(key).encode()).decode())
    with open(OUT / "ratings.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip", "rating", "guess"])
        for name in sorted(key):
            writer.writerow([name, "", ""])
    print(f"{len(key)} clips in {OUT}/, fill {OUT}/ratings.csv (rating 1-5, guess real or generated)")
    print("do not open key.txt before scoring")


def score() -> None:
    key = json.loads(base64.b64decode((OUT / "key.txt").read_text()))
    with open(OUT / "ratings.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["rating"].strip()]
    missing = sorted(set(key) - {r["clip"] for r in rows})
    if len(missing) == len(key):
        print(f"no clip rated yet: fill the rating column of {OUT}/ratings.csv first")
        print("one line per clip, for example: clip_01,4,real")
        return
    if missing:
        print("not rated yet (left out):", ", ".join(missing))
    ratings = {s: np.array([float(r["rating"]) for r in rows if key[r["clip"]] == s]) for s in ("real", "ours", "control")}

    print(f"\n{'source':<9} {'n':>3} {'mean rating':>12} {'sd':>6}   guessed real")
    for source, values in ratings.items():
        guesses = [r["guess"].strip().lower() for r in rows if key[r["clip"]] == source]
        real_share = np.mean([g.startswith("r") for g in guesses]) if guesses else float("nan")
        sd = values.std(ddof=1) if len(values) > 1 else float("nan")
        print(f"{source:<9} {len(values):>3} {values.mean():>12.2f} {sd:>6.2f}   {100 * real_share:.0f}%")

    print()
    for a, b in (("ours", "control"), ("ours", "real"), ("real", "control")):
        if len(ratings[a]) and len(ratings[b]):
            p = permutation_test(ratings[a], ratings[b])
            print(f"{a} vs {b}: difference {ratings[a].mean() - ratings[b].mean():+.2f}, permutation p = {p:.3f}")

    print("\nper clip:")
    for r in sorted(rows, key=lambda r: r["clip"]):
        print(f"  {r['clip']}  {key[r['clip']]:<8} rating {r['rating']}  guess {r['guess']}")
    print("\ncaveat: one listener who built the model, 6 clips per source; a hint, not proof")


if __name__ == "__main__":
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
    command = next((a for a in sys.argv[1:] if not a.startswith("--")), "")
    if command == "make":
        make(Path(opts.get("root", "../POP909-Dataset/POP909")))
    elif command == "score":
        score()
    else:
        print(__doc__)
